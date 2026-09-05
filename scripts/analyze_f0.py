#!/usr/bin/env python3
"""Summarise F0 fine-tuning smoke runs: loss curves from StarVLA trainer logs + layer drift from probe JSONL.

    python scripts/analyze_f0.py --run f0_oft LOG JSONL --run f0_multihead LOG JSONL --out experiments/results/f0_libero_goal_smoke

StarVLA logs each metrics dict through ``rich`` (wrapped over several lines after ``Step N, Loss:``); the
parser re-joins those lines and ``ast.literal_eval``s the dict. Outputs per run ``<name>_metrics.csv`` and
``<name>_drift.csv`` (probe step x layer), plus ``summary.md`` and ``f0_curves.png``.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

STEP_RE = re.compile(r"Step (\d+), Loss:")


def parse_trainer_log(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        m = STEP_RE.search(lines[i])
        if not m:
            i += 1
            continue
        step = int(m.group(1))
        buf, j = [], i + 1
        while j < len(lines) and "})" not in buf[-1] if buf else True:
            # rich prefixes every wrapped line with spaces; tqdm progress lines can be interleaved -> skip them
            line = lines[j]
            if "%|" not in line and "it/s" not in line:
                buf.append(line.strip())
            j += 1
            if buf and "})" in buf[-1]:
                break
        text = " ".join(buf)
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                d = ast.literal_eval(text[start:end + 1])
                d = {k: v for k, v in d.items() if isinstance(v, (int, float))}
                d["step"] = step
                rows.append(d)
            except (ValueError, SyntaxError):
                pass
        i = j
    return rows


def parse_probes(path: Optional[Path]) -> List[dict]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarise_run(name: str, metrics: List[Dict], probes: List[dict], out: Path) -> Dict[str, object]:
    keys = sorted({k for r in metrics for k in r if k != "step"})
    write_csv(out / f"{name}_metrics.csv", sorted(metrics, key=lambda r: r["step"]), ["step"] + keys)

    drift_rows = []
    for rec in probes:
        per_layer = rec["drift"]["per_layer"]
        values = list(per_layer.values())
        n_frozen = min(18, len(values))  # F0 freezes LLM layers 0-17
        row = {
            "step": rec["step"], "mean": rec["drift"]["mean"], "max": rec["drift"]["max"], "max_layer": rec["drift"]["max_layer"],
            "frozen_mean": sum(values[:n_frozen]) / max(n_frozen, 1),
            "trainable_mean": sum(values[n_frozen:]) / max(len(values) - n_frozen, 1),
            "secondary_mean": rec.get("drift_secondary", {}).get("mean", float("nan")),
            "secondary_max": rec.get("drift_secondary", {}).get("max", float("nan")),
            "embed_changed_rows": rec.get("embed_tokens", {}).get("changed_rows", float("nan")),
            "embed_rel_change": rec.get("embed_tokens", {}).get("relative_frobenius_change", float("nan")),
        }
        row.update(per_layer)
        drift_rows.append(row)
    if drift_rows:
        layer_keys = [k for k in drift_rows[0] if k.startswith("layer_")]
        head = ["step", "mean", "max", "max_layer", "frozen_mean", "trainable_mean", "secondary_mean", "secondary_max", "embed_changed_rows", "embed_rel_change"]
        write_csv(out / f"{name}_drift.csv", drift_rows, head + layer_keys)

    loss_key = "action_dit_loss"
    losses = [(r["step"], r[loss_key]) for r in metrics if loss_key in r]
    head_losses = {k: [(r["step"], r[k]) for r in metrics if k in r] for k in keys if k.startswith("lab/loss_")}

    def window(series, lo, hi):
        vals = [v for s, v in series if lo <= s <= hi]
        return sum(vals) / len(vals) if vals else float("nan")

    last = max(s for s, _ in losses) if losses else 0
    # timing/model of a logged step includes the probe forward passes when a probe fired at that step, so the
    # per-step cost is the median; the probe overhead is reported separately from the steps that carry it.
    step_times = sorted(r["timing/model"] for r in metrics if "timing/model" in r and 1 <= r["step"] <= last)
    median_step = step_times[len(step_times) // 2] if step_times else float("nan")
    probe_steps = {p["step"] for p in probes}
    probe_times = [r["timing/model"] - median_step for r in metrics if "timing/model" in r and r["step"] in probe_steps and r["step"] > 0]
    summary = {
        "name": name,
        "steps_logged": len(losses),
        "loss_first": losses[0][1] if losses else float("nan"),
        "loss_steps_1_50": window(losses, 1, 50),
        "loss_last_50": window(losses, max(1, last - 49), last),
        "sec_per_step": median_step,
        "probe_overhead_s": sum(probe_times) / len(probe_times) if probe_times else float("nan"),
        "mse_last": next((r["mse_score"] for r in sorted(metrics, key=lambda r: -r["step"]) if "mse_score" in r), float("nan")),
        "head_losses_last_50": {k: window(v, max(1, last - 49), last) for k, v in head_losses.items()},
        "drift": [(r["step"], r["mean"], r["max"], r["max_layer"], r["frozen_mean"], r["trainable_mean"], r["secondary_mean"], r["embed_changed_rows"]) for r in drift_rows],
    }
    return summary


def plot(runs: Dict[str, Dict], out: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # missing, or a matplotlib/numpy ABI mismatch: the tables are still written
        print(f"[warn] matplotlib unavailable ({type(exc).__name__}); skipping f0_curves.png")
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, (metrics, probes) in runs.items():
        xs = [r["step"] for r in metrics if "action_dit_loss" in r]
        ys = [r["action_dit_loss"] for r in metrics if "action_dit_loss" in r]
        axes[0].plot(xs, ys, label=name, alpha=0.85)
        for k in sorted({k for r in metrics for k in r if k.startswith("lab/loss_")}):
            axes[1].plot([r["step"] for r in metrics if k in r], [r[k] for r in metrics if k in r], label=f"{name}:{k[4:]}", alpha=0.85)
        if probes:
            steps = [p["step"] for p in probes]
            axes[2].plot(steps, [p["drift"]["mean"] for p in probes], marker="o", label=f"{name} mean")
            axes[2].plot(steps, [p["drift"]["max"] for p in probes], marker="x", linestyle="--", label=f"{name} max")
            if all("drift_secondary" in p for p in probes):
                axes[2].plot(steps, [p["drift_secondary"]["mean"] for p in probes], marker=".", linestyle=":", alpha=0.7, label=f"{name} secondary mean")
    axes[0].set_title("action loss (logged every 10 steps)"); axes[0].set_xlabel("step"); axes[0].legend()
    axes[1].set_title("per-head losses (QwenMultiHead)"); axes[1].set_xlabel("step")
    if axes[1].lines:
        axes[1].legend(fontsize=8)
    axes[2].set_title("backbone drift 1-CKA vs. pretrained VLM"); axes[2].set_xlabel("completed updates"); axes[2].set_yscale("symlog", linthresh=1e-4); axes[2].legend(fontsize=7)
    fig.tight_layout()
    path = out / "f0_curves.png"
    fig.savefig(path, dpi=130)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=3, action="append", metavar=("NAME", "LOG", "JSONL"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs, summaries = {}, []
    for name, log, jsonl in args.run:
        metrics = parse_trainer_log(Path(log))
        probes = parse_probes(Path(jsonl) if jsonl and jsonl != "-" else None)
        runs[name] = (metrics, probes)
        summaries.append(summarise_run(name, metrics, probes, out))

    lines = ["| run | logged pts | loss steps 1-50 | loss last 50 | in-train MSE (last) | s/step (median) | probe overhead s | per-head loss (last 50) |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for s in summaries:
        heads = ", ".join(f"{k[9:]}={v:.3f}" for k, v in s["head_losses_last_50"].items()) or "-"
        po = f"{s['probe_overhead_s']:.1f}" if s["probe_overhead_s"] == s["probe_overhead_s"] else "-"
        lines.append(f"| {s['name']} | {s['steps_logged']} | {s['loss_steps_1_50']:.3f} | {s['loss_last_50']:.3f} | {s['mse_last']:.4f} | {s['sec_per_step']:.2f} | {po} | {heads} |")
    lines += ["", "| run | updates | drift mean | frozen L0-17 | trainable L18-35 | max (layer) | secondary mean | embed rows changed |",
              "|---|---:|---:|---:|---:|---|---:|---:|"]
    for s in summaries:
        for step, mean, mx, layer, frozen, trainable, secondary, rows in s["drift"]:
            sec = f"{secondary:.5f}" if secondary == secondary else "-"
            emb = f"{int(rows)}" if rows == rows else "-"
            lines.append(f"| {s['name']} | {step} | {mean:.5f} | {frozen:.5f} | {trainable:.5f} | {mx:.5f} ({layer}) | {sec} | {emb} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    fig = plot(runs, out)
    print(f"\n[done] wrote {out}/summary.md" + (f" and {fig.name}" if fig else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
