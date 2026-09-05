#!/usr/bin/env python3
"""Summarise F2 frozen-backbone transfer runs (scripts/cluster/run_f2_transfer.sh).

    python scripts/analyze_f2.py --logs <dir with f2_<suite>_<head>_<backbone>.log> --out experiments/results/f2_frozen_backbone_transfer

Per (suite, head) the table compares the backbones on the fresh head's loss (first 50 / last 50 steps) and on StarVLA's
in-training MSE (``predict_action`` on a held-out training batch every ``eval_interval`` steps). Losses are comparable
across backbones within a head type only (L1 for OFT, flow-matching MSE for PI). Writes ``summary.md``,
``f2_metrics.csv`` (every logged point of every run) and ``f2_curves.png``.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_f0 import parse_trainer_log  # noqa: E402

RUN_RE = re.compile(r"^(?P<tag>[^_]+)_(?P<suite>[^_]+)_(?P<head>[^_]+)_(?P<backbone>[^_.]+)\.log$")


def window(rows: List[Dict], key: str, lo: int, hi: int) -> float:
    vals = [r[key] for r in rows if key in r and lo <= r["step"] <= hi]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True, help="directory containing the run logs")
    ap.add_argument("--tag", default="f2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone_order", default="pre,oft,mh")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs: Dict[Tuple[str, str, str], List[Dict]] = {}
    for path in sorted(Path(args.logs).glob(f"{args.tag}_*.log")):
        m = RUN_RE.match(path.name)
        if not m or m["tag"] != args.tag:
            continue
        rows = parse_trainer_log(path)
        if rows:
            runs[(m["suite"], m["head"], m["backbone"])] = rows
    if not runs:
        print("no runs found")
        return 1

    all_rows = []
    for (suite, head, backbone), rows in runs.items():
        for r in rows:
            all_rows.append({"suite": suite, "head": head, "backbone": backbone, **r})
    keys = sorted({k for r in all_rows for k in r if k not in ("suite", "head", "backbone", "step")})
    with (out / "f2_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "head", "backbone", "step"] + keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(all_rows, key=lambda r: (r["suite"], r["head"], r["backbone"], r["step"])))

    backbones = [b for b in args.backbone_order.split(",") if any(k[2] == b for k in runs)]
    backbones += sorted({k[2] for k in runs} - set(backbones))
    groups = sorted({(s, h) for s, h, _ in runs})
    lines = ["| suite | fresh head | metric | " + " | ".join(backbones) + " |", "|---|---|---|" + "---:|" * len(backbones)]
    for suite, head in groups:
        for label, fn in (
            ("loss steps 1-50", lambda rows: window(rows, "action_dit_loss", 1, 50)),
            ("loss last 50", lambda rows: window(rows, "action_dit_loss", 251, 10**9)),
            ("in-train MSE (last)", lambda rows: next((r["mse_score"] for r in sorted(rows, key=lambda r: -r["step"]) if "mse_score" in r), float("nan"))),
            ("in-train MSE (mean of evals)", lambda rows: window(rows, "mse_score", 1, 10**9)),
            ("s/step (median)", lambda rows: sorted(r["timing/model"] for r in rows if "timing/model" in r)[len([r for r in rows if "timing/model" in r]) // 2] if rows else float("nan")),
        ):
            cells = []
            for b in backbones:
                rows = runs.get((suite, head, b))
                v = fn(rows) if rows else float("nan")
                cells.append(f"{v:.4f}" if v == v else "-")
            lines.append(f"| {suite} | {head} | {label} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable ({type(exc).__name__}); skipping f2_curves.png")
        return 0
    fig, axes = plt.subplots(2, len(groups), figsize=(4.2 * len(groups), 7), squeeze=False)
    for j, (suite, head) in enumerate(groups):
        for b in backbones:
            rows = runs.get((suite, head, b))
            if not rows:
                continue
            axes[0][j].plot([r["step"] for r in rows if "action_dit_loss" in r], [r["action_dit_loss"] for r in rows if "action_dit_loss" in r], label=b)
            ev = [(r["step"], r["mse_score"]) for r in rows if "mse_score" in r]
            axes[1][j].plot([s for s, _ in ev], [v for _, v in ev], marker="o", label=b)
        axes[0][j].set_title(f"{suite} / fresh {head}: head loss"); axes[0][j].set_xlabel("step"); axes[0][j].legend(fontsize=8)
        axes[1][j].set_title("in-train MSE (predict_action)"); axes[1][j].set_xlabel("step"); axes[1][j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "f2_curves.png", dpi=130)
    print(f"[done] wrote {out}/summary.md, f2_metrics.csv, f2_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
