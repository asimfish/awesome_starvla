"""Backbone benchmark protocol (WP5): swap only the backbone, hold everything else fixed.

The protocol turns a list of backbone checkpoints into a deterministic run matrix
(backbone x benchmark x data-fraction x seed) whose StarVLA launch commands differ
*only* in the backbone initialisation and the run id. Results are written by the
runner as one JSON per run under ``results_dir`` and aggregated by ``summarize``.
"""
from __future__ import annotations

import csv
import json
import shlex
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "BackboneSpec",
    "BenchmarkSpec",
    "Protocol",
    "RunSpec",
    "build_runs",
    "render_commands",
    "write_matrix_csv",
    "read_matrix_csv",
    "summarize_results",
    "format_summary_table",
]


@dataclass(frozen=True)
class BackboneSpec:
    """A backbone initialisation to evaluate.

    ``init`` is either a HF model id / local VLM directory (fresh VLM) or a StarVLA
    checkpoint ``*.pt``; ``reload_modules`` selects which submodules to load from a
    checkpoint (VLAct protocol: backbone only, heads re-initialised).
    """

    name: str
    init: str
    reload_modules: str = "qwen_vl_interface"

    @property
    def is_checkpoint(self) -> bool:
        return self.init.endswith(".pt")


@dataclass(frozen=True)
class BenchmarkSpec:
    """A downstream benchmark with its StarVLA training yaml and evaluation entry."""

    name: str
    train_yaml: str
    eval_script: str
    metric_keys: Sequence[str] = ("success_rate",)
    data_fractions: Sequence[float] = (1.0,)
    num_gpus: int = 8
    extra_overrides: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Protocol:
    """Everything that must stay fixed while the backbone varies."""

    head: str = "QwenOFT"
    max_steps: int = 30000
    per_device_batch_size: int = 16
    learning_rate_backbone: float = 1e-5
    learning_rate_head: float = 1e-4
    seeds: Sequence[int] = (0, 1, 2)
    checkpoint_rule: str = "last"
    train_script: str = "starVLA/training/train_starvla.py"
    accelerate_config: str = "starVLA/config/deepseeds/deepspeed_zero2.yaml"
    run_root_dir: str = "playground/Checkpoints/backbone_bench"

    def fixed_overrides(self) -> Dict[str, str]:
        return {
            "framework.name": self.head,
            "trainer.max_train_steps": str(self.max_steps),
            "datasets.vla_data.per_device_batch_size": str(self.per_device_batch_size),
            "trainer.learning_rate.qwen_vl_interface": f"{self.learning_rate_backbone:g}",
            "trainer.learning_rate.action_model": f"{self.learning_rate_head:g}",
            "trainer.freeze_modules": "",
            "run_root_dir": self.run_root_dir,
        }


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    backbone: str
    benchmark: str
    data_fraction: float
    seed: int
    overrides: Mapping[str, str]
    train_yaml: str
    eval_script: str
    num_gpus: int
    metric_keys: Sequence[str]
    status: str = "planned"

    def to_row(self) -> Dict[str, str]:
        row = {k: v for k, v in asdict(self).items() if k not in ("overrides", "metric_keys")}
        row["overrides"] = json.dumps(dict(self.overrides), sort_keys=True)
        row["metric_keys"] = ",".join(self.metric_keys)
        row["data_fraction"] = f"{self.data_fraction:g}"
        return {k: str(v) for k, v in row.items()}


def _run_id(backbone: BackboneSpec, bench: BenchmarkSpec, frac: float, seed: int) -> str:
    frac_tag = "" if frac == 1.0 else f"_f{int(round(frac * 100)):03d}"
    return f"bb-{backbone.name}__{bench.name}{frac_tag}__s{seed}"


def build_runs(protocol: Protocol, backbones: Sequence[BackboneSpec], benchmarks: Sequence[BenchmarkSpec]) -> List[RunSpec]:
    """Cartesian product backbone x benchmark x fraction x seed, deterministic order."""
    runs: List[RunSpec] = []
    fixed = protocol.fixed_overrides()
    for bench in benchmarks:
        for backbone in backbones:
            for frac in bench.data_fractions:
                for seed in protocol.seeds:
                    overrides = dict(fixed)
                    overrides.update(bench.extra_overrides)
                    overrides["seed"] = str(seed)
                    if frac != 1.0:
                        overrides["datasets.vla_data.data_fraction"] = f"{frac:g}"
                    if backbone.is_checkpoint:
                        overrides["trainer.pretrained_checkpoint"] = backbone.init
                        overrides["trainer.reload_modules"] = backbone.reload_modules
                    else:
                        overrides["framework.qwenvl.base_vlm"] = backbone.init
                    rid = _run_id(backbone, bench, frac, seed)
                    overrides["run_id"] = rid
                    runs.append(
                        RunSpec(
                            run_id=rid,
                            backbone=backbone.name,
                            benchmark=bench.name,
                            data_fraction=frac,
                            seed=seed,
                            overrides=overrides,
                            train_yaml=bench.train_yaml,
                            eval_script=bench.eval_script,
                            num_gpus=bench.num_gpus,
                            metric_keys=tuple(bench.metric_keys),
                        )
                    )
    return runs


def render_commands(run: RunSpec, protocol: Protocol, starvla_root: str = ".") -> Dict[str, str]:
    """Render the StarVLA train / eval shell commands for one run."""
    override_args = " ".join(f"--{k} {shlex.quote(v)}" for k, v in sorted(run.overrides.items()))
    train = (
        f"cd {shlex.quote(starvla_root)} && accelerate launch "
        f"--config_file {protocol.accelerate_config} --num_processes {run.num_gpus} "
        f"{protocol.train_script} --config_yaml {run.train_yaml} {override_args}"
    )
    ckpt_dir = f"{protocol.run_root_dir}/{run.run_id}"
    eval_cmd = f"cd {shlex.quote(starvla_root)} && bash {run.eval_script} {ckpt_dir} {run.seed}"
    return {"train": train, "eval": eval_cmd, "checkpoint_dir": ckpt_dir}


def varying_keys(runs: Sequence[RunSpec]) -> List[str]:
    """Override keys that differ across runs (protocol audit: should be backbone/seed/run_id/fraction only)."""
    keys = sorted({k for r in runs for k in r.overrides})
    out = []
    for k in keys:
        values = {r.overrides.get(k) for r in runs}
        if len(values) > 1:
            out.append(k)
    return out


_CSV_FIELDS = ["run_id", "backbone", "benchmark", "data_fraction", "seed", "status", "train_yaml", "eval_script", "num_gpus", "metric_keys", "overrides"]


def write_matrix_csv(runs: Sequence[RunSpec], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.to_row()[k] for k in _CSV_FIELDS})
    return path


def read_matrix_csv(path: str | Path) -> List[RunSpec]:
    runs = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            runs.append(
                RunSpec(
                    run_id=row["run_id"],
                    backbone=row["backbone"],
                    benchmark=row["benchmark"],
                    data_fraction=float(row["data_fraction"]),
                    seed=int(row["seed"]),
                    overrides=json.loads(row["overrides"]),
                    train_yaml=row["train_yaml"],
                    eval_script=row["eval_script"],
                    num_gpus=int(row["num_gpus"]),
                    metric_keys=tuple(row["metric_keys"].split(",")),
                    status=row.get("status", "planned"),
                )
            )
    return runs


def summarize_results(results_dir: str | Path, metric: str = "success_rate") -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate ``<results_dir>/<run_id>.json`` files into backbone -> benchmark -> {mean, std, n}.

    Each result JSON must contain ``backbone``, ``benchmark`` and the metric; an optional
    ``data_fraction`` is folded into the benchmark key as ``<benchmark>@<fraction>``.
    """
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for p in sorted(Path(results_dir).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if metric not in d:
            continue
        bench = d["benchmark"]
        frac = float(d.get("data_fraction", 1.0))
        if frac != 1.0:
            bench = f"{bench}@{frac:g}"
        buckets.setdefault(d["backbone"], {}).setdefault(bench, []).append(float(d[metric]))
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for bb, benches in buckets.items():
        out[bb] = {}
        for bench, vals in benches.items():
            out[bb][bench] = {
                "mean": statistics.fmean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n": float(len(vals)),
            }
    return out


def format_summary_table(summary: Mapping[str, Mapping[str, Mapping[str, float]]]) -> str:
    """Markdown table: rows = backbones, columns = benchmarks, cells = mean ± std (n)."""
    benches = sorted({b for row in summary.values() for b in row})
    lines = ["| backbone | " + " | ".join(benches) + " |", "|---|" + "---|" * len(benches)]
    for bb in sorted(summary):
        cells = []
        for b in benches:
            s = summary[bb].get(b)
            cells.append("–" if s is None else f"{s['mean']:.1f} ± {s['std']:.1f} (n={int(s['n'])})")
        lines.append(f"| {bb} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
