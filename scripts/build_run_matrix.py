#!/usr/bin/env python3
"""Generate experiments/run_matrix.csv from starvla_lab/configs (protocol_f1.yaml + matrix_R0_R9.yaml).

Usage: python3 scripts/build_run_matrix.py [--print-commands N]
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from starvla_lab.bench import BackboneSpec, BenchmarkSpec, Protocol, build_runs, render_commands, varying_keys, write_matrix_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-commands", type=int, default=0, help="print the first N train/eval commands")
    args = ap.parse_args()

    cfg_dir = ROOT / "code" / "starvla_lab" / "configs"
    f1 = yaml.safe_load((cfg_dir / "protocol_f1.yaml").read_text(encoding="utf-8"))
    matrix = yaml.safe_load((cfg_dir / "matrix_R0_R9.yaml").read_text(encoding="utf-8"))

    p = f1["protocol"]
    protocol = Protocol(
        head=p["head"], max_steps=p["max_steps"], per_device_batch_size=p["per_device_batch_size"],
        learning_rate_backbone=float(p["learning_rate_backbone"]), learning_rate_head=float(p["learning_rate_head"]),
        seeds=tuple(p["seeds"]), checkpoint_rule=p["checkpoint_rule"], train_script=p["train_script"],
        accelerate_config=p["accelerate_config"], run_root_dir=p["run_root_dir"],
    )
    benchmarks = [
        BenchmarkSpec(b["name"], b["train_yaml"], b["eval_script"], tuple(b.get("metric_keys", ["success_rate"])),
                      tuple(b.get("data_fractions", [1.0])), int(b.get("num_gpus", 8)))
        for b in f1["benchmarks"]
    ]
    ckpt_glob = matrix["pretrain_common"]["checkpoint_glob"]
    backbones = []
    for rid, v in matrix["variants"].items():
        if v.get("protocol") == "eventvla":
            continue  # R9 uses the EventVLA protocol, not F1
        init = v.get("backbone_init") or ckpt_glob.format(run_id=f"pretrain_{rid}")
        backbones.append(BackboneSpec(rid, init))

    runs = build_runs(protocol, backbones, benchmarks)
    out = write_matrix_csv(runs, ROOT / "experiments" / "run_matrix.csv")
    pre_hours = sum(int(v.get("est_gpu_hours", 0)) for v in matrix["variants"].values())
    print(f"wrote {out} with {len(runs)} downstream runs "
          f"({len(backbones)} backbones x {sum(len(b.data_fractions) for b in benchmarks)} benchmark settings x {len(protocol.seeds)} seeds)")
    print(f"pre-training budget (16xA100 GPU-hours, from matrix): {pre_hours}")
    print("keys that vary across runs:", varying_keys(runs))
    for r in runs[: args.print_commands]:
        c = render_commands(r, protocol, starvla_root="<StarVLA>")
        print(f"\n# {r.run_id}\n{c['train']}\n{c['eval']}")


if __name__ == "__main__":
    main()
