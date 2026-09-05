import json
from pathlib import Path

import pytest

from starvla_lab.bench import (
    BackboneSpec,
    BenchmarkSpec,
    Protocol,
    build_runs,
    format_summary_table,
    read_matrix_csv,
    render_commands,
    summarize_results,
    varying_keys,
    write_matrix_csv,
)

BACKBONES = [
    BackboneSpec("qwen3vl", "Qwen/Qwen3-VL-4B-Instruct"),
    BackboneSpec("vlact", "/ckpt/vlact/steps_100000_pytorch_model.pt"),
    BackboneSpec("oft_only", "/ckpt/oft_only/steps_100000_pytorch_model.pt"),
]
BENCHES = [
    BenchmarkSpec("libero_plus", "examples/simBenchmarks/LIBERO-plus/train_files/x.yaml", "examples/simBenchmarks/LIBERO-plus/eval_files/eval.sh"),
    BenchmarkSpec("robocasa_gr1", "examples/simBenchmarks/Robocasa_tabletop/train_files/x.yaml", "examples/simBenchmarks/Robocasa_tabletop/eval_files/eval.sh", data_fractions=(0.2, 1.0)),
]
PROTO = Protocol(seeds=(0, 1))


def test_matrix_size_and_ids_unique():
    runs = build_runs(PROTO, BACKBONES, BENCHES)
    assert len(runs) == 3 * (1 + 2) * 2
    assert len({r.run_id for r in runs}) == len(runs)
    assert any("_f020" in r.run_id for r in runs)


def test_only_backbone_seed_fraction_vary():
    runs = build_runs(PROTO, BACKBONES, [BENCHES[0]])
    keys = set(varying_keys(runs))
    assert keys <= {"framework.qwenvl.base_vlm", "trainer.pretrained_checkpoint", "trainer.reload_modules", "seed", "run_id"}
    for r in runs:
        assert r.overrides["framework.name"] == "QwenOFT"
        assert r.overrides["trainer.freeze_modules"] == ""


def test_checkpoint_vs_fresh_vlm_overrides():
    runs = build_runs(PROTO, BACKBONES, [BENCHES[0]])
    fresh = [r for r in runs if r.backbone == "qwen3vl"][0]
    ckpt = [r for r in runs if r.backbone == "vlact"][0]
    assert "framework.qwenvl.base_vlm" in fresh.overrides and "trainer.pretrained_checkpoint" not in fresh.overrides
    assert ckpt.overrides["trainer.pretrained_checkpoint"].endswith(".pt")
    assert ckpt.overrides["trainer.reload_modules"] == "qwen_vl_interface"


def test_render_commands_contains_overrides_and_eval():
    run = build_runs(PROTO, BACKBONES[:1], BENCHES[:1])[0]
    cmd = render_commands(run, PROTO, starvla_root="/opt/starVLA")
    assert cmd["train"].startswith("cd /opt/starVLA && accelerate launch")
    assert "--framework.name QwenOFT" in cmd["train"] and f"--run_id {run.run_id}" in cmd["train"]
    assert cmd["eval"].endswith(f"{PROTO.run_root_dir}/{run.run_id} 0")


def test_csv_roundtrip(tmp_path: Path):
    runs = build_runs(PROTO, BACKBONES, BENCHES)
    p = write_matrix_csv(runs, tmp_path / "m.csv")
    back = read_matrix_csv(p)
    assert [r.run_id for r in back] == [r.run_id for r in runs]
    assert back[0].overrides == dict(runs[0].overrides)
    assert back[-1].data_fraction == runs[-1].data_fraction


def test_summarize_results_mean_std(tmp_path: Path):
    for i, v in enumerate([80.0, 82.0, 84.0]):
        (tmp_path / f"r{i}.json").write_text(json.dumps({"backbone": "vlact", "benchmark": "libero_plus", "success_rate": v}))
    (tmp_path / "g.json").write_text(json.dumps({"backbone": "vlact", "benchmark": "robocasa_gr1", "data_fraction": 0.2, "success_rate": 49.5}))
    (tmp_path / "skip.json").write_text(json.dumps({"backbone": "x", "benchmark": "y"}))
    s = summarize_results(tmp_path)
    assert s["vlact"]["libero_plus"]["mean"] == pytest.approx(82.0)
    assert s["vlact"]["libero_plus"]["std"] == pytest.approx(2.0)
    assert s["vlact"]["robocasa_gr1@0.2"]["n"] == 1
    table = format_summary_table(s)
    assert "| vlact |" in table and "82.0 ± 2.0 (n=3)" in table
