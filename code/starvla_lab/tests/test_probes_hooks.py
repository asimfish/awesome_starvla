import json

import numpy as np
import pytest
import torch

from starvla_lab.probes import ProbeRunner, ProbeSchedule, install_hook_example, read_jsonl, to_jsonable


def test_schedule_due_and_validation():
    sched = ProbeSchedule(every_n_steps=3, fn=lambda step: {})
    assert [s for s in range(10) if sched.due(s)] == [0, 3, 6, 9]
    late = ProbeSchedule(4, lambda step: {}, start_step=5)
    assert [s for s in range(15) if late.due(s)] == [5, 9, 13]
    assert late.name == "<lambda>"

    def refcoco_iou(step):
        return {"iou": 0.5}

    assert ProbeSchedule(1, refcoco_iou).name == "refcoco_iou"
    with pytest.raises(ValueError):
        ProbeSchedule(0, refcoco_iou)
    with pytest.raises(ValueError):
        ProbeSchedule(1, refcoco_iou, start_step=-1)


def test_runner_fires_once_per_step_and_jsonl_round_trips(tmp_path):
    calls = {"drift": [], "vlm": []}

    def drift_probe(step):
        calls["drift"].append(step)
        return {"per_layer": torch.tensor([0.1, 0.2]), "mean": torch.tensor(0.15), "np": np.float32(1.5)}

    def vlm_probe(step):
        calls["vlm"].append(step)
        return {"refcoco_iou": 0.42, "nested": {"a": np.arange(2)}}

    path = tmp_path / "logs" / "probes.jsonl"
    runner = ProbeRunner(
        [ProbeSchedule(2, drift_probe, name="drift"), ProbeSchedule(3, vlm_probe, name="vlm")],
        jsonl_path=path,
        static_fields={"run": "R5"},
        clock=lambda: 123.0,
    )
    for step in range(7):
        for _ in range(2):  # gradient accumulation calls the hook twice per optimizer step
            runner.maybe_run(step)
    assert calls["drift"] == [0, 2, 4, 6] and calls["vlm"] == [0, 3, 6]

    records = read_jsonl(path)
    assert records == runner.records and len(records) == 7
    drift_records = [r for r in records if r["probe"] == "drift"]
    assert drift_records[0] == {
        "step": 0,
        "probe": "drift",
        "time": 123.0,
        "run": "R5",
        "per_layer": [pytest.approx(0.1), pytest.approx(0.2)],
        "mean": pytest.approx(0.15),
        "np": 1.5,
    }
    vlm_records = [r for r in records if r["probe"] == "vlm"]
    assert vlm_records[-1]["step"] == 6 and vlm_records[-1]["nested"] == {"a": [0, 1]}
    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_runner_disabled_force_run_and_add():
    fired = []
    runner = ProbeRunner(enabled=False, jsonl_path=None)
    runner.add(ProbeSchedule(1, lambda step: fired.append(step) or {"x": 1}, name="a"))
    runner.add(ProbeSchedule(1, lambda step: {"y": 2}, name="b"))
    assert runner.maybe_run(0) == [] and runner.run(0) == [] and fired == []
    runner.enabled = True
    assert [r["probe"] for r in runner.run(7, only=["b"])] == ["b"]
    assert [r["probe"] for r in runner.run(8)] == ["a", "b"]
    assert fired == [8]
    assert runner.maybe_run(8) == []  # already ran at step 8
    assert len(runner.maybe_run(9)) == 2
    assert runner.records[-1]["probe"] == "b" and runner.records[-1]["y"] == 2


def test_to_jsonable_covers_common_types(tmp_path):
    out = to_jsonable({"t": torch.ones(2, 2), "s": torch.tensor(3), "n": np.int64(4), "p": tmp_path, 1: (1, 2), "set": {3}})
    assert out == {"t": [[1.0, 1.0], [1.0, 1.0]], "s": 3, "n": 4, "p": str(tmp_path), "1": [1, 2], "set": [3]}
    json.dumps(out)


class _MockTrainer:
    """Mimics the parts of StarVLA's ``TrainStarVLA`` that ``install_hook_example`` touches."""

    def __init__(self, grad_accum=2):
        self.completed_steps = 0
        self.grad_accum = grad_accum
        self.logged = []

    def _log_metrics(self, metrics):
        self.logged.append(dict(metrics))

    def train(self, steps):
        for micro in range(steps * self.grad_accum):
            if (micro + 1) % self.grad_accum == 0:
                self.completed_steps += 1
            self._log_metrics({"loss": 1.0})


def test_install_hook_example_runs_probes_inside_mock_starvla_loop(tmp_path):
    trainer = _MockTrainer()
    runner = ProbeRunner(
        [ProbeSchedule(2, lambda step: {"mean": 0.25 * step, "per_layer": [0.1, 0.2], "flag": True}, name="drift")],
        jsonl_path=tmp_path / "p.jsonl",
    )
    uninstall = install_hook_example(trainer, runner)
    trainer.train(steps=5)
    assert [r["step"] for r in runner.records] == [0, 2, 4]
    assert [r["step"] for r in read_jsonl(tmp_path / "p.jsonl")] == [0, 2, 4]
    merged = [m for m in trainer.logged if "probe/drift/mean" in m]
    assert [m["probe/drift/mean"] for m in merged] == [0.0, 0.5, 1.0]
    assert all("probe/drift/per_layer" not in m and "probe/drift/flag" not in m for m in merged)
    assert len(trainer.logged) == 10 and all(m["loss"] == 1.0 for m in trainer.logged)
    uninstall()
    trainer.train(steps=2)
    assert [r["step"] for r in runner.records] == [0, 2, 4]
