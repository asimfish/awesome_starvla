"""Step-triggered probe hooks that append results to a JSONL file from any training loop (WP1).

A ``ProbeSchedule`` pairs a period with a callback ``fn(step) -> Mapping`` (a drift measurement, a
cross-head probe, a VLM capability evaluation such as RefCOCO IoU, ...). ``ProbeRunner.maybe_run(step)``
fires every due schedule at most once per step and writes one JSON line per firing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

ProbeFn = Callable[[int], Optional[Mapping[str, Any]]]


def to_jsonable(obj: Any) -> Any:
    """Recursively convert tensors / numpy values / paths into JSON-serialisable Python objects."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.dim() == 0 else obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


@dataclass
class ProbeSchedule:
    """Run ``fn(step)`` whenever ``step >= start_step`` and ``(step - start_step) % every_n_steps == 0``."""

    every_n_steps: int
    fn: ProbeFn
    name: str = ""
    start_step: int = 0

    def __post_init__(self) -> None:
        if self.every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1")
        if self.start_step < 0:
            raise ValueError("start_step must be >= 0")
        if not self.name:
            self.name = getattr(self.fn, "__name__", "") or "probe"

    def due(self, step: int) -> bool:
        return step >= self.start_step and (step - self.start_step) % self.every_n_steps == 0


class ProbeRunner:
    """Fire due ``ProbeSchedule``s at most once per step and log records to memory and optionally JSONL.

    Each record is ``{"step", "probe", "time", **static_fields, **fn(step)}``. Pass ``jsonl_path=None`` on
    non-main ranks (probes may still need to run on every rank) or ``enabled=False`` to skip entirely.
    """

    def __init__(
        self,
        schedules: Sequence[ProbeSchedule] = (),
        jsonl_path: Optional[Union[str, Path]] = None,
        static_fields: Optional[Mapping[str, Any]] = None,
        enabled: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.schedules: List[ProbeSchedule] = list(schedules)
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self.static_fields = dict(static_fields or {})
        self.enabled = enabled
        self.clock = clock
        self.records: List[Dict[str, Any]] = []
        self._last_step: Dict[int, int] = {}

    def add(self, schedule: ProbeSchedule) -> ProbeSchedule:
        self.schedules.append(schedule)
        return schedule

    def maybe_run(self, step: int) -> List[Dict[str, Any]]:
        """Run every schedule that is due at ``step`` and has not run at this step yet."""
        if not self.enabled:
            return []
        out: List[Dict[str, Any]] = []
        for idx, schedule in enumerate(self.schedules):
            if schedule.due(step) and self._last_step.get(idx) != step:
                out.append(self._run_one(idx, schedule, step))
        return out

    def run(self, step: int, only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """Force-run all schedules (or those named in ``only``) at ``step``, e.g. at the end of training."""
        if not self.enabled:
            return []
        wanted = set(only) if only is not None else None
        return [self._run_one(i, s, step) for i, s in enumerate(self.schedules) if wanted is None or s.name in wanted]

    def _run_one(self, idx: int, schedule: ProbeSchedule, step: int) -> Dict[str, Any]:
        metrics = schedule.fn(step) or {}
        record: Dict[str, Any] = {"step": int(step), "probe": schedule.name, "time": float(self.clock())}
        record.update(self.static_fields)
        record.update(to_jsonable(metrics))
        self._last_step[idx] = step
        self.records.append(record)
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        return record


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read every non-empty line of a JSONL file written by ``ProbeRunner``."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def install_hook_example(trainer: Any, runner: ProbeRunner, metric_prefix: str = "probe") -> Callable[[], None]:
    """Attach ``runner`` to a StarVLA trainer instance without editing StarVLA; returns an ``uninstall``.

    StarVLA's ``TrainStarVLA.train()`` (``starVLA/training/train_starvla.py``) calls
    ``self._log_metrics(step_metrics)`` once per micro-step after ``self.completed_steps`` has been
    advanced, so wrapping that bound method sees every optimizer step; the runner de-duplicates the
    repeated calls that gradient accumulation produces at one step. Scalar probe outputs are merged
    into ``step_metrics`` as ``"<metric_prefix>/<probe>/<key>"`` so they reach W&B with the training curves.

    Minimal example::

        trainer = TrainStarVLA(cfg)
        model = trainer.accelerator.unwrap_model(trainer.model)
        tracker = DriftTracker(extract_fn, probe_batch, reference=frozen_vlm_copy)

        def drift_probe(step):
            tracker.update(model, step)
            return tracker.summary()

        runner = ProbeRunner(
            [
                ProbeSchedule(500, drift_probe, name="drift"),
                ProbeSchedule(2000, lambda step: {"refcoco_iou": refcoco_iou(model)}, name="vlm_caps"),
            ],
            jsonl_path=f"{cfg.output_dir}/probes.jsonl" if trainer.accelerator.is_main_process else None,
        )
        uninstall = install_hook_example(trainer, runner)
        trainer.train()
        uninstall()
    """
    original = trainer._log_metrics

    def patched(metrics: Dict[str, Any]) -> Any:
        step = getattr(trainer, "completed_steps", None)
        if isinstance(step, int):
            for record in runner.maybe_run(step):
                for key, value in record.items():
                    if key not in ("step", "probe", "time") and isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[f"{metric_prefix}/{record['probe']}/{key}"] = value
        return original(metrics)

    trainer._log_metrics = patched

    def uninstall() -> None:
        trainer._log_metrics = original

    return uninstall
