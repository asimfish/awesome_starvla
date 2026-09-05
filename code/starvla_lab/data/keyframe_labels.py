"""Keyframe (task-critical event) labels for ``KeyframeHead`` (WP3b data preparation).

Two label sources share one on-disk format ``<cache_dir>/<trajectory_id>.json`` = ``{"events": [t0, t1, ...]}``:

* :func:`heuristic_keyframe_steps` — a label-free baseline: steps where a gripper dimension crosses its
  open/close threshold (grasp / release events) plus optional velocity-minimum ("pause") events.
* :class:`FunctionLabeler` — wraps any ``label_fn(frames, instruction) -> list[int]`` (e.g. an offline
  Qwen3-VL prompt as in EventVLA) so VLM labels drop into the same cache.

:class:`KeyframeLabelTransform` converts absolute event steps into chunk-relative ``keyframe_steps`` for a
sample at step ``t`` with horizon ``H`` (events outside ``[t, t+H)`` are dropped; a sample whose trajectory
has no label file is left without the field, i.e. masked by ``QwenMultiHeadLab``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["heuristic_keyframe_steps", "FunctionLabeler", "save_labels", "load_labels", "KeyframeLabelTransform", "chunk_relative_events"]


def heuristic_keyframe_steps(
    actions: np.ndarray,
    gripper_dims: Sequence[int],
    threshold: float = 0.5,
    min_gap: int = 5,
    pause_dims: Optional[Sequence[int]] = None,
    pause_quantile: float = 0.05,
) -> List[int]:
    """Event steps from gripper threshold crossings (and optional motion pauses) in ``actions [T, D]``."""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("actions must be [T, D]")
    events: List[int] = []
    for d in gripper_dims:
        closed = a[:, d] > threshold
        change = np.flatnonzero(closed[1:] != closed[:-1]) + 1
        events.extend(int(i) for i in change)
    if pause_dims:
        vel = np.linalg.norm(np.diff(a[:, list(pause_dims)], axis=0), axis=1)
        if vel.size:
            thr = np.quantile(vel, pause_quantile)
            idx = np.flatnonzero(vel <= thr) + 1
            events.extend(int(i) for i in idx)
    events = sorted(set(events))
    merged: List[int] = []
    for e in events:
        if not merged or e - merged[-1] >= min_gap:
            merged.append(e)
    return merged


class FunctionLabeler:
    """Adapter for VLM / rule labelers: ``label_fn(frames, instruction) -> Iterable[int]`` (absolute steps)."""

    def __init__(self, label_fn: Callable[[Sequence[Any], str], Iterable[int]], min_gap: int = 1) -> None:
        self.label_fn, self.min_gap = label_fn, int(min_gap)

    def __call__(self, frames: Sequence[Any], instruction: str) -> List[int]:
        raw = sorted({int(t) for t in self.label_fn(frames, instruction) if 0 <= int(t) < len(frames)})
        out: List[int] = []
        for t in raw:
            if not out or t - out[-1] >= self.min_gap:
                out.append(t)
        return out


def save_labels(cache_dir: str | Path, trajectory_id: Any, events: Sequence[int], meta: Optional[Mapping[str, Any]] = None) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{trajectory_id}.json"
    path.write_text(json.dumps({"events": [int(e) for e in events], **({"meta": dict(meta)} if meta else {})}), encoding="utf-8")
    return path


def load_labels(cache_dir: str | Path, trajectory_id: Any) -> Optional[List[int]]:
    path = Path(cache_dir) / f"{trajectory_id}.json"
    if not path.exists():
        return None
    return [int(e) for e in json.loads(path.read_text(encoding="utf-8"))["events"]]


def chunk_relative_events(events: Sequence[int], t: int, horizon: int) -> List[int]:
    """Absolute event steps -> offsets in ``[0, horizon)`` relative to chunk start ``t``."""
    return [int(e) - int(t) for e in events if int(t) <= int(e) < int(t) + int(horizon)]


class KeyframeLabelTransform:
    """Sample-level transform adding ``keyframe_steps`` (chunk-relative) from cached absolute event labels."""

    def __init__(self, cache_dir: str | Path, horizon: int, traj_key: str = "trajectory_id", step_key: str = "step") -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.cache_dir, self.horizon, self.traj_key, self.step_key = Path(cache_dir), int(horizon), traj_key, step_key
        self._cache: Dict[str, Optional[List[int]]] = {}

    def _events(self, traj: Any) -> Optional[List[int]]:
        key = str(traj)
        if key not in self._cache:
            self._cache[key] = load_labels(self.cache_dir, key)
        return self._cache[key]

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(sample)
        traj, step = sample.get(self.traj_key), sample.get(self.step_key)
        if traj is None or step is None:
            return out
        events = self._events(traj)
        if events is None:
            return out
        out["keyframe_steps"] = chunk_relative_events(events, int(step), self.horizon)
        return out
