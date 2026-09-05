"""Offline future-frame feature cache for ``FutureFeaturePredictionHead`` (WP3b data preparation).

Pipeline: a frozen feature extractor (SigLIP / DINO pooled features, injected as a callable) is run once
per trajectory and cached as ``<cache_dir>/<trajectory_id>.npy`` with shape ``[T, d_feat]``. At training
time :class:`FutureFeatureTransform` looks up the trajectory / step of each sample and attaches
``future_features`` ``[len(offsets), d_feat]`` and ``future_features_mask`` ``[len(offsets)]`` (0 where the
future frame does not exist), which is exactly what ``QwenMultiHeadLab`` reads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["extract_trajectory_features", "build_feature_cache", "FeatureCache", "future_targets", "FutureFeatureTransform"]

Extractor = Callable[[Sequence[Any]], np.ndarray]  # list of frames -> [T, d_feat]


def extract_trajectory_features(frames: Sequence[Any], extractor: Extractor, batch_size: int = 64) -> np.ndarray:
    """Run ``extractor`` over ``frames`` in batches and return ``[T, d_feat]`` float32."""
    chunks: List[np.ndarray] = []
    for start in range(0, len(frames), max(1, batch_size)):
        out = np.asarray(extractor(list(frames[start : start + batch_size])), dtype=np.float32)
        if out.ndim != 2:
            raise ValueError(f"extractor must return [batch, d_feat], got shape {out.shape}")
        chunks.append(out)
    feats = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)
    if feats.shape[0] != len(frames):
        raise ValueError(f"extractor returned {feats.shape[0]} rows for {len(frames)} frames")
    return feats


def build_feature_cache(
    trajectories: Iterable[Tuple[Any, Sequence[Any]]],
    extractor: Extractor,
    cache_dir: str | Path,
    batch_size: int = 64,
    overwrite: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Cache features for ``(trajectory_id, frames)`` pairs; returns ``{trajectory_id: (T, d_feat)}``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Tuple[int, int]] = {}
    for traj_id, frames in trajectories:
        path = cache_dir / f"{traj_id}.npy"
        if path.exists() and not overwrite:
            arr = np.load(path, mmap_mode="r")
        else:
            arr = extract_trajectory_features(frames, extractor, batch_size)
            np.save(path, arr)
        written[str(traj_id)] = tuple(arr.shape)
    return written


class FeatureCache:
    """Lazy, memory-mapped reader for ``build_feature_cache`` outputs."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._open: Dict[str, np.ndarray] = {}

    def __contains__(self, traj_id: Any) -> bool:
        return (self.cache_dir / f"{traj_id}.npy").exists()

    def get(self, traj_id: Any) -> np.ndarray:
        key = str(traj_id)
        if key not in self._open:
            self._open[key] = np.load(self.cache_dir / f"{key}.npy", mmap_mode="r")
        return self._open[key]


def future_targets(feats_seq: np.ndarray, t: int, offsets: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """``(features [len(offsets), d_feat], mask [len(offsets)])`` for frames ``t + offset``; missing frames are zero + mask 0."""
    T, d = feats_seq.shape
    out = np.zeros((len(offsets), d), dtype=np.float32)
    mask = np.zeros((len(offsets),), dtype=np.float32)
    for i, off in enumerate(offsets):
        idx = t + int(off)
        if 0 <= idx < T:
            out[i] = feats_seq[idx]
            mask[i] = 1.0
    return out, mask


class FutureFeatureTransform:
    """Sample-level transform adding ``future_features`` / ``future_features_mask`` from a :class:`FeatureCache`.

    The trajectory id and step are read from the sample (``traj_key`` / ``step_key``); samples without them or
    without a cached trajectory get an all-zero mask so the auxiliary loss ignores them.
    """

    def __init__(self, cache: FeatureCache, offsets: Sequence[int], traj_key: str = "trajectory_id", step_key: str = "step") -> None:
        if not offsets:
            raise ValueError("offsets must be non-empty")
        self.cache, self.offsets, self.traj_key, self.step_key = cache, [int(o) for o in offsets], traj_key, step_key

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(sample)
        traj, step = sample.get(self.traj_key), sample.get(self.step_key)
        if traj is None or step is None or traj not in self.cache:
            return out  # no field -> QwenMultiHeadLab masks this sample out of the feature-prediction loss
        feats, mask = future_targets(np.asarray(self.cache.get(traj)), int(step), self.offsets)
        out["future_features"], out["future_features_mask"] = feats, mask
        return out
