"""Trajectory-level subsampling for data-fraction curves (protocol F1, RoboCasa-GR1 10/20/50/100 %).

StarVLA's ``LeRobotSingleDataset`` indexes samples through ``all_steps = [(trajectory_id, base_index), ...]``;
``TrajectorySubset`` keeps every step of a deterministic, seeded subset of trajectories so that the
fraction refers to *demonstrations* (as in the VLAct data-efficiency curve), not to random frames.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["select_trajectories", "TrajectorySubset", "make_fraction_hook", "install_fraction_hook"]


def select_trajectories(trajectory_ids: Sequence[Any], fraction: float, seed: int = 0, min_keep: int = 1) -> List[Any]:
    """Deterministically pick ``ceil(fraction * N)`` trajectory ids (at least ``min_keep``).

    The permutation depends only on ``(sorted ids, seed)``, so baselines and variants trained with the
    same fraction and seed see exactly the same demonstrations.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    ids = sorted(trajectory_ids, key=lambda x: (str(type(x)), x))
    if fraction == 1.0:
        return list(ids)
    n_keep = max(min_keep, int(np.ceil(fraction * len(ids))))
    digest = hashlib.sha256(("|".join(map(str, ids)) + f"|{seed}").encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    perm = rng.permutation(len(ids))
    return [ids[i] for i in sorted(perm[:n_keep])]


class TrajectorySubset:
    """Map-style proxy exposing only the steps whose trajectory is in the selected subset.

    Requires ``dataset.all_steps`` (sequence of ``(trajectory_id, base_index)``) and ``dataset.trajectory_ids``;
    other attributes are forwarded to the wrapped dataset.
    """

    def __init__(self, dataset: Any, fraction: float, seed: int = 0, min_keep: int = 1) -> None:
        self._dataset = dataset
        self.fraction = float(fraction)
        self.seed = int(seed)
        self.kept_trajectories = select_trajectories(list(dataset.trajectory_ids), fraction, seed, min_keep)
        keep = set(self.kept_trajectories)
        self._indices: List[int] = [i for i, (traj, _) in enumerate(dataset.all_steps) if traj in keep]
        if not self._indices:
            raise ValueError("trajectory subset is empty")

    @property
    def all_steps(self) -> List[Tuple[Any, int]]:
        steps = self._dataset.all_steps
        return [steps[i] for i in self._indices]

    @property
    def trajectory_ids(self) -> np.ndarray:
        return np.asarray(self.kept_trajectories)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int):
        return self._dataset[self._indices[index]]

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def summary(self) -> Dict[str, Any]:
        return {
            "fraction": self.fraction,
            "seed": self.seed,
            "trajectories_kept": len(self.kept_trajectories),
            "trajectories_total": int(len(self._dataset.trajectory_ids)),
            "steps_kept": len(self._indices),
            "steps_total": int(len(self._dataset.all_steps)),
        }


def make_fraction_hook(fraction: float, seed: int = 0, inner: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
    """Wrap a StarVLA ``DataConfig.make_dataset``-style factory so it returns a ``TrajectorySubset``.

    ``inner`` is the original factory (``lambda *a, **k: TrajectorySubset(inner(*a, **k), ...)``); with
    ``inner=None`` the returned hook expects the dataset as its single argument.
    """

    def hook(*args, **kwargs):
        dataset = inner(*args, **kwargs) if inner is not None else args[0]
        if float(fraction) >= 1.0:
            return dataset
        return TrajectorySubset(dataset, fraction, seed)

    return hook


def install_fraction_hook(fraction: float, seed: int = 0, module: Any = None, factory_name: str = "make_LeRobotSingleDataset") -> Callable[[], None]:
    """Monkeypatch StarVLA's single-dataset factory so every dataset in a mixture is a ``TrajectorySubset``.

    ``module`` defaults to ``starVLA.dataloader.lerobot_datasets`` (whose ``get_vla_dataset`` builds a
    ``LeRobotMixtureDataset`` from ``make_LeRobotSingleDataset`` calls). Returns an ``uninstall`` callable.
    With ``fraction >= 1`` nothing is patched.
    """
    if module is None:
        from starVLA.dataloader import lerobot_datasets as module  # type: ignore  # StarVLA runtime only
    if float(fraction) >= 1.0:
        return lambda: None
    original = getattr(module, factory_name)

    def patched(*args, **kwargs):
        return TrajectorySubset(original(*args, **kwargs), fraction, seed)

    patched.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(module, factory_name, patched)

    def uninstall() -> None:
        setattr(module, factory_name, original)

    return uninstall
