"""Partially unified 20-D cross-embodiment action layout (VLAct recipe (d)).

Unified slots (0-based; the paper counts from 1):
    0-5    bimanual left-arm joint angles      (periodic)
    6-11   bimanual right-arm joint angles     (periodic)
    12-17  single-arm delta end-effector (xyz + rot)
    18     shared gripper  (Franka gripper == AgileX left gripper)
    19     right gripper

Each embodiment is described by an :class:`EmbodimentLayout`: for native dim ``i`` the
unified slot is ``slots[i]``; ``periodic`` lists native dims that are joint angles. Adding a
new robot only requires a new dict entry (see ``DEFAULT_LAYOUTS``).

The sample-level transform reads ``sample["robot_tag"]`` (StarVLA embodiment tag), rewrites
``sample["action"]`` to ``[T, 20]`` and adds ``action_mask`` / ``periodic_mask`` (``[T, 20]``
bool, the shape ``QwenOFT.masked_l1_loss`` already consumes). Inactive slots are zero-filled
so the loss must be masked (VLAct Fig. 4: naive zero-padding without a mask hurts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from .wrap_aware_loss import wrap_to_pi

UNIFIED_DIM = 20

LEFT_ARM_JOINTS: Tuple[int, ...] = tuple(range(0, 6))
RIGHT_ARM_JOINTS: Tuple[int, ...] = tuple(range(6, 12))
SINGLE_ARM_DELTA_EE: Tuple[int, ...] = tuple(range(12, 18))
SHARED_GRIPPER: int = 18
RIGHT_GRIPPER: int = 19


@dataclass(frozen=True)
class EmbodimentLayout:
    """``slots[i]`` is the unified slot of native dim ``i``; ``periodic`` are native dims to wrap."""

    slots: Tuple[int, ...]
    periodic: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        slots = tuple(int(s) for s in self.slots)
        periodic = tuple(int(p) for p in self.periodic)
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "periodic", periodic)
        if len(set(slots)) != len(slots):
            raise ValueError(f"duplicate unified slots in layout: {slots}")
        if any(s < 0 for s in slots):
            raise ValueError(f"negative slot index in layout: {slots}")
        bad = [p for p in periodic if p < 0 or p >= len(slots)]
        if bad:
            raise ValueError(f"periodic native dims {bad} out of range for native_dim={len(slots)}")

    @property
    def native_dim(self) -> int:
        return len(self.slots)

    @property
    def periodic_slots(self) -> Tuple[int, ...]:
        return tuple(self.slots[p] for p in self.periodic)

    def active_mask(self, unified_dim: int = UNIFIED_DIM) -> np.ndarray:
        mask = np.zeros(unified_dim, dtype=bool)
        mask[list(self.slots)] = True
        return mask

    def periodic_mask(self, unified_dim: int = UNIFIED_DIM) -> np.ndarray:
        mask = np.zeros(unified_dim, dtype=bool)
        if self.periodic:
            mask[list(self.periodic_slots)] = True
        return mask

    def validate(self, unified_dim: int) -> None:
        if max(self.slots) >= unified_dim:
            raise ValueError(f"slot {max(self.slots)} >= unified_dim {unified_dim}")


# Franka / LIBERO: delta EE xyz(3) + delta rot(3) + gripper(1) -> 7 dims.
FRANKA_DELTA_EE_GRIPPER = EmbodimentLayout(slots=SINGLE_ARM_DELTA_EE + (SHARED_GRIPPER,))
# AgileX (StarVLA AgilexDataConfig order): left_joints(6), right_joints(6), left_gripper, right_gripper.
AGILEX_BIMANUAL_JOINTS = EmbodimentLayout(
    slots=LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + (SHARED_GRIPPER, RIGHT_GRIPPER),
    periodic=tuple(range(12)),
)

DEFAULT_LAYOUTS: Dict[str, EmbodimentLayout] = {
    "franka": FRANKA_DELTA_EE_GRIPPER,
    "agilex": AGILEX_BIMANUAL_JOINTS,
}


class UnifiedActionLayout:
    """Registry ``robot_tag -> EmbodimentLayout`` plus the to/from-unified conversions."""

    def __init__(
        self,
        layouts: Optional[Mapping[str, EmbodimentLayout]] = None,
        unified_dim: int = UNIFIED_DIM,
    ) -> None:
        self.unified_dim = int(unified_dim)
        self._layouts: Dict[str, EmbodimentLayout] = {}
        for tag, layout in (DEFAULT_LAYOUTS if layouts is None else layouts).items():
            self.register(tag, layout)

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "UnifiedActionLayout":
        """Build from a plain dict, e.g. the ``framework.unified_layout`` YAML node.

        ``{"unified_dim": 20, "layouts": {"franka": {"preset": "franka"},
        "new_embodiment": {"slots": [...], "periodic": [...]}}}``. ``layouts: null`` uses defaults.
        """
        cfg = dict(cfg or {})
        unified_dim = int(cfg.get("unified_dim", UNIFIED_DIM))
        raw_layouts = cfg.get("layouts", None)
        if raw_layouts is None:
            return cls(None, unified_dim=unified_dim)
        layouts: Dict[str, EmbodimentLayout] = {}
        for tag, spec in dict(raw_layouts).items():
            if isinstance(spec, EmbodimentLayout):
                layouts[tag] = spec
            elif isinstance(spec, str):
                layouts[tag] = _preset(spec)
            elif "preset" in spec:
                layouts[tag] = _preset(spec["preset"])
            else:
                layouts[tag] = EmbodimentLayout(slots=tuple(spec["slots"]), periodic=tuple(spec.get("periodic", ())))
        return cls(layouts, unified_dim=unified_dim)

    def register(self, tag: str, layout: EmbodimentLayout) -> None:
        layout.validate(self.unified_dim)
        self._layouts[str(tag)] = layout

    @property
    def tags(self) -> Tuple[str, ...]:
        return tuple(self._layouts)

    def get(self, robot_tag: Optional[str]) -> EmbodimentLayout:
        if robot_tag is None or robot_tag not in self._layouts:
            raise KeyError(
                f"unknown robot_tag {robot_tag!r}; registered tags: {sorted(self._layouts)}. "
                "Register it with UnifiedActionLayout.register(tag, EmbodimentLayout(...))."
            )
        return self._layouts[robot_tag]

    def masks(self, robot_tag: str) -> Tuple[np.ndarray, np.ndarray]:
        layout = self.get(robot_tag)
        return layout.active_mask(self.unified_dim), layout.periodic_mask(self.unified_dim)

    def to_unified(self, action: np.ndarray, robot_tag: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``action[..., D_native] -> (unified[..., D_u], active_mask[D_u], periodic_mask[D_u])``."""
        layout = self.get(robot_tag)
        action = np.asarray(action)
        if action.shape[-1] != layout.native_dim:
            raise ValueError(
                f"robot_tag {robot_tag!r} expects native action dim {layout.native_dim}, got {action.shape[-1]}"
            )
        unified = np.zeros(action.shape[:-1] + (self.unified_dim,), dtype=action.dtype)
        unified[..., list(layout.slots)] = action
        return unified, layout.active_mask(self.unified_dim), layout.periodic_mask(self.unified_dim)

    def from_unified(self, unified: np.ndarray, robot_tag: str) -> np.ndarray:
        """Inverse of :meth:`to_unified`: gather the embodiment's active slots back in native order."""
        layout = self.get(robot_tag)
        unified = np.asarray(unified)
        if unified.shape[-1] != self.unified_dim:
            raise ValueError(f"expected unified dim {self.unified_dim}, got {unified.shape[-1]}")
        return unified[..., list(layout.slots)]


def _preset(name: str) -> EmbodimentLayout:
    if name not in DEFAULT_LAYOUTS:
        raise KeyError(f"unknown layout preset {name!r}; available: {sorted(DEFAULT_LAYOUTS)}")
    return DEFAULT_LAYOUTS[name]


class UnifiedActionTransform:
    """Sample-dict transform for StarVLA ``{"action", "robot_tag", ...}`` samples.

    Args:
        layout: registry used to map ``robot_tag`` to slots.
        wrap_period: if set, periodic dims are wrapped with :func:`wrap_to_pi` (data-side VLAct eq. 1).
            Use ``2*pi`` for raw radians, ``2.0`` for ``[-pi, pi] -> [-1, 1]`` normalised joints, or a
            per-slot array. Leave ``None`` when the dataset already wraps.
    """

    def __init__(self, layout: Optional[UnifiedActionLayout] = None, wrap_period=None) -> None:
        self.layout = layout if layout is not None else UnifiedActionLayout()
        self.wrap_period = wrap_period

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(sample)
        action = np.asarray(sample["action"])
        unified, active, periodic = self.layout.to_unified(action, sample.get("robot_tag"))
        if self.wrap_period is not None and periodic.any():
            period = np.asarray(self.wrap_period, dtype=np.float64)
            idx = np.flatnonzero(periodic)
            per_dim = period[idx] if period.ndim > 0 else period
            unified[..., idx] = wrap_to_pi(unified[..., idx].astype(np.float64), per_dim).astype(unified.dtype)
        time_shape = unified.shape[:-1]
        out["action"] = unified
        out["action_mask"] = np.broadcast_to(active, time_shape + active.shape).copy()
        out["periodic_mask"] = np.broadcast_to(periodic, time_shape + periodic.shape).copy()
        return out


class TransformedDataset:
    """Map-style proxy applying ``transform`` to every item; other attributes are forwarded."""

    def __init__(self, dataset, transform: Callable[[Mapping[str, Any]], Dict[str, Any]]) -> None:
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        return self._transform(self._dataset[index])

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._dataset, name)


def make_dataset_hook(transform: Callable[[Mapping[str, Any]], Dict[str, Any]]) -> Callable[..., Any]:
    """Return a ``make_dataset(**kwargs)`` callable for a StarVLA ``DataConfig``.

    ``lerobot_datasets.make_LeRobotSingleDataset`` calls ``data_config.make_dataset(dataset_path=...,
    modality_configs=..., transforms=..., embodiment_tag=..., video_backend=..., delete_pause_frame=...,
    data_cfg=..., dataset_name=...)`` when present. The returned dataset subclasses
    ``LeRobotSingleDataset`` and applies ``transform`` inside ``_pack_sample`` so that
    ``LeRobotMixtureDataset`` (which calls ``_pack_sample`` directly) also yields unified samples.
    """

    def make_dataset(**kwargs):
        from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

        class UnifiedLayoutSingleDataset(LeRobotSingleDataset):
            def _pack_sample(self, data: dict) -> dict:
                return transform(super()._pack_sample(data))

        kwargs.pop("dataset_name", None)
        return UnifiedLayoutSingleDataset(**kwargs)

    return make_dataset
