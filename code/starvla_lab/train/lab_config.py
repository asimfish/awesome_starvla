"""Configuration schema for the lab training hooks (``cfg.trainer.lab.*``).

Works on plain dicts, ``SimpleNamespace``-like objects and OmegaConf nodes so the same reader is
used by the StarVLA entry script and by CPU tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, List, Mapping, Optional, Sequence

__all__ = ["LLRDConfig", "AuxSchedulerConfig", "ProbesConfig", "HeadDropoutConfig", "LabConfig", "cfg_get", "cfg_set"]


def cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from dicts / objects / OmegaConf nodes; ``default`` when any hop is missing or None."""
    node = cfg
    for key in dotted.split("."):
        if node is None:
            return default
        if isinstance(node, Mapping):
            if key not in node:
                return default
            node = node[key]
        else:
            if not hasattr(node, key):
                return default
            node = getattr(node, key)
    return default if node is None else node


def cfg_set(cfg: Any, dotted: str, value: Any) -> None:
    """Write ``a.b.c = value`` in place, creating intermediate dict nodes when the container is a dict."""
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if isinstance(node, Mapping):
            if key not in node or node[key] is None:
                node[key] = {}
            node = node[key]
        else:
            if not hasattr(node, key) or getattr(node, key) is None:
                setattr(node, key, {})
            node = getattr(node, key)
    if isinstance(node, Mapping):
        node[keys[-1]] = value
    else:
        setattr(node, keys[-1], value)


def _fill(dc_cls, node: Any):
    kwargs = {}
    for f in fields(dc_cls):
        val = cfg_get(node, f.name, None) if node is not None else None
        if val is None:
            continue
        if is_dataclass(f.type) if isinstance(f.type, type) else False:
            kwargs[f.name] = _fill(f.type, val)
        elif isinstance(val, (list, tuple)) or type(val).__name__ == "ListConfig":
            kwargs[f.name] = list(val)
        else:
            kwargs[f.name] = val
    return dc_cls(**kwargs)


@dataclass
class LLRDConfig:
    enabled: bool = False
    decay: float = 0.9
    head_lr: Optional[float] = None
    drift_driven: bool = False
    drift_high: float = 0.10
    drift_low: float = 0.05
    down_factor: float = 0.5
    up_factor: float = 1.1
    min_scale: float = 0.05


@dataclass
class AuxSchedulerConfig:
    enabled: bool = False
    strategy: str = "fixed"
    ratio_min: float = 0.1
    ratio_max: float = 0.5
    loss_scale_min: float = 0.1
    loss_scale_max: float = 1.0
    init_u: float = 1.0
    drift_high: float = 0.10
    drift_low: float = 0.05
    gain: float = 2.0
    max_step: float = 0.1
    loss_scale_key: str = "trainer.loss_scale.vlm"
    sample_prob_key: str = "trainer.vlm_sample_prob"


@dataclass
class ProbesConfig:
    enabled: bool = False
    every_n_steps: int = 2000
    jsonl_path: str = "lab_probes.jsonl"
    probe_batch_size: int = 64
    layers: Optional[List[int]] = None
    drift_summary: str = "mean"        # which DriftTracker summary scalar drives the schedulers
    calibrate_only: bool = False       # record drift but do not act on it (threshold calibration run)


@dataclass
class HeadDropoutConfig:
    enabled: bool = False
    p_all: float = 0.5
    seed: int = 0


@dataclass
class LabConfig:
    llrd: LLRDConfig = field(default_factory=LLRDConfig)
    aux_scheduler: AuxSchedulerConfig = field(default_factory=AuxSchedulerConfig)
    probes: ProbesConfig = field(default_factory=ProbesConfig)
    head_dropout: HeadDropoutConfig = field(default_factory=HeadDropoutConfig)

    @classmethod
    def from_cfg(cls, cfg: Any, key: str = "trainer.lab") -> "LabConfig":
        node = cfg_get(cfg, key, None)
        if node is None:
            return cls()
        return cls(
            llrd=_fill(LLRDConfig, cfg_get(node, "llrd")),
            aux_scheduler=_fill(AuxSchedulerConfig, cfg_get(node, "aux_scheduler")),
            probes=_fill(ProbesConfig, cfg_get(node, "probes")),
            head_dropout=_fill(HeadDropoutConfig, cfg_get(node, "head_dropout")),
        )

    def any_enabled(self) -> bool:
        return any((self.llrd.enabled, self.aux_scheduler.enabled, self.probes.enabled, self.head_dropout.enabled))

    def validate(self) -> None:
        if self.llrd.drift_driven and not self.probes.enabled:
            raise ValueError("trainer.lab.llrd.drift_driven requires trainer.lab.probes.enabled (drift comes from the probes)")
        if self.aux_scheduler.enabled and self.aux_scheduler.strategy == "drift" and not self.probes.enabled:
            raise ValueError("trainer.lab.aux_scheduler.strategy=drift requires trainer.lab.probes.enabled")
        if self.probes.enabled and self.probes.every_n_steps <= 0:
            raise ValueError("trainer.lab.probes.every_n_steps must be positive")
