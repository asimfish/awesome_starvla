"""Layer-wise learning-rate decay (LLRD) parameter groups and a drift-driven online controller (WP2).

Static schedule (``layerwise_lr_decay_groups``)::

    decoder layer i (0 = closest to the input):  lr = base_lr * decay ** (n_layers - i) * multiplier_i
    vision encoder, token embeddings:            lr = base_lr * decay ** (n_layers + 1)
    other backbone params (final norm, lm_head): lr = base_lr
    everything outside the backbone (heads):     lr = head_lr

Freeze rules use the ``vlact_ext.freeze_rules`` syntax (exact path, ``re:``, ``path[lo:hi]``,
``llm_layers_below:N``). Frozen parameters are *excluded* from the optimizer, matching StarVLA's own
``build_param_lr_groups``; call ``vlact_ext.freeze_rules.freeze_by_rules`` with the same spec to also
switch off their gradients (StarVLA does this in ``freeze_backbones``).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

import torch
from torch import Tensor, nn

try:
    from vlact_ext import freeze_rules as _fr
except ImportError:  # repo layout: code/starvla_lab/schedules/llrd.py -> code/
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vlact_ext import freeze_rules as _fr

DEFAULT_LLM_LAYERS_PATH = _fr.DEFAULT_LLM_LAYERS_PATH
DEFAULT_VISUAL_PATH = _fr.DEFAULT_VISUAL_PATH
DEFAULT_EMBED_PATH = _fr.DEFAULT_EMBED_PATH


def _collect(module: nn.Module, frozen: Set[int], used: Set[int]) -> List[nn.Parameter]:
    params = [p for p in module.parameters() if id(p) not in frozen and id(p) not in used]
    used.update(id(p) for p in params)
    return params


def layerwise_lr_decay_groups(
    model: nn.Module,
    base_lr: float,
    decay: float,
    llm_layers_path: str = DEFAULT_LLM_LAYERS_PATH,
    n_layers: Optional[int] = None,
    head_lr: Optional[float] = None,
    freeze_rules_spec: _fr.RuleSpec = "",
    backbone_path: Optional[str] = None,
    visual_path: Optional[str] = DEFAULT_VISUAL_PATH,
    embed_path: Optional[str] = DEFAULT_EMBED_PATH,
    layer_multipliers: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Build StarVLA-style ``{"params", "lr", "name"}`` groups with layer-wise decayed learning rates.

    Decoder-layer groups additionally carry ``"layer_index"`` so ``layer_group_index`` can map them for
    ``DriftDrivenLLRD``. ``layer_multipliers`` (e.g. from ``probes.drift_to_llrd_decay``) scale each
    layer's lr. ``backbone_path`` defaults to the first component of ``llm_layers_path``; parameters
    outside it get ``head_lr`` (default ``base_lr``). Frozen parameters are left out of every group.
    """
    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    head_lr = base_lr if head_lr is None else head_lr
    backbone_path = llm_layers_path.split(".")[0] if backbone_path is None else backbone_path

    container = _fr.get_submodule(model, llm_layers_path)
    num_actual = len(container)  # type: ignore[arg-type]
    n = num_actual if n_layers is None else int(n_layers)
    if n < num_actual:
        raise ValueError(f"n_layers={n} is smaller than the {num_actual} layers found at {llm_layers_path!r}")
    multipliers = [1.0] * num_actual if layer_multipliers is None else [float(m) for m in layer_multipliers]
    if len(multipliers) != num_actual:
        raise ValueError(f"layer_multipliers has {len(multipliers)} entries for {num_actual} layers")
    if any(m <= 0 for m in multipliers):
        raise ValueError("layer_multipliers must be positive")

    frozen = _fr.resolve_frozen_param_ids(model, freeze_rules_spec, llm_layers_path=llm_layers_path)
    used: Set[int] = set()
    groups: List[Dict[str, Any]] = []

    for i in range(num_actual):
        params = _collect(container[i], frozen, used)
        if params:
            lr = base_lr * decay ** (n - i) * multipliers[i]
            groups.append({"params": params, "lr": lr, "name": f"{llm_layers_path}.{i}", "layer_index": i})

    deep_lr = base_lr * decay ** (n + 1)
    for path in (visual_path, embed_path):
        if not path:
            continue
        try:
            module = _fr.get_submodule(model, path)
        except AttributeError:
            warnings.warn(f"LLRD path {path!r} not found in model; skipped", stacklevel=2)
            continue
        params = _collect(module, frozen, used)
        if params:
            groups.append({"params": params, "lr": deep_lr, "name": path})

    try:
        backbone = _fr.get_submodule(model, backbone_path)
    except AttributeError:
        warnings.warn(f"backbone path {backbone_path!r} not found in model; remaining params use head_lr", stacklevel=2)
    else:
        params = _collect(backbone, frozen, used)
        if params:
            groups.append({"params": params, "lr": base_lr, "name": backbone_path})

    rest = [p for p in model.parameters() if id(p) not in used and id(p) not in frozen]
    if rest:
        groups.append({"params": rest, "lr": head_lr, "name": "head"})
    return groups


def layer_group_index(groups: Iterable[Mapping[str, Any]]) -> Dict[int, int]:
    """``{layer_index: position}`` for groups produced by ``layerwise_lr_decay_groups`` (or ``optimizer.param_groups``)."""
    return {int(g["layer_index"]): idx for idx, g in enumerate(groups) if "layer_index" in g}


class DriftDrivenLLRD:
    """Online per-layer lr multipliers driven by drift measurements (deterministic, hysteretic).

    Each measured layer keeps a multiplier ``m in [min_scale, 1]`` applied to its reference lr:
    drift above ``drift_high`` multiplies ``m`` by ``down_factor``; drift below ``drift_low`` multiplies
    it by ``up_factor`` (slow recovery, capped at 1); in between it is held. With ``lr_scheduler`` given
    (any ``torch.optim.lr_scheduler`` exposing ``base_lrs``, e.g. the ``LambdaLR`` returned by
    ``transformers.get_scheduler``), the reference is the scheduler's base lr and the multiplier is written
    into ``base_lrs`` so later scheduler steps keep it; otherwise the group lr is set directly.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        layer_group_index: Mapping[int, int],
        lr_scheduler: Optional[Any] = None,
        drift_high: float = 0.10,
        drift_low: float = 0.05,
        down_factor: float = 0.5,
        up_factor: float = 1.1,
        min_scale: float = 0.05,
    ) -> None:
        if not 0.0 <= drift_low <= drift_high:
            raise ValueError("need 0 <= drift_low <= drift_high")
        if not 0.0 < down_factor < 1.0:
            raise ValueError("down_factor must be in (0, 1)")
        if up_factor < 1.0:
            raise ValueError("up_factor must be >= 1")
        if not 0.0 < min_scale <= 1.0:
            raise ValueError("min_scale must be in (0, 1]")
        if lr_scheduler is not None and not hasattr(lr_scheduler, "base_lrs"):
            raise TypeError("lr_scheduler must expose `base_lrs` (torch LRScheduler / transformers LambdaLR)")
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.layer_group_index = {int(k): int(v) for k, v in layer_group_index.items()}
        self.drift_high, self.drift_low = drift_high, drift_low
        self.down_factor, self.up_factor, self.min_scale = down_factor, up_factor, min_scale
        self.multipliers: Dict[int, float] = {i: 1.0 for i in self.layer_group_index}
        if lr_scheduler is not None:
            self.reference_lrs = {g: float(lr_scheduler.base_lrs[g]) for g in self.layer_group_index.values()}
        else:
            self.reference_lrs = {g: float(optimizer.param_groups[g]["lr"]) for g in self.layer_group_index.values()}

    def step(self, drift_per_layer: Union[Tensor, Sequence[float], Mapping[int, float]]) -> Dict[int, float]:
        """Update multipliers from per-layer drift (indexed by layer index) and apply them; returns the multipliers."""
        for i in self.layer_group_index:
            drift = float(drift_per_layer[i])
            m = self.multipliers[i]
            if drift > self.drift_high:
                m = max(self.min_scale, m * self.down_factor)
            elif drift < self.drift_low:
                m = min(1.0, m * self.up_factor)
            self.multipliers[i] = m
        self.apply()
        return dict(self.multipliers)

    def apply(self) -> None:
        """Write ``reference_lr * multiplier`` into the optimizer (and scheduler base lrs when present)."""
        for i, g in self.layer_group_index.items():
            group = self.optimizer.param_groups[g]
            target = self.reference_lrs[g] * self.multipliers[i]
            if self.lr_scheduler is not None:
                prev = float(self.lr_scheduler.base_lrs[g])
                self.lr_scheduler.base_lrs[g] = target
                if prev > 0:
                    group["lr"] = group["lr"] * (target / prev)
            else:
                group["lr"] = target

    def lrs(self) -> Dict[int, float]:
        """Current optimizer lr of every controlled layer."""
        return {i: float(self.optimizer.param_groups[g]["lr"]) for i, g in self.layer_group_index.items()}
