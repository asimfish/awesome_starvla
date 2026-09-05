"""Layer-wise representation drift of a training model relative to a frozen reference (WP1).

Drift of layer ``l`` is ``1 - CKA(ref_l, cur_l)`` on a fixed probe batch, where the per-layer
representations come from a user-supplied ``extract_fn(model, batch) -> Sequence[Tensor]`` so the
tracker is independent of StarVLA / transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor

from .cka import layerwise_cka

ExtractFn = Callable[[Any, Any], Sequence[Tensor]]


@dataclass
class DriftRecord:
    step: Optional[int]
    drift: Tensor  # [L], float64, cpu


class DriftTracker:
    """Track ``1 - CKA`` per layer between a reference and the current model on one probe batch.

    ``reference`` is either a model (representations are extracted once with ``extract_fn``) or a
    precomputed sequence of per-layer tensors. ``layers`` selects indices from the list returned by
    ``extract_fn`` (default: all). Representations are stored on ``device`` (cpu by default); the CKA of each
    layer pair is computed on ``compute_device`` (default: wherever the tensors are), so a GPU can do the
    fp64 Gram products while the reference stays in host memory.
    """

    def __init__(
        self,
        extract_fn: ExtractFn,
        probe_batch: Any,
        reference: Union[Any, Sequence[Tensor]],
        layers: Optional[Sequence[int]] = None,
        layer_names: Optional[Sequence[str]] = None,
        chunk_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        compute_device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.extract_fn = extract_fn
        self.probe_batch = probe_batch
        self.layers = list(layers) if layers is not None else None
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.compute_device = torch.device(compute_device) if compute_device is not None else None
        self.history: List[DriftRecord] = []

        if isinstance(reference, (list, tuple)):
            self.reference = self._select([r.detach().to(self.device) for r in reference])
        else:
            self.reference = self._extract(reference)
        n_layers = len(self.reference)
        if layer_names is None:
            indices = self.layers if self.layers is not None else range(n_layers)
            layer_names = [f"layer_{i}" for i in indices]
        if len(layer_names) != n_layers:
            raise ValueError(f"got {len(layer_names)} layer names for {n_layers} layers")
        self.layer_names = list(layer_names)

    def _select(self, reps: List[Tensor]) -> List[Tensor]:
        if self.layers is None:
            return reps
        return [reps[i] for i in self.layers]

    @torch.no_grad()
    def _extract(self, model: Any) -> List[Tensor]:
        reps = [r.detach().to(self.device) for r in self.extract_fn(model, self.probe_batch)]
        return self._select(reps)

    @property
    def num_layers(self) -> int:
        return len(self.reference)

    def update(self, model: Any, step: Optional[int] = None) -> Tensor:
        """Measure the current model; append and return per-layer drift ``[L]`` in ``[0, 1]``."""
        current = self._extract(model)
        if len(current) != self.num_layers:
            raise ValueError(f"extract_fn returned {len(current)} layers, reference has {self.num_layers}")
        cka = layerwise_cka(self.reference, current, chunk_size=self.chunk_size, device=self.compute_device)
        drift = (1.0 - cka).clamp(0.0, 1.0)
        self.history.append(DriftRecord(step=step, drift=drift))
        return drift

    def latest(self) -> Optional[Tensor]:
        return self.history[-1].drift if self.history else None

    def steps(self) -> List[Optional[int]]:
        return [rec.step for rec in self.history]

    def per_layer(self) -> Tensor:
        """History as a ``[T, L]`` tensor (``T`` updates so far)."""
        if not self.history:
            return torch.zeros(0, self.num_layers, dtype=torch.float64)
        return torch.stack([rec.drift for rec in self.history])

    def summary(self) -> Dict[str, Any]:
        """Scalars of the latest measurement: ``mean``, ``max``, ``max_layer`` and the per-layer dict."""
        latest = self.latest()
        if latest is None:
            per_layer = {name: 0.0 for name in self.layer_names}
            return {"step": None, "num_updates": 0, "mean": 0.0, "max": 0.0, "max_layer": None, "per_layer": per_layer}
        values = [float(v) for v in latest]
        max_idx = int(torch.argmax(latest))
        return {
            "step": self.history[-1].step,
            "num_updates": len(self.history),
            "mean": float(latest.mean()),
            "max": values[max_idx],
            "max_layer": self.layer_names[max_idx],
            "per_layer": dict(zip(self.layer_names, values)),
        }


def drift_to_llrd_decay(
    drift_per_layer: Union[Tensor, Sequence[float]],
    base_decay: float = 1.0,
    min_decay: float = 0.1,
    drift_max: float = 0.5,
    power: float = 1.0,
) -> Tensor:
    """Map per-layer drift to per-layer learning-rate multipliers in ``[min_decay, base_decay]``.

    ``m_l = min_decay + (base_decay - min_decay) * (1 - clip(drift_l / drift_max, 0, 1)) ** power``:
    monotonically non-increasing in drift, ``base_decay`` at zero drift and ``min_decay`` once drift
    reaches ``drift_max``. Multiply onto the static LLRD learning rates of the corresponding layers.
    """
    if not 0.0 < min_decay <= base_decay:
        raise ValueError("need 0 < min_decay <= base_decay")
    if drift_max <= 0 or power <= 0:
        raise ValueError("drift_max and power must be positive")
    drift = torch.as_tensor(drift_per_layer, dtype=torch.float64)
    frac = (drift / drift_max).clamp(0.0, 1.0)
    return min_decay + (base_decay - min_decay) * (1.0 - frac) ** power
