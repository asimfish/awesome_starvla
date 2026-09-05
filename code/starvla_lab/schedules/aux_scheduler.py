"""Auxiliary (VLM) data scheduler: VLM batch sampling probability and ``loss_scale.vlm`` per step (WP4).

Both outputs are driven by one normalised control ``u in [0, 1]``::

    vlm_sample_prob = ratio_min      + u * (ratio_max - ratio_min)
    vlm_loss_scale  = loss_scale_min + u * (loss_scale_max - loss_scale_min)

``fixed`` keeps ``u`` constant, ``linear`` anneals it from 1 to 0 over ``total_steps``, and ``drift``
raises it when the drift signal (e.g. ``DriftTracker.summary()["mean"]``) exceeds ``drift_high`` and
lowers it when the signal falls below ``drift_low`` (hysteresis band in between), moving at most
``max_step`` per new measurement.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Tuple

STRATEGIES = ("fixed", "linear", "drift")


def _set_dotted(cfg: Any, dotted_key: str, value: Any) -> None:
    """Assign ``value`` at a dotted path on an attribute- or item-style config (OmegaConf, dict, namespace)."""
    *parents, leaf = dotted_key.split(".")
    node = cfg
    for key in parents:
        if isinstance(node, dict):
            node = node[key]
        else:
            node = getattr(node, key)
    if isinstance(node, dict):
        node[leaf] = value
    else:
        setattr(node, leaf, value)


class AuxDataScheduler:
    """Return ``(vlm_sample_prob, vlm_loss_scale)`` for each training step under one of three strategies."""

    def __init__(
        self,
        strategy: str = "fixed",
        ratio_min: float = 0.1,
        ratio_max: float = 0.5,
        loss_scale_min: float = 0.1,
        loss_scale_max: float = 1.0,
        total_steps: Optional[int] = None,
        init_u: float = 1.0,
        drift_high: float = 0.10,
        drift_low: float = 0.05,
        gain: float = 2.0,
        max_step: float = 0.1,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        if not 0.0 <= ratio_min <= ratio_max <= 1.0:
            raise ValueError("need 0 <= ratio_min <= ratio_max <= 1")
        if not 0.0 <= loss_scale_min <= loss_scale_max:
            raise ValueError("need 0 <= loss_scale_min <= loss_scale_max")
        if not 0.0 <= init_u <= 1.0:
            raise ValueError("init_u must be in [0, 1]")
        if strategy == "linear" and (total_steps is None or total_steps <= 0):
            raise ValueError("linear strategy needs total_steps > 0")
        if not 0.0 <= drift_low <= drift_high:
            raise ValueError("need 0 <= drift_low <= drift_high")
        if gain <= 0 or max_step <= 0:
            raise ValueError("gain and max_step must be positive")
        self.strategy = strategy
        self.ratio_min, self.ratio_max = ratio_min, ratio_max
        self.loss_scale_min, self.loss_scale_max = loss_scale_min, loss_scale_max
        self.total_steps = total_steps
        self.drift_high, self.drift_low = drift_high, drift_low
        self.gain, self.max_step = gain, max_step
        self.u = init_u
        self.last_step: Optional[int] = None

    def _outputs(self) -> Tuple[float, float]:
        prob = self.ratio_min + self.u * (self.ratio_max - self.ratio_min)
        scale = self.loss_scale_min + self.u * (self.loss_scale_max - self.loss_scale_min)
        return prob, scale

    @property
    def vlm_sample_prob(self) -> float:
        return self._outputs()[0]

    @property
    def vlm_loss_scale(self) -> float:
        return self._outputs()[1]

    def step(self, step: int, drift: Optional[float] = None) -> Tuple[float, float]:
        """Advance to ``step``; for ``drift`` the control only moves when a new ``drift`` value is supplied."""
        self.last_step = step
        if self.strategy == "linear":
            assert self.total_steps is not None
            self.u = 1.0 - min(1.0, max(0.0, step / self.total_steps))
        elif self.strategy == "drift" and drift is not None:
            if drift > self.drift_high:
                excess = drift - self.drift_high
            elif drift < self.drift_low:
                excess = drift - self.drift_low
            else:
                excess = 0.0
            delta = max(-self.max_step, min(self.max_step, self.gain * excess))
            self.u = min(1.0, max(0.0, self.u + delta))
        return self._outputs()

    def sample_vlm(self, rng: random.Random) -> bool:
        """Bernoulli draw with the current VLM sampling probability (for skipping the VLM batch in a co-train step)."""
        return rng.random() < self.vlm_sample_prob

    def apply_to_cfg(
        self,
        cfg: Any,
        loss_scale_key: str = "trainer.loss_scale.vlm",
        sample_prob_key: Optional[str] = "trainer.vlm_sample_prob",
    ) -> Tuple[float, float]:
        """Write the current outputs into a StarVLA config in place.

        ``train_starvla_cotrain.py`` multiplies the VLM loss by ``cfg.trainer.loss_scale.vlm`` inside
        ``_train_step``, so updating that key before each step changes the effective VLM loss weight; the
        sampling probability is written to ``sample_prob_key`` for a patched step to consult (``None`` skips it).
        """
        prob, scale = self._outputs()
        _set_dotted(cfg, loss_scale_key, scale)
        if sample_prob_key:
            _set_dotted(cfg, sample_prob_key, prob)
        return prob, scale
