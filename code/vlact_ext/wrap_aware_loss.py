"""Wrap-aware L1 loss for periodic joint dimensions (VLAct recipe (e)).

Data side:  ``wrap_to_pi(a) = (a + pi) mod 2pi - pi``
Loss side:  ``delta = ((a_hat - a) + pi) mod 2pi - pi``, ``L_wrap = |delta|`` on periodic dims,
            plain L1 on the remaining dims, masked mean over active dims.

``period`` is the length of one full turn *in the space the actions live in*: ``2*pi`` for
raw radians, ``2.0`` when joints were min-max normalised from ``[-pi, pi]`` to ``[-1, 1]``.
It may also be a per-dimension array broadcastable to the last axis.

For flow-matching heads (PI / GR00T) the wrap-aware term must be applied to the *sample*
(final generated action), not to the velocity target ``a - eps``: velocity residuals are not
periodic. Use ``flow_matching_sample_estimate`` to get the one-step clean-sample estimate
``x1_hat = x_t + (1 - t) * v_hat`` and feed it to ``masked_wrap_aware_l1``.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch

TWO_PI = 2.0 * math.pi

Period = Union[float, np.ndarray, torch.Tensor]


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


def wrap_to_pi(x, period: Period = TWO_PI):
    """Map ``x`` into ``[-period/2, period/2)``. Works on torch tensors and numpy arrays."""
    if _is_tensor(x):
        p = torch.as_tensor(period, dtype=x.dtype, device=x.device)
        half = p / 2
        return torch.remainder(x + half, p) - half
    x = np.asarray(x)
    p = np.asarray(period, dtype=x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64)
    half = p / 2
    return np.mod(x + half, p) - half


def wrap_aware_residual(pred, target, periodic_mask=None, period: Period = TWO_PI):
    """``pred - target``, wrapped on dims where ``periodic_mask`` is True."""
    delta = pred - target
    if periodic_mask is None:
        return delta
    wrapped = wrap_to_pi(delta, period)
    if _is_tensor(delta):
        mask = torch.as_tensor(periodic_mask, device=delta.device).to(torch.bool)
        return torch.where(mask, wrapped, delta)
    mask = np.asarray(periodic_mask, dtype=bool)
    return np.where(mask, wrapped, delta)


def _expand_mask_torch(mask, ref: torch.Tensor) -> torch.Tensor:
    m = torch.as_tensor(mask, device=ref.device).to(torch.bool)
    try:
        m = m.expand(ref.shape)
    except RuntimeError as exc:
        raise ValueError(f"mask shape {tuple(m.shape)} is not broadcastable to {tuple(ref.shape)}") from exc
    return m.to(ref.dtype)


def masked_wrap_aware_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    active_mask: Optional[torch.Tensor] = None,
    periodic_mask: Optional[torch.Tensor] = None,
    period: Period = TWO_PI,
) -> torch.Tensor:
    """Masked mean of ``|wrap_aware_residual|``.

    Args:
        pred, target: same shape, typically ``[B, T, D]``.
        active_mask: bool, broadcastable to ``pred`` (``[D]``, ``[T, D]``, ``[B, 1, D]``, ``[B, T, D]``).
            ``None`` means every cell is active. All-False returns ``0`` (no NaN, graph kept).
        periodic_mask: bool, broadcastable to ``pred``; dims where the residual is wrapped.
            ``None`` degenerates to plain masked L1.
        period: scalar or per-dim period of the periodic dims.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    err = wrap_aware_residual(pred, target, periodic_mask, period).abs()
    if active_mask is None:
        return err.mean()
    valid = _expand_mask_torch(active_mask, err)
    return (err * valid).sum() / valid.sum().clamp_min(1.0)


def masked_wrap_aware_l1_np(
    pred: np.ndarray,
    target: np.ndarray,
    active_mask: Optional[np.ndarray] = None,
    periodic_mask: Optional[np.ndarray] = None,
    period: Period = TWO_PI,
) -> float:
    """NumPy twin of :func:`masked_wrap_aware_l1` (for data-side checks and evaluation)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    err = np.abs(wrap_aware_residual(pred, target, periodic_mask, period))
    if active_mask is None:
        return float(err.mean())
    valid = np.broadcast_to(np.asarray(active_mask, dtype=bool), err.shape).astype(np.float64)
    denom = valid.sum()
    if denom == 0:
        return 0.0
    return float((err * valid).sum() / denom)


def flow_matching_sample_estimate(noisy: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """One-step clean-sample estimate for the StarVLA flow-matching convention.

    StarVLA heads use ``x_t = (1 - t) * eps + t * a`` and predict ``v = a - eps``, hence
    ``a = x_t + (1 - t) * v``. ``t`` must broadcast against ``noisy`` (``[B, 1, 1]``).
    """
    return noisy + (1.0 - t) * velocity
