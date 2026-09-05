"""Linear centered kernel alignment (CKA) between two representations of the same samples.

Linear CKA in feature form (Kornblith et al., 2019)::

    CKA(X, Y) = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)

with column-centred ``Xc``, ``Yc``. It equals the kernel form ``HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L))``
with ``K = X X^T``. The kernel form costs O(N^2 d), the feature form O(N d^2); ``linear_cka`` picks the
cheaper one and can stream the feature form over row chunks for large ``N``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor


def _as_matrix(x: Tensor) -> Tensor:
    if x.dim() < 2:
        raise ValueError(f"expected a tensor of shape [N, ...], got {tuple(x.shape)}")
    return x.reshape(x.shape[0], -1)


def _check_pair(X: Tensor, Y: Tensor) -> None:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of samples, got {X.shape[0]} and {Y.shape[0]}")
    if X.shape[0] < 2:
        raise ValueError("CKA needs at least two samples")


def _cross_moments(X: Tensor, Y: Tensor, dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor]:
    Xc = X.to(dtype)
    Yc = Y.to(dtype)
    Xc = Xc - Xc.mean(dim=0, keepdim=True)
    Yc = Yc - Yc.mean(dim=0, keepdim=True)
    return Xc.T @ Yc, Xc.T @ Xc, Yc.T @ Yc


def _chunked_cross_moments(X: Tensor, Y: Tensor, chunk_size: int, dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor]:
    n = X.shape[0]
    mean_x = torch.zeros(X.shape[1], dtype=dtype, device=X.device)
    mean_y = torch.zeros(Y.shape[1], dtype=dtype, device=Y.device)
    for lo in range(0, n, chunk_size):
        mean_x += X[lo : lo + chunk_size].to(dtype).sum(dim=0)
        mean_y += Y[lo : lo + chunk_size].to(dtype).sum(dim=0)
    mean_x /= n
    mean_y /= n

    xty = torch.zeros(X.shape[1], Y.shape[1], dtype=dtype, device=X.device)
    xtx = torch.zeros(X.shape[1], X.shape[1], dtype=dtype, device=X.device)
    yty = torch.zeros(Y.shape[1], Y.shape[1], dtype=dtype, device=Y.device)
    for lo in range(0, n, chunk_size):
        xc = X[lo : lo + chunk_size].to(dtype) - mean_x
        yc = Y[lo : lo + chunk_size].to(dtype) - mean_y
        xty += xc.T @ yc
        xtx += xc.T @ xc
        yty += yc.T @ yc
    return xty, xtx, yty


def _cka_kernel_form(X: Tensor, Y: Tensor, dtype: torch.dtype, eps: float) -> Tensor:
    Xc = X.to(dtype)
    Yc = Y.to(dtype)
    Xc = Xc - Xc.mean(dim=0, keepdim=True)
    Yc = Yc - Yc.mean(dim=0, keepdim=True)
    K = Xc @ Xc.T
    L = Yc @ Yc.T
    hsic_kl = (K * L).sum()
    denom = torch.sqrt((K * K).sum() * (L * L).sum())
    return hsic_kl / denom.clamp_min(eps)


def linear_cka(
    X: Tensor,
    Y: Tensor,
    chunk_size: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
) -> Tensor:
    """Linear CKA between ``X`` and ``Y`` (both ``[N, ...]``, trailing dims flattened) as a 0-d tensor.

    Invariant to orthogonal transforms and isotropic scaling of either input; ``linear_cka(X, X) == 1``.
    ``chunk_size`` streams the feature-form Gram accumulation over rows of ``chunk_size`` samples so
    memory is O(d^2) regardless of ``N``. Degenerate (constant) inputs return 0.
    """
    X = _as_matrix(X)
    Y = _as_matrix(Y)
    _check_pair(X, Y)
    n = X.shape[0]
    if chunk_size is not None and chunk_size > 0 and n > chunk_size:
        xty, xtx, yty = _chunked_cross_moments(X, Y, chunk_size, dtype)
    elif n <= max(X.shape[1], Y.shape[1]):
        return _cka_kernel_form(X, Y, dtype, eps)
    else:
        xty, xtx, yty = _cross_moments(X, Y, dtype)
    denom = torch.linalg.norm(xtx) * torch.linalg.norm(yty)
    return (xty * xty).sum() / denom.clamp_min(eps)


def layerwise_cka(
    reps_a: Sequence[Tensor],
    reps_b: Sequence[Tensor],
    chunk_size: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Per-layer linear CKA between two lists of representations; returns a ``[L]`` tensor."""
    if len(reps_a) != len(reps_b):
        raise ValueError(f"layer count mismatch: {len(reps_a)} vs {len(reps_b)}")
    if not reps_a:
        return torch.zeros(0, dtype=dtype)
    values = [linear_cka(a, b, chunk_size=chunk_size, dtype=dtype) for a, b in zip(reps_a, reps_b)]
    return torch.stack([v.to("cpu") for v in values])
