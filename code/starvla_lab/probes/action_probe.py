"""Cross-head action probes fitted on frozen backbone hidden states (WP1).

A probe maps hidden states ``H`` (``[N, d]`` or ``[N, K, d]`` at the ``K`` action-query positions) to
action chunks ``A`` (``[N, K*D]`` or ``[N, K, D]``). Low held-out error of a *linear* probe means the
action is linearly readable from the backbone regardless of which decoder head was used for
pre-training, which is the quantity we want to correlate with cross-head fine-tuning success.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn


def r2_score(pred: Tensor, target: Tensor) -> float:
    """Variance-weighted R^2 pooled over all output dimensions (1.0 for a perfect fit)."""
    p = pred.reshape(pred.shape[0], -1).to(torch.float64)
    t = target.reshape(target.shape[0], -1).to(torch.float64)
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean(dim=0, keepdim=True)) ** 2).sum()
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def mae_score(pred: Tensor, target: Tensor) -> float:
    """Mean absolute error over all elements."""
    return float((pred.to(torch.float64) - target.reshape(pred.shape).to(torch.float64)).abs().mean())


class _MetricsMixin:
    def predict(self, H: Tensor) -> Tensor:  # pragma: no cover - overridden
        raise NotImplementedError

    def r2(self, H: Tensor, A: Tensor) -> float:
        return r2_score(self.predict(H), A)

    def mae(self, H: Tensor, A: Tensor) -> float:
        return mae_score(self.predict(H), A)

    def metrics(self, H: Tensor, A: Tensor) -> Dict[str, float]:
        pred = self.predict(H)
        return {"mae": mae_score(pred, A), "r2": r2_score(pred, A)}


def _split_positions(H: Tensor, A: Tensor) -> Tuple[Tensor, Tensor]:
    """Return ``H`` as ``[N, K, d]`` and ``A`` as ``[N, K, D]`` for per-position fitting."""
    if H.dim() != 3:
        raise ValueError(f"per_position=True needs H of shape [N, K, d], got {tuple(H.shape)}")
    n, k, _ = H.shape
    if A.shape[0] != n:
        raise ValueError(f"H and A must have the same number of samples, got {n} and {A.shape[0]}")
    if A.dim() == 3:
        if A.shape[1] != k:
            raise ValueError(f"A has {A.shape[1]} positions but H has {k}")
        return H, A
    if A.dim() == 2 and A.shape[1] % k == 0:
        return H, A.reshape(n, k, A.shape[1] // k)
    raise ValueError(f"cannot split A of shape {tuple(A.shape)} into {k} positions")


def _ridge_solve(X: Tensor, Y: Tensor, ridge: float) -> Tuple[Tensor, Tensor]:
    """Ridge regression with intercept for ``X[..., N, F]``, ``Y[..., N, O]``; returns ``W[..., F, O]``, ``b[..., O]``.

    Uses the primal (F x F) system when ``N >= F`` and the dual (N x N) system otherwise; both are
    the exact minimiser of ``||Xc W - Yc||^2 + ridge ||W||^2``.
    """
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    mean_x = X.mean(dim=-2, keepdim=True)
    mean_y = Y.mean(dim=-2, keepdim=True)
    Xc = X - mean_x
    Yc = Y - mean_y
    n, f = X.shape[-2], X.shape[-1]
    if n >= f:
        gram = Xc.transpose(-1, -2) @ Xc + ridge * torch.eye(f, dtype=X.dtype, device=X.device)
        W = torch.linalg.solve(gram, Xc.transpose(-1, -2) @ Yc)
    else:
        kernel = Xc @ Xc.transpose(-1, -2) + ridge * torch.eye(n, dtype=X.dtype, device=X.device)
        W = Xc.transpose(-1, -2) @ torch.linalg.solve(kernel, Yc)
    b = (mean_y - mean_x @ W).squeeze(-2)
    return W, b


@dataclass
class LinearProbe(_MetricsMixin):
    """Closed-form ridge probe. ``weight`` is ``[F, O]`` (flattened) or ``[K, d, D]`` (per position)."""

    weight: Tensor
    bias: Tensor
    ridge: float
    per_position: bool
    out_shape: Tuple[int, ...]

    def predict(self, H: Tensor) -> Tensor:
        n = H.shape[0]
        X = H.to(self.weight.dtype)
        if self.per_position:
            pred = torch.einsum("nkd,kdo->nko", X, self.weight) + self.bias
        else:
            pred = X.reshape(n, -1) @ self.weight + self.bias
        return pred.reshape(n, *self.out_shape)


def fit_linear_probe(H: Tensor, A: Tensor, ridge: float = 1e-3, per_position: bool = False) -> LinearProbe:
    """Fit a ridge-regression probe from hidden states ``H`` to actions ``A``.

    ``per_position=False`` flattens ``[N, K, d] -> [N, K*d]`` and fits one regression; ``per_position=True``
    fits an independent regression at each of the ``K`` action-query positions.
    """
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if H.shape[0] != A.shape[0]:
        raise ValueError(f"H and A must have the same number of samples, got {H.shape[0]} and {A.shape[0]}")
    out_shape = tuple(A.shape[1:])
    if per_position:
        Hk, Ak = _split_positions(H, A)
        W, b = _ridge_solve(Hk.transpose(0, 1), Ak.transpose(0, 1), ridge)  # [K, d, D], [K, D]
    else:
        W, b = _ridge_solve(H.reshape(H.shape[0], -1), A.reshape(A.shape[0], -1), ridge)
    return LinearProbe(weight=W, bias=b, ridge=ridge, per_position=per_position, out_shape=out_shape)


class MLPProbe(_MetricsMixin):
    """Two-layer MLP probe operating on standardised inputs/outputs (statistics from the fit data)."""

    def __init__(
        self,
        net: nn.Module,
        in_mean: Tensor,
        in_std: Tensor,
        out_mean: Tensor,
        out_std: Tensor,
        out_shape: Tuple[int, ...],
        losses: Optional[List[float]] = None,
    ) -> None:
        self.net = net
        self.in_mean = in_mean
        self.in_std = in_std
        self.out_mean = out_mean
        self.out_std = out_std
        self.out_shape = out_shape
        self.losses: List[float] = list(losses or [])

    @torch.no_grad()
    def predict(self, H: Tensor) -> Tensor:
        n = H.shape[0]
        x = (H.reshape(n, -1).to(self.in_mean) - self.in_mean) / self.in_std
        y = self.net(x) * self.out_std + self.out_mean
        return y.reshape(n, *self.out_shape)


def fit_mlp_probe(
    H: Tensor,
    A: Tensor,
    hidden: int = 256,
    steps: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
    batch_size: Optional[int] = None,
    weight_decay: float = 0.0,
) -> MLPProbe:
    """Fit a small MLP probe with Adam on the MSE of standardised targets (deterministic for a fixed seed).

    ``batch_size=None`` runs full-batch steps; otherwise each step uses a seeded random mini-batch.
    """
    if H.shape[0] != A.shape[0]:
        raise ValueError(f"H and A must have the same number of samples, got {H.shape[0]} and {A.shape[0]}")
    n = H.shape[0]
    X = H.reshape(n, -1).detach().to(torch.float32)
    Y = A.reshape(n, -1).detach().to(torch.float32)
    in_mean, in_std = X.mean(dim=0), X.std(dim=0, unbiased=False).clamp_min(1e-6)
    out_mean, out_std = Y.mean(dim=0), Y.std(dim=0, unbiased=False).clamp_min(1e-6)
    Xn = (X - in_mean) / in_std
    Yn = (Y - out_mean) / out_std

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Linear(hidden, Y.shape[1]))
    net = net.to(X.device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    losses: List[float] = []
    for _ in range(steps):
        if batch_size is None or batch_size >= n:
            xb, yb = Xn, Yn
        else:
            idx = torch.randperm(n, generator=generator)[:batch_size].to(X.device)
            xb, yb = Xn[idx], Yn[idx]
        loss = nn.functional.mse_loss(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    net.eval()
    return MLPProbe(net, in_mean, in_std, out_mean, out_std, tuple(A.shape[1:]), losses)


@dataclass
class VariantProbeResult:
    """Held-out probe metrics for one pre-training variant."""

    variant: str
    n_fit: int
    n_eval: int
    linear_mae: float
    linear_r2: float
    linear_readability: float
    mlp_mae: Optional[float] = None
    mlp_r2: Optional[float] = None
    nonlinear_gain: Optional[float] = None


@dataclass
class ProbeReport:
    """Probe metrics for several variants evaluated on the same samples, plus the fit configuration."""

    results: Dict[str, VariantProbeResult]
    config: Dict[str, Any] = field(default_factory=dict)

    def ranking(self, key: str = "linear_readability", descending: bool = True) -> List[str]:
        return sorted(self.results, key=lambda name: getattr(self.results[name], key), reverse=descending)

    def to_dict(self) -> Dict[str, Any]:
        return {"config": dict(self.config), "results": {k: asdict(v) for k, v in self.results.items()}}

    def save_json(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def cross_head_probe_report(
    H_by_variant: Mapping[str, Tensor],
    A: Tensor,
    ridge: float = 1e-3,
    per_position: bool = False,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    fit_mlp: bool = False,
    mlp_kwargs: Optional[Mapping[str, Any]] = None,
) -> ProbeReport:
    """Fit probes on hidden states from several variants (same samples, same ``A``) and compare them.

    One seeded split is shared by all variants. ``linear_readability`` is the held-out linear R^2 clipped
    to ``[0, 1]``; with ``fit_mlp=True`` the report also contains MLP metrics and
    ``nonlinear_gain = mlp_r2 - linear_r2`` (how much an MLP decodes beyond the linear probe).
    ``holdout_fraction=0`` evaluates in-sample.
    """
    if not H_by_variant:
        raise ValueError("H_by_variant must contain at least one variant")
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    n = A.shape[0]
    for name, H in H_by_variant.items():
        if H.shape[0] != n:
            raise ValueError(f"variant {name!r} has {H.shape[0]} samples but A has {n}")
    perm = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(seed))
    n_eval = int(round(n * holdout_fraction))
    if n_eval == 0:
        fit_idx = eval_idx = perm
    else:
        eval_idx, fit_idx = perm[:n_eval], perm[n_eval:]

    results: Dict[str, VariantProbeResult] = {}
    for name, H in H_by_variant.items():
        fit_i, eval_i = fit_idx.to(H.device), eval_idx.to(H.device)
        A_fit, A_eval = A[fit_i.to(A.device)], A[eval_i.to(A.device)]
        linear = fit_linear_probe(H[fit_i], A_fit, ridge=ridge, per_position=per_position).metrics(H[eval_i], A_eval)
        result = VariantProbeResult(
            variant=name,
            n_fit=int(fit_idx.numel()),
            n_eval=int(eval_idx.numel()),
            linear_mae=linear["mae"],
            linear_r2=linear["r2"],
            linear_readability=min(1.0, max(0.0, linear["r2"])),
        )
        if fit_mlp:
            mlp = fit_mlp_probe(H[fit_i], A_fit, seed=seed, **dict(mlp_kwargs or {})).metrics(H[eval_i], A_eval)
            result.mlp_mae, result.mlp_r2 = mlp["mae"], mlp["r2"]
            result.nonlinear_gain = mlp["r2"] - linear["r2"]
        results[name] = result

    config = {
        "ridge": ridge,
        "per_position": per_position,
        "holdout_fraction": holdout_fraction,
        "seed": seed,
        "fit_mlp": fit_mlp,
        "mlp_kwargs": dict(mlp_kwargs or {}),
        "n_samples": n,
        "action_shape": list(A.shape[1:]),
    }
    return ProbeReport(results=results, config=config)
