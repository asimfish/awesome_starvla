# Copyright 2026 awesome_starvla contributors. MIT License.
"""Future visual-feature prediction head (WP3 (a)).

Predicts the features a frozen extractor (DINO / SigLIP pooled embeddings, ...) would produce for the
frame observed ``offset`` steps ahead, from the backbone's action-query hidden states. Predicting
features rather than pixels keeps the auxiliary head cheap while still tying the representation to
scene dynamics (the world-model argument).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

POOLINGS: Tuple[str, ...] = ("offset", "mean")


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean of ``values`` over ``mask`` (broadcast to ``values.shape``); exactly 0 for an empty mask."""
    if mask is None:
        return values.mean()
    valid = mask.to(dtype=values.dtype).expand(values.shape)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def feature_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    cosine_weight: float = 1.0,
    mse_weight: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """``cosine_weight * (1 - cos) + mse_weight * MSE`` averaged over the valid ``[B, N]`` entries.

    ``pred`` / ``target`` are ``[B, N, d_feat]``; ``valid_mask`` is ``[B, N]`` (bool). Reductions run in
    fp32 and an all-False mask yields 0 without NaN.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must have the same shape")
    pred32, target32 = pred.float(), target.float()
    cosine = 1.0 - F.cosine_similarity(pred32, target32, dim=-1, eps=eps)
    mse = (pred32 - target32).pow(2).mean(dim=-1)
    cosine_loss = masked_mean(cosine, valid_mask)
    mse_loss = masked_mean(mse, valid_mask)
    return {
        "loss": cosine_weight * cosine_loss + mse_weight * mse_loss,
        "cosine_loss": cosine_loss,
        "mse_loss": mse_loss,
    }


def targets_from_sequence(
    feats_seq: torch.Tensor,
    t: Union[int, Sequence[int], torch.Tensor],
    offsets: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice ``feats_seq[b, t_b + offset]`` for every offset of a ``[B, T, d_feat]`` feature sequence.

    ``t`` is one step for the whole batch or one per sample. Returns ``(targets [B, N, d_feat],
    valid [B, N])``; entries whose index falls outside ``[0, T)`` are zeroed and marked invalid.
    """
    feats = torch.as_tensor(feats_seq)
    if feats.ndim != 3:
        raise ValueError(f"feats_seq must be [B, T, d_feat], got {tuple(feats.shape)}")
    batch, length, _ = feats.shape
    steps = torch.as_tensor(t, dtype=torch.long, device=feats.device).reshape(-1)
    if steps.numel() == 1:
        steps = steps.expand(batch)
    elif steps.numel() != batch:
        raise ValueError(f"t must be a scalar or have {batch} entries, got {steps.numel()}")
    offs = torch.as_tensor([int(o) for o in offsets], dtype=torch.long, device=feats.device)
    index = steps[:, None] + offs[None, :]
    valid = (index >= 0) & (index < length)
    gathered = feats.gather(1, index.clamp(0, length - 1)[..., None].expand(-1, -1, feats.shape[-1]))
    targets = torch.where(valid[..., None], gathered, torch.zeros_like(gathered))
    return targets, valid


class FutureFeaturePredictionHead(nn.Module):
    """``h [B, K, d] -> [B, len(offsets), feat_dim]`` predictions of future visual features.

    ``pooling="offset"`` reads the query state at position ``offset - 1`` (the query that plans the
    action whose outcome is the target frame, clamped to ``[0, K - 1]``); ``"mean"`` averages all K
    states. A learned per-offset embedding is added so that several offsets share one MLP.
    """

    def __init__(
        self,
        hidden_dim: int,
        feat_dim: int,
        offsets: Sequence[int] = (1,),
        *,
        pooling: str = "offset",
        mlp_hidden: Optional[int] = None,
        dropout: float = 0.0,
        cosine_weight: float = 1.0,
        mse_weight: float = 1.0,
    ) -> None:
        super().__init__()
        offsets = tuple(int(o) for o in offsets)
        if not offsets or any(o < 0 for o in offsets) or len(set(offsets)) != len(offsets):
            raise ValueError(f"offsets must be non-empty, unique and >= 0, got {offsets}")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        self.offsets = offsets
        self.feat_dim = int(feat_dim)
        self.pooling = pooling
        self.cosine_weight = float(cosine_weight)
        self.mse_weight = float(mse_weight)
        mlp_hidden = int(mlp_hidden or hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.offset_embed = nn.Parameter(torch.zeros(len(offsets), hidden_dim))
        nn.init.normal_(self.offset_embed, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, self.feat_dim),
        )

    @property
    def num_offsets(self) -> int:
        return len(self.offsets)

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        """``[B, K, d] -> [B, len(offsets), d]`` according to ``pooling``."""
        if h.ndim != 3:
            raise ValueError(f"h must be [B, K, d], got {tuple(h.shape)}")
        if self.pooling == "mean":
            return h.mean(dim=1, keepdim=True).expand(-1, self.num_offsets, -1)
        chunk_len = h.shape[1]
        index = torch.tensor([min(max(o - 1, 0), chunk_len - 1) for o in self.offsets], device=h.device)
        return h.index_select(1, index)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.norm(self.pool(h))
        return self.mlp(z + self.offset_embed.to(dtype=z.dtype))

    def loss(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return feature_prediction_loss(pred, target, valid_mask, cosine_weight=self.cosine_weight, mse_weight=self.mse_weight)
