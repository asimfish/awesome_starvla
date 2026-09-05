# Copyright 2026 awesome_starvla contributors. MIT License.
"""Keyframe (event) prediction head and its inference-side write policy (WP3 (b)).

Follows the EventVLA KEM protocol without reusing its code: chunk-level probabilities that step
``t0 + i`` is a task-critical event, Gaussian-smoothed soft labels with sequence-averaged BCE, and
``threshold -> 1-D NMS -> cooldown -> bounded FIFO`` when turning probabilities into discrete writes
to a raw-image evidence memory. A teacher-to-student curriculum decides how often training uses the
ground-truth memory instead of the model's own writes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ProbsLike = Union[torch.Tensor, np.ndarray, Sequence[float]]


class KeyframeHead(nn.Module):
    """Per-step keyframe logits ``[B, H]`` from action-query hidden states ``[B, K, d]``.

    One logit per query position (``H = K``). When ``horizon`` is given and differs from K the K
    logits are resampled to ``horizon`` steps with adaptive average pooling.
    """

    def __init__(self, hidden_dim: int, horizon: Optional[int] = None, *, mlp_hidden: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        if horizon is not None and int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = None if horizon is None else int(horizon)
        mlp_hidden = int(mlp_hidden or hidden_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"h must be [B, K, d], got {tuple(h.shape)}")
        logits = self.mlp(h).squeeze(-1)
        if self.horizon is not None and self.horizon != logits.shape[1]:
            logits = F.adaptive_avg_pool1d(logits.unsqueeze(1), self.horizon).squeeze(1)
        return logits

    def probabilities(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(h))


def soft_keyframe_labels(
    event_steps: Sequence[Sequence[int]],
    horizon: int,
    sigma: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sparse keyframe offsets (one integer list per sample) -> ``[B, horizon]`` soft labels.

    Each event contributes a Gaussian bump of peak 1 (``sigma <= 0`` gives a one-hot); overlapping
    bumps are combined with ``max`` and the result is clipped to ``[0, 1]``. Offsets outside
    ``[0, horizon)`` are dropped, so a sample without in-range events gets all zeros.
    """
    horizon = int(horizon)
    labels = torch.zeros(len(event_steps), horizon, dtype=dtype, device=device)
    positions = torch.arange(horizon, dtype=torch.float32, device=device)
    for row, steps in enumerate(event_steps):
        for step in steps:
            step = int(step)
            if not 0 <= step < horizon:
                continue
            if sigma > 0:
                bump = torch.exp(-((positions - step) ** 2) / (2.0 * float(sigma) ** 2))
            else:
                bump = (positions == step).to(torch.float32)
            labels[row] = torch.maximum(labels[row], bump.to(dtype))
    return labels.clamp_(0.0, 1.0)


def keyframe_bce_loss(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """Sequence-averaged BCE-with-logits over ``[B, H]``; ``mask`` is per sample ``[B]`` or per step ``[B, H]``.

    Computed in fp32; an all-False mask returns 0 without NaN.
    """
    if logits.shape != soft_labels.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and soft_labels {tuple(soft_labels.shape)} must match")
    logits32, labels32 = logits.float(), soft_labels.float()
    weight = None if pos_weight is None else torch.full((logits32.shape[-1],), float(pos_weight), device=logits32.device)
    bce = F.binary_cross_entropy_with_logits(logits32, labels32, pos_weight=weight, reduction="none")
    if mask is None:
        return bce.mean()
    mask = mask.to(device=bce.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask[:, None].expand_as(bce)
    valid = mask.to(bce.dtype)
    return (bce * valid).sum() / valid.sum().clamp_min(1.0)


def _as_prob_vector(probs: ProbsLike) -> np.ndarray:
    if isinstance(probs, torch.Tensor):
        array = probs.detach().float().cpu().numpy()
    else:
        array = np.asarray(probs, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"probs must be a 1-D sequence [H], got shape {array.shape}")
    return array


def nms_1d(probs: ProbsLike, threshold: float, window: int) -> List[int]:
    """Indices above ``threshold`` that are the maximum within ``+-window`` steps, in temporal order.

    Greedy: candidates are visited by descending probability (earlier index wins ties) and each kept
    peak suppresses the others within its window. ``window=0`` keeps every candidate.
    """
    array = _as_prob_vector(probs)
    candidates = [i for i in range(array.shape[0]) if array[i] >= threshold]
    candidates.sort(key=lambda i: (-float(array[i]), i))
    suppressed = np.zeros(array.shape[0], dtype=bool)
    kept: List[int] = []
    for i in candidates:
        if suppressed[i]:
            continue
        kept.append(i)
        suppressed[max(0, i - window) : i + window + 1] = True
    return sorted(kept)


class KeyframeWritePolicy:
    """Turns a chunk of keyframe probabilities into discrete absolute write steps.

    ``decide(probs [H], t0)`` applies ``threshold -> 1-D NMS(nms_window) -> cooldown`` (a write must
    be at least ``cooldown`` steps away from every remembered write; duplicates are always dropped)
    and records the accepted steps in a FIFO of at most ``max_events`` entries, mirroring the bounded
    evidence memory. The policy is stateful across calls; use ``reset()`` at episode boundaries.
    """

    def __init__(self, threshold: float = 0.5, nms_window: int = 0, cooldown: int = 0, max_events: Optional[int] = None) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if nms_window < 0 or cooldown < 0:
            raise ValueError("nms_window and cooldown must be >= 0")
        if max_events is not None and int(max_events) < 1:
            raise ValueError(f"max_events must be >= 1 or None, got {max_events}")
        self.threshold = float(threshold)
        self.nms_window = int(nms_window)
        self.cooldown = int(cooldown)
        self.max_events = None if max_events is None else int(max_events)
        self._events: Deque[int] = deque(maxlen=self.max_events)

    @property
    def events(self) -> List[int]:
        """Remembered write steps, oldest first."""
        return list(self._events)

    def reset(self) -> None:
        self._events.clear()

    def _blocked(self, step: int, pending: Sequence[int]) -> bool:
        for other in list(self._events) + list(pending):
            if other == step or abs(step - other) < self.cooldown:
                return True
        return False

    def decide(self, probs: ProbsLike, t0: int = 0) -> List[int]:
        """Absolute steps ``t0 + i`` to write for the chunk ``probs``, in temporal order."""
        writes: List[int] = []
        for offset in nms_1d(probs, self.threshold, self.nms_window):
            step = int(t0) + offset
            if self._blocked(step, writes):
                continue
            writes.append(step)
        for step in writes:
            self._events.append(step)
        return writes


class EvidenceMemory:
    """Bounded FIFO of ``(timestep, image)`` evidence entries."""

    def __init__(self, max_events: int) -> None:
        if int(max_events) < 1:
            raise ValueError(f"max_events must be >= 1, got {max_events}")
        self.max_events = int(max_events)
        self._buffer: Deque[Tuple[int, Any]] = deque(maxlen=self.max_events)

    def write(self, timestep: int, image: Any) -> Optional[Tuple[int, Any]]:
        """Append an entry; returns the evicted oldest entry when the memory was full."""
        evicted = self._buffer[0] if len(self._buffer) == self.max_events else None
        self._buffer.append((int(timestep), image))
        return evicted

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def entries(self) -> List[Tuple[int, Any]]:
        return list(self._buffer)

    @property
    def timesteps(self) -> List[int]:
        return [step for step, _ in self._buffer]

    @property
    def images(self) -> List[Any]:
        return [image for _, image in self._buffer]


@dataclass(frozen=True)
class TeacherStudentCurriculum:
    """Probability of filling the training memory from ground-truth keyframes at a given step.

    Stays at ``start`` for ``warmup`` steps, moves linearly to ``end`` over ``transition`` steps
    (default: the rest of ``total_steps``) and stays there.
    """

    total_steps: int
    warmup: int = 0
    transition: Optional[int] = None
    start: float = 1.0
    end: float = 0.0

    def __post_init__(self) -> None:
        if self.total_steps < 0 or self.warmup < 0:
            raise ValueError("total_steps and warmup must be >= 0")
        if self.transition is not None and self.transition < 0:
            raise ValueError("transition must be >= 0 or None")
        if not (0.0 <= self.start <= 1.0 and 0.0 <= self.end <= 1.0):
            raise ValueError("start and end must be probabilities in [0, 1]")

    @property
    def transition_steps(self) -> int:
        if self.transition is not None:
            return int(self.transition)
        return max(int(self.total_steps) - int(self.warmup), 0)

    def progress(self, step: int) -> float:
        """Fraction of the teacher-to-student transition completed at ``step``, in ``[0, 1]``."""
        elapsed = int(step) - int(self.warmup)
        if elapsed < 0:
            return 0.0
        if self.transition_steps <= 0:
            return 1.0
        return min(elapsed / float(self.transition_steps), 1.0)

    def teacher_prob(self, step: int) -> float:
        prob = self.start + (self.end - self.start) * self.progress(step)
        return min(max(float(prob), 0.0), 1.0)

    __call__ = teacher_prob
