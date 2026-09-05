"""Training-overhead measurement (WP6): seconds/step, samples/s and peak memory per head configuration.

Aligned with the StarVLA efficiency report (issue #158): step latency and sample
throughput are reported separately. Runs on any device; real numbers need a GPU.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

__all__ = ["OverheadResult", "measure_step_overhead", "compare_configs", "HeadDropoutSchedule", "write_overhead_csv"]


@dataclass
class OverheadResult:
    name: str
    steps: int
    batch_size: int
    sec_per_step: float
    samples_per_sec: float
    peak_mem_mb: float
    device: str

    def to_row(self) -> Dict[str, str]:
        return {k: (f"{v:.4f}" if isinstance(v, float) else str(v)) for k, v in asdict(self).items()}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_step_overhead(
    name: str,
    step_fn: Callable[[int], torch.Tensor],
    batch_size: int,
    steps: int = 10,
    warmup: int = 2,
    device: Optional[torch.device] = None,
) -> OverheadResult:
    """Time ``step_fn`` (one forward+backward returning the loss) after ``warmup`` iterations."""
    device = device or torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for i in range(warmup):
        step_fn(i)
    _sync(device)
    t0 = time.perf_counter()
    for i in range(steps):
        loss = step_fn(warmup + i)
        if loss.requires_grad:
            loss.backward()
    _sync(device)
    elapsed = time.perf_counter() - t0
    sec = elapsed / max(steps, 1)
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    return OverheadResult(name, steps, batch_size, sec, batch_size / sec if sec > 0 else float("inf"), peak, str(device))


def compare_configs(
    builders: Mapping[str, Callable[[], Callable[[int], torch.Tensor]]],
    batch_size: int,
    steps: int = 10,
    warmup: int = 2,
    device: Optional[torch.device] = None,
) -> List[OverheadResult]:
    """Measure several named configurations; each builder returns a fresh ``step_fn``."""
    results = []
    for name, build in builders.items():
        step_fn = build()
        results.append(measure_step_overhead(name, step_fn, batch_size, steps, warmup, device))
    return results


class HeadDropoutSchedule:
    """Deterministically pick which heads are active at each step (cost-reduction option for multi-head pre-training).

    ``p_all`` is the probability that all heads are active; otherwise exactly one head is
    sampled uniformly. With ``p_all=1`` every head is always active.
    """

    def __init__(self, heads: Sequence[str], p_all: float = 0.5, seed: int = 0):
        if not heads:
            raise ValueError("heads must be non-empty")
        if not 0.0 <= p_all <= 1.0:
            raise ValueError("p_all must be in [0, 1]")
        self.heads = tuple(heads)
        self.p_all = p_all
        self._gen = torch.Generator().manual_seed(seed)

    def active(self, step: int) -> tuple:
        if self.p_all >= 1.0 or torch.rand(1, generator=self._gen).item() < self.p_all:
            return self.heads
        idx = int(torch.randint(len(self.heads), (1,), generator=self._gen).item())
        return (self.heads[idx],)

    def expected_head_evals_per_step(self) -> float:
        return self.p_all * len(self.heads) + (1.0 - self.p_all) * 1.0


def write_overhead_csv(results: Iterable[OverheadResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_row() for r in results]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)
    return path
