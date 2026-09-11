"""Plug the PyAV / libdav1d decoder-thread leak in StarVLA's LeRobot video path (F5 hang, 2026-09-09).

StarVLA's ``gr00t_lerobot/video.py`` (``video_backend: torchvision_av`` or ``pyav``) opens a PyAV container for
every decode. ``InputContainer`` <-> ``Stream`` <-> ``CodecContext`` form a reference cycle, so ``container.close()``
does not free the codec context: its libdav1d worker pool (16 ``dav1d-worker`` threads on a 128-core node) lives
until the cyclic GC collects generation 1 or 2. Inside a DataLoader worker with a large long-lived heap those
collections become rare; the F5 OFT run leaked 2000-3200 threads per worker within 374 steps and then deadlocked
(worker stuck in ``poll``, main process waiting forever on the result queue, GPU at 0 % for 69 h).

Measured on the node with the LIBERO AV1 videos (64 decodes): no GC -> +640 threads (peak +1400);
``gc.collect(1)`` after every decode -> flat, ~2 ms per call against ~20 ms per decode.

:func:`install_decoder_gc` wraps the decode functions where they are *bound* -- ``datasets.py`` imports them by name
at import time, so patching ``video.py`` alone would miss the hot path -- and must run in the main process before
the DataLoader forks its workers (fork start method: the workers inherit the patched modules).
"""
from __future__ import annotations

import functools
import gc
import importlib
from typing import Callable, Dict, Iterable, Sequence, Tuple

__all__ = ["DEFAULT_TARGETS", "DecoderGC", "install_decoder_gc"]

# (module, decode function names bound in that module)
DEFAULT_TARGETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("starVLA.dataloader.gr00t_lerobot.datasets", ("get_frames_by_timestamps", "get_all_frames")),
    ("starVLA.dataloader.gr00t_lerobot.video", ("get_frames_by_timestamps", "get_all_frames", "get_frames_by_indices")),
)


class DecoderGC:
    """Call ``fn`` and run ``gc.collect(generation)`` after every ``every``-th call (also when ``fn`` raises)."""

    def __init__(self, fn: Callable, every: int = 1, generation: int = 1):
        if every < 1:
            raise ValueError("every must be >= 1")
        if generation not in (0, 1, 2):
            raise ValueError("generation must be 0, 1 or 2")
        self.fn, self.every, self.generation, self.calls, self.collections = fn, int(every), int(generation), 0, 0
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        try:
            return self.fn(*args, **kwargs)
        finally:
            self.calls += 1
            if self.calls % self.every == 0:
                gc.collect(self.generation)
                self.collections += 1


def install_decoder_gc(every: int = 1, generation: int = 1,
                       targets: Iterable[Tuple[str, Sequence[str]]] = DEFAULT_TARGETS) -> Dict[str, int]:
    """Wrap the decode functions in place. Idempotent: an already wrapped name is left alone. Modules that are not
    importable (no StarVLA on the path) are skipped. Returns ``{module: names wrapped}``; ``every <= 0`` disables
    the patch and returns ``{}``."""
    if every <= 0:
        return {}
    wrapped: Dict[str, int] = {}
    for module_name, names in targets:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        n = 0
        for name in names:
            fn = getattr(module, name, None)
            if fn is None or isinstance(fn, DecoderGC):
                continue
            setattr(module, name, DecoderGC(fn, every=every, generation=generation))
            n += 1
        if n:
            wrapped[module_name] = n
    return wrapped
