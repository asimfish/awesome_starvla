"""starvla_lab.data.decoder_gc: wrap StarVLA's video decode functions with a post-call gc.collect."""
from __future__ import annotations

import sys
import types

import pytest

from starvla_lab.data.decoder_gc import DecoderGC, install_decoder_gc


def _fake_module(name: str, monkeypatch) -> types.ModuleType:
    mod = types.ModuleType(name)
    calls = []
    mod.calls = calls
    mod.get_frames_by_timestamps = lambda path, ts: calls.append(("ts", path)) or [path]
    mod.get_all_frames = lambda path: calls.append(("all", path)) or [path]
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_wrapper_collects_after_every_nth_call(monkeypatch):
    collected = []
    monkeypatch.setattr("starvla_lab.data.decoder_gc.gc.collect", lambda gen=2: collected.append(gen) or 0)
    fn = DecoderGC(lambda x: x * 2, every=2, generation=1)
    assert [fn(1), fn(2), fn(3)] == [2, 4, 6]
    assert fn.calls == 3 and fn.collections == 1 and collected == [1]
    assert fn.__name__ == "<lambda>"  # functools.update_wrapper keeps the metadata


def test_wrapper_collects_even_when_decode_raises(monkeypatch):
    collected = []
    monkeypatch.setattr("starvla_lab.data.decoder_gc.gc.collect", lambda gen=2: collected.append(gen) or 0)

    def boom(_):
        raise MemoryError("ENOMEM")

    fn = DecoderGC(boom, every=1, generation=1)
    with pytest.raises(MemoryError):
        fn("x")
    assert collected == [1]


def test_wrapper_rejects_bad_parameters():
    with pytest.raises(ValueError):
        DecoderGC(lambda: None, every=0)
    with pytest.raises(ValueError):
        DecoderGC(lambda: None, generation=3)


def test_install_wraps_bound_names_idempotently(monkeypatch):
    mod = _fake_module("fake_lerobot_datasets", monkeypatch)
    targets = (("fake_lerobot_datasets", ("get_frames_by_timestamps", "get_all_frames", "missing_name")),
               ("fake_module_not_importable", ("get_all_frames",)))
    collected = []
    monkeypatch.setattr("starvla_lab.data.decoder_gc.gc.collect", lambda gen=2: collected.append(gen) or 0)

    assert install_decoder_gc(every=1, generation=1, targets=targets) == {"fake_lerobot_datasets": 2}
    assert isinstance(mod.get_frames_by_timestamps, DecoderGC) and isinstance(mod.get_all_frames, DecoderGC)
    inner = mod.get_frames_by_timestamps
    assert install_decoder_gc(every=1, targets=targets) == {}  # second install: nothing new wrapped
    assert mod.get_frames_by_timestamps is inner  # and no double wrapping

    assert mod.get_frames_by_timestamps("ep0.mp4", [0.5]) == ["ep0.mp4"]
    assert mod.get_all_frames("ep1.mp4") == ["ep1.mp4"]
    assert mod.calls == [("ts", "ep0.mp4"), ("all", "ep1.mp4")]
    assert collected == [1, 1]


def test_install_disabled_with_nonpositive_every(monkeypatch):
    mod = _fake_module("fake_lerobot_datasets_off", monkeypatch)
    original = mod.get_all_frames
    assert install_decoder_gc(every=0, targets=(("fake_lerobot_datasets_off", ("get_all_frames",)),)) == {}
    assert mod.get_all_frames is original


def test_lab_config_reads_decoder_gc_every():
    from omegaconf import OmegaConf

    from starvla_lab.train.lab_config import LabConfig

    assert LabConfig.from_cfg(OmegaConf.create({"trainer": {"lab": {"mode": "single"}}})).decoder_gc_every == 1
    assert LabConfig.from_cfg(OmegaConf.create({"trainer": {"lab": {"decoder_gc_every": 0}}})).decoder_gc_every == 0
    assert LabConfig.from_cfg(OmegaConf.create({"trainer": {}})).decoder_gc_every == 1
