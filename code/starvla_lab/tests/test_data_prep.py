import json
from pathlib import Path

import numpy as np
import pytest

from starvla_lab.data import (
    FeatureCache,
    FunctionLabeler,
    FutureFeatureTransform,
    KeyframeLabelTransform,
    TrajectorySubset,
    build_feature_cache,
    chunk_relative_events,
    future_targets,
    heuristic_keyframe_steps,
    load_labels,
    make_fraction_hook,
    save_labels,
    select_trajectories,
)


class _FakeDataset:
    def __init__(self, lengths):
        self.trajectory_ids = np.arange(len(lengths))
        self.all_steps = [(t, s) for t, L in enumerate(lengths) for s in range(L)]
        self.tag = "franka"

    def __len__(self):
        return len(self.all_steps)

    def __getitem__(self, i):
        t, s = self.all_steps[i]
        return {"trajectory_id": t, "step": s, "action": np.zeros((4, 7))}


def test_select_trajectories_is_deterministic_and_sized():
    ids = list(range(20))
    a = select_trajectories(ids, 0.2, seed=1)
    b = select_trajectories(ids, 0.2, seed=1)
    c = select_trajectories(ids, 0.2, seed=2)
    assert a == b and len(a) == 4 and a != c
    assert select_trajectories(ids, 1.0) == ids
    assert len(select_trajectories(ids, 0.01)) == 1
    with pytest.raises(ValueError):
        select_trajectories(ids, 0.0)


def test_trajectory_subset_keeps_whole_trajectories_and_forwards_attrs():
    ds = _FakeDataset([3, 5, 2, 4])
    sub = TrajectorySubset(ds, 0.5, seed=0)
    kept = set(sub.kept_trajectories)
    assert len(kept) == 2 and all(t in kept for t, _ in sub.all_steps)
    assert len(sub) == sum(len_ for t, len_ in zip(range(4), [3, 5, 2, 4]) if t in kept)
    assert sub[0]["trajectory_id"] in kept and sub.tag == "franka"
    assert sub.summary()["trajectories_total"] == 4 and sub.summary()["steps_total"] == 14
    hook = make_fraction_hook(1.0)
    assert hook(ds) is ds
    assert isinstance(make_fraction_hook(0.5, seed=3, inner=lambda: ds)(), TrajectorySubset)


def test_feature_cache_and_future_transform(tmp_path: Path):
    def extractor(frames):
        return np.stack([np.full((3,), float(f)) for f in frames])

    trajs = [("ep0", [0, 1, 2, 3]), ("ep1", [10, 11])]
    shapes = build_feature_cache(trajs, extractor, tmp_path / "feats", batch_size=3)
    assert shapes == {"ep0": (4, 3), "ep1": (2, 3)}
    cache = FeatureCache(tmp_path / "feats")
    feats, mask = future_targets(np.asarray(cache.get("ep0")), t=2, offsets=[1, 4])
    assert mask.tolist() == [1.0, 0.0] and feats[0].tolist() == [3.0, 3.0, 3.0] and feats[1].sum() == 0
    tf = FutureFeatureTransform(cache, offsets=[1, 2])
    out = tf({"trajectory_id": "ep1", "step": 0, "lang": "x"})
    assert out["future_features"].shape == (2, 3) and out["future_features_mask"].tolist() == [1.0, 0.0]
    assert "future_features" not in tf({"trajectory_id": "missing", "step": 0})
    assert "future_features" not in tf({"lang": "no ids"})
    # cache reuse: second build does not call the extractor
    calls = []
    build_feature_cache(trajs, lambda f: calls.append(1) or extractor(f), tmp_path / "feats")
    assert calls == []


def test_heuristic_keyframes_from_gripper_and_pauses():
    T = 40
    actions = np.zeros((T, 7), dtype=np.float32)
    actions[:, 0] = np.linspace(0, 1, T)              # moving arm dim
    actions[10:25, 6] = 1.0                            # gripper closes at 10, opens at 25
    ev = heuristic_keyframe_steps(actions, gripper_dims=[6], threshold=0.5, min_gap=5)
    assert ev == [10, 25]
    actions[30:34, 0] = actions[29, 0]                 # pause
    ev2 = heuristic_keyframe_steps(actions, gripper_dims=[6], pause_dims=[0], pause_quantile=0.05, min_gap=3)
    assert 10 in ev2 and 25 in ev2 and any(30 <= e <= 34 for e in ev2)
    with pytest.raises(ValueError):
        heuristic_keyframe_steps(np.zeros(5), [0])


def test_labeler_cache_and_chunk_relative_transform(tmp_path: Path):
    labeler = FunctionLabeler(lambda frames, instr: [3, 3, 9, 30, -1], min_gap=2)
    events = labeler(list(range(20)), "open the drawer")
    assert events == [3, 9]
    save_labels(tmp_path / "kf", "ep7", events, meta={"source": "test"})
    assert load_labels(tmp_path / "kf", "ep7") == [3, 9] and load_labels(tmp_path / "kf", "nope") is None
    assert json.loads((tmp_path / "kf" / "ep7.json").read_text())["meta"]["source"] == "test"
    assert chunk_relative_events([3, 9, 20], t=2, horizon=8) == [1, 7]
    tf = KeyframeLabelTransform(tmp_path / "kf", horizon=8)
    assert tf({"trajectory_id": "ep7", "step": 2})["keyframe_steps"] == [1, 7]
    assert tf({"trajectory_id": "ep7", "step": 15})["keyframe_steps"] == []
    assert "keyframe_steps" not in tf({"trajectory_id": "unknown", "step": 0})


def test_install_fraction_hook_patches_named_factory():
    import types

    from starvla_lab.data import install_fraction_hook

    ds = _FakeDataset([3, 5, 2, 4])
    fake_module = types.SimpleNamespace(make_LeRobotSingleDataset=lambda *a, **k: ds)
    uninstall = install_fraction_hook(0.5, seed=0, module=fake_module)
    wrapped = fake_module.make_LeRobotSingleDataset("root", "name", "franka")
    assert isinstance(wrapped, TrajectorySubset) and len(wrapped) < len(ds)
    uninstall()
    assert fake_module.make_LeRobotSingleDataset("root", "name", "franka") is ds
    assert install_fraction_hook(1.0, module=fake_module)() is None and fake_module.make_LeRobotSingleDataset("r", "n", "f") is ds
