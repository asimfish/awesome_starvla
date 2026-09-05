import copy

import pytest
import torch
from torch import nn

from starvla_lab.probes import DriftRecord, DriftTracker, drift_to_llrd_decay


class _Stack(nn.Module):
    """Tiny residual stack standing in for a VLM backbone; ``extract`` returns every layer's output."""

    def __init__(self, dim=8, n_layers=4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def extract(self, x):
        reps = []
        for layer in self.layers:
            x = x + torch.tanh(layer(x))
            reps.append(x)
        return reps


def _extract_fn(model, batch):
    return model.extract(batch)


@pytest.fixture
def setup():
    ref = _Stack()
    cur = copy.deepcopy(ref)
    batch = torch.randn(64, 8, generator=torch.Generator().manual_seed(1))
    return ref, cur, batch


def test_identical_model_has_zero_drift(setup):
    ref, cur, batch = setup
    tracker = DriftTracker(_extract_fn, batch, reference=ref)
    drift = tracker.update(cur, step=0)
    assert drift.shape == (4,)
    assert torch.all(drift < 1e-6)
    assert tracker.latest() is drift
    assert tracker.summary()["mean"] < 1e-6 and tracker.summary()["step"] == 0


def test_perturbing_a_layer_drifts_it_and_downstream_layers_only(setup):
    ref, cur, batch = setup
    tracker = DriftTracker(_extract_fn, batch, reference=ref, layer_names=["l0", "l1", "l2", "l3"])
    delta = torch.randn(cur.layers[2].weight.shape, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        cur.layers[2].weight.add_(delta * 3.0)
    drift = tracker.update(cur, step=10)
    assert torch.all(drift[:2] < 1e-6)
    assert drift[2] > 1e-3 and drift[3] > 1e-3
    assert torch.all(drift <= 1.0) and torch.all(drift >= 0.0)
    summary = tracker.summary()
    assert summary["max_layer"] in ("l2", "l3") and summary["max"] == max(summary["per_layer"].values())
    assert set(summary["per_layer"]) == {"l0", "l1", "l2", "l3"}
    # Same perturbation direction, larger magnitude -> larger drift (deterministic, unlike a fresh random draw).
    with torch.no_grad():
        cur.layers[2].weight.copy_(ref.layers[2].weight + delta * 9.0)
    drift2 = tracker.update(cur, step=20)
    assert drift2[2] > drift[2]
    assert tracker.per_layer().shape == (2, 4)


def test_history_shapes_steps_and_empty_summary(setup):
    ref, cur, batch = setup
    tracker = DriftTracker(_extract_fn, batch, reference=ref)
    assert tracker.latest() is None and tracker.per_layer().shape == (0, 4)
    empty = tracker.summary()
    assert empty["num_updates"] == 0 and empty["mean"] == 0.0 and empty["max_layer"] is None
    for step in (0, 5, 10):
        tracker.update(cur, step=step)
    assert tracker.steps() == [0, 5, 10]
    assert tracker.per_layer().shape == (3, 4)
    assert all(isinstance(rec, DriftRecord) for rec in tracker.history)
    assert tracker.summary()["num_updates"] == 3


def test_precomputed_reference_and_layer_selection(setup):
    ref, cur, batch = setup
    with torch.no_grad():
        ref_reps = ref.extract(batch)
    tracker = DriftTracker(_extract_fn, batch, reference=ref_reps, layers=[1, 3])
    assert tracker.num_layers == 2 and tracker.layer_names == ["layer_1", "layer_3"]
    with torch.no_grad():
        cur.layers[3].weight.mul_(-4.0)
    drift = tracker.update(cur)
    assert drift.shape == (2,) and drift[0] < 1e-6 < drift[1]
    assert tracker.summary()["step"] is None
    with pytest.raises(ValueError):
        DriftTracker(_extract_fn, batch, reference=ref, layer_names=["only_one"])
    with pytest.raises(ValueError):  # extract_fn yields 4 layers, reference has 3
        DriftTracker(_extract_fn, batch, reference=ref_reps[:3]).update(cur)
    three = DriftTracker(lambda m, b: m.extract(b)[:3], batch, reference=ref_reps[:3])
    assert three.update(cur).shape == (3,)


def test_drift_to_llrd_decay_is_monotone_and_bounded():
    drift = torch.tensor([0.0, 0.05, 0.1, 0.2, 0.4, 0.5, 0.9, 5.0])
    mult = drift_to_llrd_decay(drift, base_decay=1.0, min_decay=0.1, drift_max=0.5)
    assert mult.shape == drift.shape
    assert torch.all(mult[1:] <= mult[:-1])
    assert mult[0] == pytest.approx(1.0) and mult[5] == pytest.approx(0.1) and mult[-1] == pytest.approx(0.1)
    assert torch.all(mult >= 0.1) and torch.all(mult <= 1.0)
    assert mult[2] == pytest.approx(0.1 + 0.9 * 0.8)
    sharp = drift_to_llrd_decay(drift, base_decay=0.9, min_decay=0.3, drift_max=0.5, power=2.0)
    assert sharp[0] == pytest.approx(0.9) and torch.all(sharp[1:] <= sharp[:-1]) and torch.all(sharp >= 0.3)
    assert drift_to_llrd_decay([0.1, 0.2]).shape == (2,)
    with pytest.raises(ValueError):
        drift_to_llrd_decay(drift, base_decay=0.5, min_decay=0.6)
    with pytest.raises(ValueError):
        drift_to_llrd_decay(drift, drift_max=0.0)
