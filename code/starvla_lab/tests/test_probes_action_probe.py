import json

import pytest
import torch

from starvla_lab.probes import (
    LinearProbe,
    MLPProbe,
    ProbeReport,
    cross_head_probe_report,
    fit_linear_probe,
    fit_mlp_probe,
    mae_score,
    r2_score,
)


def _randn(*shape, seed: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _linear_chunk_data(n=512, k=4, d=16, dd=7, seed=0):
    """Hidden states at K action-query positions and actions that are an exact linear function of them."""
    H = _randn(n, k, d, seed=seed)
    W = _randn(k * d, k * dd, seed=seed + 1) / (k * d) ** 0.5
    b = _randn(k * dd, seed=seed + 2)
    A = (H.reshape(n, -1) @ W + b).reshape(n, k, dd)
    return H, A, W, b


def test_linear_probe_recovers_known_map_flatten():
    H, A, W, b = _linear_chunk_data()
    probe = fit_linear_probe(H, A)
    assert isinstance(probe, LinearProbe) and probe.weight.shape == (64, 28) and probe.bias.shape == (28,)
    assert probe.mae(H, A) < 1e-3
    assert probe.r2(H, A) > 0.999
    assert torch.allclose(probe.weight, W.double(), atol=1e-3)
    H_new = _randn(128, 4, 16, seed=99)
    A_new = (H_new.reshape(128, -1) @ W + b).reshape(128, 4, 7)
    pred = probe.predict(H_new)
    assert pred.shape == A_new.shape
    assert mae_score(pred, A_new) < 1e-3


def test_linear_probe_per_position_recovers_positionwise_map():
    n, k, d, dd = 400, 3, 10, 5
    H = _randn(n, k, d, seed=1)
    Wk = _randn(k, d, dd, seed=2)
    bk = _randn(k, dd, seed=3)
    A = torch.einsum("nkd,kdo->nko", H, Wk) + bk
    probe = fit_linear_probe(H, A, per_position=True)
    assert probe.weight.shape == (k, d, dd) and probe.bias.shape == (k, dd)
    assert probe.mae(H, A) < 1e-3
    assert torch.allclose(probe.weight, Wk.double(), atol=1e-3)
    # A given flat as [N, K*D] is split by position and predictions come back flat
    probe_flat_targets = fit_linear_probe(H, A.reshape(n, -1), per_position=True)
    assert probe_flat_targets.predict(H).shape == (n, k * dd)
    assert probe_flat_targets.mae(H, A.reshape(n, -1)) < 1e-3
    with pytest.raises(ValueError):
        fit_linear_probe(H.reshape(n, -1), A, per_position=True)
    with pytest.raises(ValueError):
        fit_linear_probe(H, A[:-1])


def test_dual_form_when_fewer_samples_than_features_fits_training_data():
    H = _randn(32, 64, seed=4)
    A = H @ _randn(64, 3, seed=5)
    probe = fit_linear_probe(H, A, ridge=1e-6)
    assert probe.weight.shape == (64, 3)
    assert probe.mae(H, A) < 1e-3
    assert probe.r2(H, A) > 0.999


def test_metric_helpers():
    t = _randn(50, 4, seed=6)
    assert r2_score(t, t) == 1.0 and mae_score(t, t) == 0.0
    assert r2_score(torch.zeros_like(t) + t.mean(0), t) == pytest.approx(0.0, abs=1e-9)
    assert r2_score(_randn(50, 4, seed=7), t) < 0.5
    assert mae_score(t + 1.0, t) == pytest.approx(1.0)


def test_mlp_probe_is_deterministic_and_fits():
    H = _randn(512, 8, seed=8)
    A = torch.tanh(H @ _randn(8, 4, seed=9))
    p1 = fit_mlp_probe(H, A, hidden=64, steps=300, lr=1e-2, seed=0)
    p2 = fit_mlp_probe(H, A, hidden=64, steps=300, lr=1e-2, seed=0)
    assert isinstance(p1, MLPProbe)
    assert torch.equal(p1.predict(H), p2.predict(H))
    assert p1.losses[-1] < p1.losses[0]
    assert p1.mae(H, A) < 0.1 and p1.r2(H, A) > 0.9
    linear = fit_linear_probe(H, A)
    assert p1.mae(H, A) < linear.mae(H, A)
    mini = fit_mlp_probe(H, A, hidden=32, steps=50, batch_size=64, seed=1)
    assert len(mini.losses) == 50 and mini.predict(H).shape == A.shape


def test_cross_head_report_ranks_linear_variant_first_and_round_trips_json(tmp_path):
    H_lin, A, _, _ = _linear_chunk_data(n=300, seed=10)
    H_by_variant = {
        "vlact_full": H_lin,
        "noisy": H_lin + 0.5 * _randn(*H_lin.shape, seed=11),
        "scrambled": _randn(*H_lin.shape, seed=12),
    }
    report = cross_head_probe_report(H_by_variant, A, holdout_fraction=0.25, seed=3)
    assert isinstance(report, ProbeReport)
    assert report.ranking() == ["vlact_full", "noisy", "scrambled"]
    res = report.results
    assert res["vlact_full"].linear_mae < 1e-3 and res["vlact_full"].linear_readability > 0.999
    assert res["scrambled"].linear_readability == 0.0 and res["scrambled"].linear_r2 <= 0.05
    assert res["vlact_full"].n_fit == 225 and res["vlact_full"].n_eval == 75
    assert all(r.mlp_mae is None for r in res.values())

    path = report.save_json(tmp_path / "probes" / "report.json")
    loaded = json.loads(path.read_text())
    assert loaded == report.to_dict()
    assert loaded["config"]["action_shape"] == [4, 7] and loaded["results"]["noisy"]["variant"] == "noisy"


def test_cross_head_report_with_mlp_and_in_sample_eval():
    H_lin, A, _, _ = _linear_chunk_data(n=200, seed=20)
    report = cross_head_probe_report(
        {"a": H_lin, "b": _randn(*H_lin.shape, seed=21)},
        A,
        holdout_fraction=0.0,
        fit_mlp=True,
        mlp_kwargs={"hidden": 32, "steps": 20},
    )
    for r in report.results.values():
        assert r.n_fit == r.n_eval == 200
        assert r.mlp_mae is not None and r.mlp_r2 is not None
        assert r.nonlinear_gain == pytest.approx(r.mlp_r2 - r.linear_r2)
    with pytest.raises(ValueError):
        cross_head_probe_report({"a": H_lin[:-1]}, A)
    with pytest.raises(ValueError):
        cross_head_probe_report({}, A)
