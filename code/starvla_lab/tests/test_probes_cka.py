import pytest
import torch

from starvla_lab.probes import layerwise_cka, linear_cka


def _randn(*shape, seed: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_same_representation_gives_one_in_both_forms():
    X_feature = _randn(256, 32, seed=0)  # N > d -> feature form
    X_kernel = _randn(16, 64, seed=1)  # N <= d -> kernel form
    assert abs(linear_cka(X_feature, X_feature).item() - 1.0) < 1e-6
    assert abs(linear_cka(X_kernel, X_kernel).item() - 1.0) < 1e-6
    assert abs(linear_cka(X_feature, X_feature, chunk_size=50).item() - 1.0) < 1e-6


def test_invariant_to_orthogonal_rotation_isotropic_scale_and_shift():
    X = _randn(300, 24, seed=2)
    Q, _ = torch.linalg.qr(_randn(24, 24, seed=3))
    Y = 3.7 * X @ Q + 0.5
    assert abs(linear_cka(X, Y).item() - 1.0) < 1e-6
    assert abs(linear_cka(X, Y, chunk_size=64).item() - 1.0) < 1e-6
    Xk = _randn(20, 40, seed=4)
    Qk, _ = torch.linalg.qr(_randn(40, 40, seed=5))
    assert abs(linear_cka(Xk, 0.01 * Xk @ Qk - 2.0).item() - 1.0) < 1e-6


def test_independent_random_matrices_are_far_below_one():
    X = _randn(512, 32, seed=6)
    Y = _randn(512, 32, seed=7)
    value = linear_cka(X, Y).item()
    assert 0.0 <= value < 0.2
    # a non-orthogonal but informative transform sits in between
    Z = X @ _randn(32, 16, seed=8) + 0.3 * _randn(512, 16, seed=9)
    assert 0.2 < linear_cka(X, Z).item() < 1.0


def test_chunked_kernel_and_feature_forms_agree():
    X = _randn(500, 20, seed=10)
    Y = 0.5 * X @ _randn(20, 20, seed=11) + _randn(500, 20, seed=12)
    direct = linear_cka(X, Y)
    for chunk in (1, 7, 64, 499, 500, 10_000):
        assert torch.allclose(direct, linear_cka(X, Y, chunk_size=chunk), atol=1e-9)
    Xs, Ys = _randn(40, 64, seed=13), _randn(40, 48, seed=14)
    assert torch.allclose(linear_cka(Xs, Ys), linear_cka(Xs, Ys, chunk_size=8), atol=1e-9)


def test_trailing_dims_are_flattened():
    X = _randn(64, 5, 8, seed=15)
    Y = _randn(64, 40, seed=16)
    assert torch.allclose(linear_cka(X, Y), linear_cka(X.reshape(64, -1), Y))


def test_layerwise_cka_and_errors():
    reps_a = [_randn(128, 8, seed=s) for s in (20, 21, 22)]
    reps_b = [reps_a[0], reps_a[1] @ torch.linalg.qr(_randn(8, 8, seed=23))[0], _randn(128, 8, seed=24)]
    out = layerwise_cka(reps_a, reps_b)
    assert out.shape == (3,) and out.dtype == torch.float64
    assert torch.allclose(out[:2], torch.ones(2, dtype=torch.float64), atol=1e-6)
    assert out[2] < 0.5
    assert layerwise_cka([], []).shape == (0,)
    with pytest.raises(ValueError):
        layerwise_cka(reps_a, reps_b[:2])
    with pytest.raises(ValueError):
        linear_cka(_randn(10, 4, seed=25), _randn(11, 4, seed=26))
    with pytest.raises(ValueError):
        linear_cka(_randn(8, seed=27), _randn(8, seed=28))
