import random

import pytest

from starvla_lab.schedules import STRATEGIES, AuxDataScheduler


def test_strategies_and_validation():
    assert set(STRATEGIES) >= {"fixed", "linear", "drift"}
    with pytest.raises(ValueError):
        AuxDataScheduler(strategy="bogus")
    with pytest.raises(ValueError):
        AuxDataScheduler(strategy="linear")  # needs total_steps
    with pytest.raises(ValueError):
        AuxDataScheduler(ratio_min=0.6, ratio_max=0.5)


def test_fixed_strategy_is_constant():
    s = AuxDataScheduler("fixed", ratio_min=0.1, ratio_max=0.5, loss_scale_min=0.1, loss_scale_max=1.0, init_u=1.0)
    outs = [s.step(i) for i in range(0, 5000, 1000)]
    assert all(o == (pytest.approx(0.5), pytest.approx(1.0)) for o in outs)
    prob, scale = s.step(9999)
    assert prob == pytest.approx(0.5) and scale == pytest.approx(1.0)


def test_linear_strategy_anneals_from_max_to_min():
    s = AuxDataScheduler("linear", ratio_min=0.1, ratio_max=0.5, loss_scale_min=0.2, loss_scale_max=1.0, total_steps=100)
    p0, l0 = s.step(0)
    p50, l50 = s.step(50)
    p100, l100 = s.step(100)
    p_over, _ = s.step(500)
    assert (p0, l0) == (pytest.approx(0.5), pytest.approx(1.0))
    assert (p50, l50) == (pytest.approx(0.3), pytest.approx(0.6))
    assert (p100, l100) == (pytest.approx(0.1), pytest.approx(0.2))
    assert p_over == pytest.approx(0.1)


def test_drift_strategy_moves_with_hysteresis_and_bounds():
    s = AuxDataScheduler("drift", ratio_min=0.1, ratio_max=0.5, loss_scale_min=0.1, loss_scale_max=1.0, init_u=0.5, drift_high=0.1, drift_low=0.05, gain=2.0, max_step=0.1)
    base = s.step(0)                       # no drift supplied -> unchanged
    assert base == (pytest.approx(0.3), pytest.approx(0.55))
    assert s.step(1, drift=0.07) == base   # inside the hysteresis band -> hold
    up = s.step(2, drift=0.9)              # far above high -> +max_step on u
    assert up[0] == pytest.approx(0.3 + 0.1 * 0.4) and up[1] > base[1]
    for i in range(3, 40):
        s.step(i, drift=0.9)
    assert s.step(40, drift=0.9) == (pytest.approx(0.5), pytest.approx(1.0))  # saturates at max
    for i in range(41, 80):
        s.step(i, drift=0.0)
    assert s.step(80, drift=0.0) == (pytest.approx(0.1), pytest.approx(0.1))  # saturates at min


def test_sample_vlm_matches_probability_and_apply_to_cfg_writes_keys():
    s = AuxDataScheduler("fixed", ratio_min=0.25, ratio_max=0.25, loss_scale_min=0.5, loss_scale_max=0.5)
    rng = random.Random(0)
    hits = sum(s.sample_vlm(rng) for _ in range(4000))
    assert abs(hits / 4000 - 0.25) < 0.03
    cfg = {"trainer": {"loss_scale": {"vla": 1.0, "vlm": 1.0}}}
    prob, scale = s.apply_to_cfg(cfg)
    assert cfg["trainer"]["loss_scale"]["vlm"] == pytest.approx(0.5) and scale == pytest.approx(0.5)
    assert cfg["trainer"]["vlm_sample_prob"] == pytest.approx(0.25) and prob == pytest.approx(0.25)
