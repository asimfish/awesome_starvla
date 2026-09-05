import random

import pytest
import torch
from torch import nn

from starvla_lab.schedules import DEFAULT_LLM_LAYERS_PATH, DriftDrivenLLRD, layer_group_index, layerwise_lr_decay_groups
from vlact_ext.freeze_rules import resolve_frozen_param_ids


class _Holder(nn.Module):
    def __init__(self, **children):
        super().__init__()
        for k, v in children.items():
            setattr(self, k, v)


class _LM(nn.Module):
    def __init__(self, dim=4, n=6):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n)])
        self.norm = nn.LayerNorm(dim)


def make_toy(n_layers=6, dim=4):
    inner = _Holder(visual=nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)), language_model=_LM(dim, n_layers))
    hf = _Holder(model=inner, lm_head=nn.Linear(dim, 10))
    return _Holder(qwen_vl_interface=_Holder(model=hf), heads=nn.ModuleDict({"oft": nn.Linear(dim, 7)}))


def _group_lr_by_layer(groups):
    return {int(g["layer_index"]): g["lr"] for g in groups if "layer_index" in g}


def test_layer_lrs_increase_with_depth_and_heads_get_head_lr():
    model = make_toy(6)
    groups = layerwise_lr_decay_groups(model, base_lr=1e-4, decay=0.8, head_lr=1e-3)
    by_layer = _group_lr_by_layer(groups)
    assert sorted(by_layer) == list(range(6))
    lrs = [by_layer[i] for i in range(6)]
    assert all(lrs[i] < lrs[i + 1] for i in range(5))
    assert lrs[-1] == pytest.approx(1e-4 * 0.8)
    head_params = {id(p) for p in model.heads.parameters()}
    head_groups = [g for g in groups if any(id(p) in head_params for p in g["params"])]
    assert head_groups and all(g["lr"] == pytest.approx(1e-3) for g in head_groups)
    covered = {id(p) for g in groups for p in g["params"]}
    assert covered == {id(p) for p in model.parameters()}


def test_visual_and_embed_use_deepest_decay():
    model = make_toy(4)
    groups = layerwise_lr_decay_groups(model, base_lr=1e-4, decay=0.5)
    by_layer = _group_lr_by_layer(groups)
    visual_ids = {id(p) for p in model.qwen_vl_interface.model.model.visual.parameters()}
    visual_lr = [g["lr"] for g in groups if any(id(p) in visual_ids for p in g["params"])]
    assert visual_lr and all(lr <= min(by_layer.values()) for lr in visual_lr)


def test_frozen_params_are_excluded_from_every_group():
    model = make_toy(6)
    spec = "qwen_vl_interface.model.model.visual,llm_layers_below:3"
    groups = layerwise_lr_decay_groups(model, base_lr=1e-4, decay=0.9, freeze_rules_spec=spec)
    frozen = resolve_frozen_param_ids(model, spec)
    in_groups = {id(p) for g in groups for p in g["params"]}
    assert frozen.isdisjoint(in_groups)
    assert sorted(_group_lr_by_layer(groups)) == [3, 4, 5]
    assert in_groups | frozen == {id(p) for p in model.parameters()}


def test_layer_multipliers_scale_individual_layers():
    model = make_toy(3)
    plain = _group_lr_by_layer(layerwise_lr_decay_groups(model, base_lr=1e-4, decay=1.0))
    scaled = _group_lr_by_layer(layerwise_lr_decay_groups(model, base_lr=1e-4, decay=1.0, layer_multipliers=[1.0, 0.5, 0.25]))
    assert plain == {0: pytest.approx(1e-4), 1: pytest.approx(1e-4), 2: pytest.approx(1e-4)}
    assert scaled[1] == pytest.approx(0.5e-4) and scaled[2] == pytest.approx(0.25e-4)


def test_invalid_arguments_raise():
    model = make_toy(2)
    with pytest.raises(ValueError):
        layerwise_lr_decay_groups(model, base_lr=0.0, decay=0.9)
    with pytest.raises(ValueError):
        layerwise_lr_decay_groups(model, base_lr=1e-4, decay=1.5)


def test_drift_driven_llrd_is_deterministic_hysteretic_and_bounded():
    model = make_toy(4)
    groups = layerwise_lr_decay_groups(model, base_lr=1e-4, decay=1.0)
    opt = torch.optim.SGD(groups, lr=1e-4)
    ctl = DriftDrivenLLRD(opt, layer_group_index(opt.param_groups), drift_high=0.1, drift_low=0.05, down_factor=0.5, up_factor=2.0, min_scale=0.1)
    ref = ctl.lrs()
    ctl.step([0.5, 0.5, 0.0, 0.07])           # layers 0,1 high drift -> halve; layer 2 low -> capped at 1; layer 3 in band -> hold
    lrs = ctl.lrs()
    assert lrs[0] == pytest.approx(ref[0] * 0.5) and lrs[1] == pytest.approx(ref[1] * 0.5)
    assert lrs[2] == pytest.approx(ref[2]) and lrs[3] == pytest.approx(ref[3])
    for _ in range(10):
        ctl.step([0.5, 0.0, 0.0, 0.0])        # layer 0 keeps dropping but never below min_scale; layer 1 recovers to 1
    lrs = ctl.lrs()
    assert lrs[0] == pytest.approx(ref[0] * 0.1)
    assert lrs[1] == pytest.approx(ref[1])
    # same inputs -> same trajectory
    opt2 = torch.optim.SGD(layerwise_lr_decay_groups(make_toy(4), base_lr=1e-4, decay=1.0), lr=1e-4)
    ctl2 = DriftDrivenLLRD(opt2, layer_group_index(opt2.param_groups), drift_high=0.1, drift_low=0.05, down_factor=0.5, up_factor=2.0, min_scale=0.1)
    ctl2.step([0.5, 0.5, 0.0, 0.07])
    for _ in range(10):
        ctl2.step([0.5, 0.0, 0.0, 0.0])
    assert ctl2.lrs() == pytest.approx(lrs)


def test_drift_driven_llrd_with_lr_scheduler_keeps_multiplier_across_scheduler_steps():
    model = make_toy(2)
    opt = torch.optim.SGD(layerwise_lr_decay_groups(model, base_lr=1e-4, decay=1.0), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 - 0.1 * s)
    ctl = DriftDrivenLLRD(opt, layer_group_index(opt.param_groups), lr_scheduler=sched, drift_high=0.1, drift_low=0.05, down_factor=0.5)
    ctl.step([0.9, 0.0])
    g0 = ctl.layer_group_index[0]
    assert sched.base_lrs[g0] == pytest.approx(0.5e-4)
    sched.step()
    assert opt.param_groups[g0]["lr"] == pytest.approx(0.5e-4 * 0.9)
