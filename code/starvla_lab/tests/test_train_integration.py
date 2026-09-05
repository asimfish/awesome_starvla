import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starvla_lab.train import LabConfig, LabHooks, attach_to_trainer, build_optimizer_and_scheduler, cfg_get, cfg_set
from starvla_lab.probes import read_jsonl


class _Holder(nn.Module):
    def __init__(self, **children):
        super().__init__()
        for k, v in children.items():
            setattr(self, k, v)


class _LM(nn.Module):
    def __init__(self, dim=4, n=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n)])
        self.norm = nn.LayerNorm(dim)


class _Framework(nn.Module):
    """Minimal stand-in for QwenMultiHead: StarVLA-like backbone paths, `heads`, `active_heads`."""

    def __init__(self, dim=4, n=4):
        super().__init__()
        inner = _Holder(visual=nn.Linear(dim, dim), language_model=_LM(dim, n))
        self.qwen_vl_interface = _Holder(model=_Holder(model=inner, lm_head=nn.Linear(dim, 10)))
        self.heads = nn.ModuleDict({"oft": nn.Linear(dim, 7), "pi": nn.Linear(dim, 7), "gr00t": nn.Linear(dim, 7)})
        self.active_heads = None

    def layer_reps(self, x):
        h = x
        reps = []
        for layer in self.qwen_vl_interface.model.model.language_model.layers:
            h = torch.tanh(layer(h))
            reps.append(h)
        return reps


def _extract(model, batch):
    return model.layer_reps(batch)


class _Trainer:
    def __init__(self, cfg, model, optimizer, lr_scheduler=None):
        self.config, self.model, self.optimizer, self.lr_scheduler = cfg, model, optimizer, lr_scheduler
        self.completed_steps = 0
        self.calls = []

    def _train_step(self, batch_vla, batch_vlm):
        self.calls.append((batch_vla, batch_vlm))
        with torch.no_grad():  # simulate drift: perturb layer 2 more each step
            self.model.qwen_vl_interface.model.model.language_model.layers[2].weight.add_(0.5)
        self.completed_steps += 1
        return {"action_dit_loss": 1.0}


def _cfg(**lab):
    return {
        "trainer": {
            "max_train_steps": 100, "weight_decay": 0.0, "freeze_modules": "",
            "learning_rate": {"base": 1e-5, "qwen_vl_interface": 1e-5, "action_model": 1e-4},
            "loss_scale": {"vla": 1.0, "vlm": 1.0},
            "lab": lab,
        }
    }


def test_lab_config_reads_nested_dict_and_validates():
    lab = LabConfig.from_cfg(_cfg(llrd={"enabled": True, "decay": 0.8}, probes={"enabled": True, "every_n_steps": 5, "layers": [1, 2]}))
    assert lab.llrd.enabled and lab.llrd.decay == 0.8 and lab.probes.layers == [1, 2]
    assert LabConfig.from_cfg({"trainer": {}}).any_enabled() is False
    bad = LabConfig.from_cfg(_cfg(llrd={"enabled": True, "drift_driven": True}))
    with pytest.raises(ValueError):
        bad.validate()
    ns = SimpleNamespace(trainer=SimpleNamespace(lab=SimpleNamespace(head_dropout=SimpleNamespace(enabled=True, p_all=0.2))))
    assert LabConfig.from_cfg(ns).head_dropout.p_all == 0.2
    d = {}
    cfg_set(d, "trainer.loss_scale.vlm", 0.5)
    assert cfg_get(d, "trainer.loss_scale.vlm") == 0.5


def test_build_optimizer_falls_back_when_llrd_disabled():
    model = _Framework()
    sentinel = object()
    out = build_optimizer_and_scheduler(model, _cfg(), LabConfig.from_cfg(_cfg()), lambda opt: None, fallback=lambda m, c: sentinel)
    assert out is sentinel


def test_build_optimizer_llrd_groups_freeze_and_scheduler():
    model = _Framework(n=4)
    cfg = _cfg(llrd={"enabled": True, "decay": 0.5})
    cfg["trainer"]["freeze_modules"] = "qwen_vl_interface.model.model.visual,llm_layers_below:1"
    made = {}

    def factory(opt):
        made["opt"] = opt
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    opt, sched = build_optimizer_and_scheduler(model, cfg, LabConfig.from_cfg(cfg), factory)
    assert made["opt"] is opt and isinstance(opt, torch.optim.AdamW)
    layer_lrs = {g["layer_index"]: g["lr"] for g in opt.param_groups if "layer_index" in g}
    assert sorted(layer_lrs) == [1, 2, 3] and layer_lrs[3] == pytest.approx(1e-5 * 0.5)
    frozen = {id(p) for p in model.qwen_vl_interface.model.model.visual.parameters()}
    assert frozen.isdisjoint({id(p) for g in opt.param_groups for p in g["params"]})
    head_ids = {id(p) for p in model.heads.parameters()}
    assert all(g["lr"] == pytest.approx(1e-4) for g in opt.param_groups if any(id(p) in head_ids for p in g["params"]))


def test_build_optimizer_reads_starvla_optimizer_block():
    """StarVLA yamls carry trainer.optimizer.{betas, weight_decay, eps}; the legacy flat trainer.weight_decay must lose."""
    model = _Framework(n=2)
    cfg = _cfg(llrd={"enabled": True, "decay": 0.9})
    cfg["trainer"]["optimizer"] = {"name": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 1e-3}
    opt, _ = build_optimizer_and_scheduler(model, cfg, LabConfig.from_cfg(cfg), lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1.0))
    assert opt.defaults["weight_decay"] == pytest.approx(1e-3) and opt.defaults["betas"] == (0.9, 0.95) and opt.defaults["eps"] == pytest.approx(1e-8)


def test_hooks_aux_scheduler_and_head_dropout_apply_each_step():
    model = _Framework()
    cfg = _cfg(aux_scheduler={"enabled": True, "strategy": "linear", "ratio_min": 0.1, "ratio_max": 0.5, "loss_scale_min": 0.2, "loss_scale_max": 1.0},
               head_dropout={"enabled": True, "p_all": 0.0, "seed": 3})
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = attach_to_trainer(_Trainer(cfg, model, opt), LabHooks(_Trainer(cfg, model, opt), LabConfig.from_cfg(cfg)))
    m0 = trainer._train_step("vla", "vlm")
    assert cfg["trainer"]["loss_scale"]["vlm"] == pytest.approx(1.0) and m0["lab/vlm_loss_scale"] == pytest.approx(1.0)
    assert len(model.active_heads) == 1 and m0["lab/active_heads"] in {"oft", "pi", "gr00t"}
    for _ in range(49):
        trainer._train_step("vla", "vlm")
    m50 = trainer._train_step("vla", "vlm")  # step index 50 of 100 -> u = 0.5
    assert m50["lab/vlm_loss_scale"] == pytest.approx(0.6) and cfg["trainer"]["vlm_sample_prob"] == pytest.approx(0.3)


def test_hooks_probes_drive_llrd_and_write_jsonl(tmp_path: Path):
    torch.manual_seed(0)
    model = _Framework(n=4)
    cfg = _cfg(llrd={"enabled": True, "decay": 1.0, "drift_driven": True, "drift_high": 0.01, "drift_low": 0.001, "down_factor": 0.5, "min_scale": 0.1},
               probes={"enabled": True, "every_n_steps": 2, "jsonl_path": str(tmp_path / "p.jsonl"), "layers": None})
    opt, sched = build_optimizer_and_scheduler(model, cfg, LabConfig.from_cfg(cfg), lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1.0))
    trainer = _Trainer(cfg, model, opt, sched)
    batch = torch.randn(32, 4)
    hooks = LabHooks(trainer, LabConfig.from_cfg(cfg), extract_fn=_extract, probe_batch=batch)
    attach_to_trainer(trainer, hooks)
    ref_lrs = dict(hooks.llrd.lrs())
    for _ in range(6):
        trainer._train_step("vla", "vlm")
    lrs = hooks.llrd.lrs()
    assert lrs[2] < ref_lrs[2] and lrs[3] < ref_lrs[3]          # perturbed layer and downstream drift -> lr reduced
    assert lrs[0] == pytest.approx(ref_lrs[0]) and lrs[1] == pytest.approx(ref_lrs[1])
    records = read_jsonl(tmp_path / "p.jsonl")
    assert len(records) == 3 and records[0]["step"] == 0 and "drift" in records[-1]
    assert hooks.last_drift is not None and hooks.last_drift > 0


def test_calibrate_only_records_but_does_not_act(tmp_path: Path):
    model = _Framework(n=3)
    cfg = _cfg(llrd={"enabled": True, "decay": 1.0, "drift_driven": True, "drift_high": 0.01, "drift_low": 0.001},
               probes={"enabled": True, "every_n_steps": 1, "jsonl_path": str(tmp_path / "c.jsonl"), "calibrate_only": True})
    opt, sched = build_optimizer_and_scheduler(model, cfg, LabConfig.from_cfg(cfg), lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1.0))
    trainer = _Trainer(cfg, model, opt, sched)
    hooks = LabHooks(trainer, LabConfig.from_cfg(cfg), extract_fn=_extract, probe_batch=torch.randn(16, 4))
    attach_to_trainer(trainer, hooks)
    before = dict(hooks.llrd.lrs())
    for _ in range(4):
        trainer._train_step("vla", "vlm")
    assert hooks.llrd.lrs() == pytest.approx(before)
    assert len(read_jsonl(tmp_path / "c.jsonl")) == 4 and hooks.last_drift > 0


def test_per_head_losses_from_forward_reach_the_metrics():
    class _MultiLoss(_Framework):
        def forward(self, x):
            return {"action_loss": torch.tensor(1.5), "loss_oft": torch.tensor(1.0), "loss_pi": torch.tensor(0.5), "aux": "ignored"}

    model = _MultiLoss()
    cfg = _cfg(head_dropout={"enabled": True, "p_all": 1.0})
    trainer = _Trainer(cfg, model, torch.optim.SGD(model.parameters(), lr=1e-3))
    original = trainer._train_step

    def step_with_forward(batch_vla, batch_vlm):
        model.forward(batch_vla)  # StarVLA calls .forward directly (no nn.Module hooks)
        return original(batch_vla, batch_vlm)

    trainer._train_step = step_with_forward
    attach_to_trainer(trainer, LabHooks(trainer, LabConfig.from_cfg(cfg)))
    m = trainer._train_step(torch.zeros(1, 4), None)
    assert m["lab/loss_oft"] == pytest.approx(1.0) and m["lab/loss_pi"] == pytest.approx(0.5) and "lab/action_loss" not in m


def test_probes_enabled_requires_extract_fn():
    model = _Framework()
    cfg = _cfg(probes={"enabled": True, "every_n_steps": 10})
    trainer = _Trainer(cfg, model, torch.optim.SGD(model.parameters(), lr=1e-3))
    with pytest.raises(ValueError):
        LabHooks(trainer, LabConfig.from_cfg(cfg))


def test_select_mode_auto_single_cotrain():
    from starvla_lab.train.train_starvla_lab import select_mode

    assert select_mode({"trainer": {}, "datasets": {"vla_data": {}}}) == "single"
    assert select_mode({"trainer": {}, "datasets": {"vla_data": {}, "vlm_data": {"dataset_use": "x"}}}) == "cotrain"
    assert select_mode({"trainer": {"lab": {"mode": "single"}}, "datasets": {"vlm_data": {}}}) == "single"
    with pytest.raises(ValueError):
        select_mode({"trainer": {"lab": {"mode": "bogus"}}})
