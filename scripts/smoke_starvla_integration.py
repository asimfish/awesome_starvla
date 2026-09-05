#!/usr/bin/env python3
"""CPU smoke test: real StarVLA action heads + ``Qwen_MultiHead`` + the full ``starvla_lab`` training wiring.

What it proves (no GPU, no model weights, ~20 s):

1. StarVLA imports under Python >= 3.10 and its three head factories build tiny OFT / GR00T / PI heads.
2. ``vlact_ext.Qwen_MultiHead`` accepts those *real* heads (not the unit-test mocks), runs
   ``forward`` (three losses summed, wrap-aware term on) and ``predict_action`` per head, and
   ``vlact_ext.flow_matching_loss`` reproduces the two StarVLA flow-matching heads' own ``forward``
   losses bit-for-bit under the same seed.
3. ``starvla_lab.train`` wires everything a real run would use: LLRD parameter groups with a
   ``llm_layers_below:1`` freeze rule, head dropout, drift probes written to JSONL, drift-driven LLRD and
   the auxiliary-data scheduler writing ``trainer.loss_scale.vlm`` back into the config.

The backbone is a random tiny stand-in that mirrors the *module tree* of Qwen3-VL under StarVLA
(``qwen_vl_interface.model.model.{visual, language_model.{embed_tokens, layers, norm}}``), so the freeze /
LLRD path syntax used in the YAML configs is exercised verbatim. Real Qwen3-VL-4B weights still require a
GPU; see ``code/vlact_ext/README.md`` §5.

Run from the repo root::

    bash scripts/setup_cpu_env.sh                      # once: Python 3.12 venv + StarVLA (editable) + deps
    PYTHONPATH=code:../starVLA_code .venv-starvla/bin/python scripts/smoke_starvla_integration.py

Exit code 2 = StarVLA not importable (wrong interpreter / PYTHONPATH); 1 = an assertion failed.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "code"
STARVLA_ROOT = REPO.parent / "starVLA_code"
for p in (str(CODE), str(STARVLA_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

HIDDEN, NUM_LAYERS, HORIZON, ACTION_DIM, DIT_WIDTH = 32, 3, 4, 7, 64
ACTION_QUERY = "\U0001F50D"  # QwenOFT's single-token action query (magnifier emoji)


# ──────────────────────────────────────────────────────────────────────
#  1. StarVLA imports and real head constructors
# ──────────────────────────────────────────────────────────────────────
def check_starvla_imports() -> None:
    import starVLA  # noqa: F401
    from omegaconf import OmegaConf  # noqa: F401

    from starVLA.model.framework.base_framework import build_framework  # noqa: F401
    from starVLA.model.framework.share_tools import populate_layerwise_dit_cfg  # noqa: F401
    from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model  # noqa: F401
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model  # noqa: F401,F811
    from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model  # noqa: F401,F811

    print("[ok] StarVLA core imports")


def _head_cfg(**action_model_overrides):
    """A ``framework.action_model`` view the StarVLA head factories read, shrunk to CPU size."""
    from omegaconf import OmegaConf

    base = {
        "action_dim": ACTION_DIM,
        "state_dim": 0,
        "action_horizon": HORIZON,
        "action_hidden_dim": HIDDEN,
        "add_pos_embed": True,
        "max_seq_len": 256,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 32,
        "num_inference_timesteps": 2,
        "num_target_vision_tokens": 4,
        "repeated_diffusion_steps": 1,
    }
    base.update(action_model_overrides)
    return OmegaConf.create({"framework": {"action_model": base}})


def build_real_heads():
    from starVLA.model.framework.share_tools import populate_layerwise_dit_cfg
    from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model as build_gr00t
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model as build_pi
    from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model as build_oft

    oft = build_oft(config=_head_cfg(action_model_type="MLP"))

    gr00t = build_gr00t(
        config=_head_cfg(
            action_model_type="DiT-B",
            hidden_size=DIT_WIDTH,
            diffusion_model_cfg={
                "cross_attention_dim": HIDDEN,
                "dropout": 0.0,
                "final_dropout": False,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 2,
                "output_dim": DIT_WIDTH,
                "positional_embeddings": None,
            },
        )
    )

    pi_cfg = _head_cfg(
        action_model_type="LayerwiseFM",
        diffusion_model_cfg={
            "action_dit_hidden_dim": DIT_WIDTH,
            "dropout": 0.0,
            "final_dropout": False,
            "interleave_self_attention": False,
            "norm_type": "ada_norm",
            "positional_embeddings": None,
            "attention_head_dim": 32,
        },
    )
    populate_layerwise_dit_cfg(pi_cfg, dit_hidden_dim=DIT_WIDTH, num_dit_layers=NUM_LAYERS)
    pi = build_pi(config=pi_cfg)
    project_layers = nn.ModuleList(
        nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, DIT_WIDTH)) for _ in range(NUM_LAYERS)
    )
    print(
        "[ok] real StarVLA heads: "
        f"oft={type(oft).__name__}, gr00t={type(gr00t).__name__}, pi={type(pi).__name__}"
    )
    return {"oft": oft, "gr00t": gr00t, "pi": pi}, project_layers


# ──────────────────────────────────────────────────────────────────────
#  2. Tiny backbone with the Qwen3-VL module tree StarVLA exposes
# ──────────────────────────────────────────────────────────────────────
class _Tokenizer:
    ACTION_ID = 2

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": [self.ACTION_ID if c == ACTION_QUERY else 3 + ord(c) % 60 for c in text]}


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, HIDDEN)
        self.layers = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(NUM_LAYERS))
        self.norm = nn.LayerNorm(HIDDEN)


class _InnerModel(nn.Module):
    """Stands for ``Qwen3VLModel``: ``.visual`` + ``.language_model``."""

    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(HIDDEN, HIDDEN)
        self.language_model = _LanguageModel()


class _ForConditionalGeneration(nn.Module):
    """Stands for ``Qwen3VLForConditionalGeneration``: ``.model`` + ``.lm_head`` + ``.config``."""

    def __init__(self):
        super().__init__()
        self.model = _InnerModel()
        self.lm_head = nn.Linear(HIDDEN, 128, bias=False)
        self.config = SimpleNamespace(
            hidden_size=HIDDEN, text_config=SimpleNamespace(hidden_size=HIDDEN, num_hidden_layers=NUM_LAYERS)
        )

    @property
    def device(self):
        return self.lm_head.weight.device

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True, **_):
        lm = self.model.language_model
        h = lm.embed_tokens(input_ids)
        hidden_states = [h]
        for layer in lm.layers:
            h = torch.tanh(layer(h))
            hidden_states.append(h)
        hidden_states[-1] = lm.norm(hidden_states[-1])
        return SimpleNamespace(hidden_states=tuple(hidden_states), logits=self.lm_head(hidden_states[-1]))


class TinyQwenVLInterface(nn.Module):
    """Mirrors StarVLA's ``_QWen3_VL_Interface`` surface: ``.model``, ``.processor.tokenizer``, ``build_qwenvl_inputs``."""

    def __init__(self):
        super().__init__()
        self.model = _ForConditionalGeneration()
        self.processor = SimpleNamespace(tokenizer=_Tokenizer())

    def build_qwenvl_inputs(self, images, instructions, **_):
        rows = [self.processor.tokenizer(text)["input_ids"] for text in instructions]
        L = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), L, dtype=torch.long)
        mask = torch.zeros(len(rows), L, dtype=torch.long)
        for i, row in enumerate(rows):  # left padding like StarVLA (padding_side = "left")
            ids[i, L - len(row):] = torch.tensor(row)
            mask[i, L - len(row):] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def forward(self, **kwargs):
        return self.model(**kwargs)


def framework_config():
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "framework": {
                "name": "QwenMultiHead",
                "qwenvl": {"base_vlm": "tiny-random", "vl_hidden_dim": HIDDEN, "num_vl_layers": NUM_LAYERS},
                "action_model": {"action_dim": ACTION_DIM, "state_dim": 0, "action_horizon": HORIZON, "repeated_diffusion_steps": 1},
                "heads": {
                    "oft": {"enabled": True, "loss_weight": 1.0},
                    "gr00t": {"enabled": True, "loss_weight": 0.5},
                    "pi": {"enabled": True, "loss_weight": 0.5},
                },
                "predict_head": "oft",
                "mask_oft_queries_for_fm_heads": True,
                "wrap_aware": {"enabled": True, "period": 2.0, "fm_sample_loss_weight": 0.5},
                "unified_layout": {"enabled": False, "unified_dim": ACTION_DIM, "layouts": None},
            },
            "datasets": {"vla_data": {}},
            "trainer": {},
        }
    )


def make_batch(n: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    batch = []
    for i in range(n):
        img = Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8))
        batch.append(
            {
                "action": rng.uniform(-1, 1, size=(HORIZON, ACTION_DIM)).astype(np.float32),
                "image": [img],
                "lang": f"pick up object {i}",
                "state": rng.uniform(-1, 1, size=(1, ACTION_DIM)).astype(np.float32),
            }
        )
    return batch


# ──────────────────────────────────────────────────────────────────────
#  3. Framework forward / predict with the real heads
# ──────────────────────────────────────────────────────────────────────
def check_framework(model) -> None:
    from vlact_ext.multihead_framework import FRAMEWORK_REGISTRY, Qwen_MultiHead

    assert FRAMEWORK_REGISTRY["QwenMultiHead"] is Qwen_MultiHead
    batch = make_batch()
    out = model(batch)
    for key in ("action_loss", "loss_oft", "loss_gr00t", "loss_pi"):
        assert key in out and torch.isfinite(out[key]).all(), f"{key} missing or non-finite: {out.get(key)}"
    expected = out["loss_oft"] + 0.5 * out["loss_gr00t"] + 0.5 * out["loss_pi"]
    assert torch.allclose(out["action_loss"], expected, atol=1e-5), "action_loss != sum_h w_h * L_h"
    out["action_loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients reached any parameter"

    model.eval()
    shapes = {}
    for head in ("oft", "gr00t", "pi"):
        pred = model.predict_action(examples=[batch[0]], head=head)
        shapes[head] = tuple(pred["normalized_actions"].shape)
        assert shapes[head] == (1, HORIZON, ACTION_DIM), f"{head}: {shapes[head]}"
    model.train()
    print(
        f"[ok] forward: action_loss={out['action_loss'].item():.4f} "
        f"(oft {out['loss_oft'].item():.3f}, gr00t {out['loss_gr00t'].item():.3f}, pi {out['loss_pi'].item():.3f}); "
        f"predict_action per head -> {shapes['oft']}"
    )


def check_flow_matching_equivalence(model) -> None:
    """``vlact_ext.flow_matching_loss`` must reproduce StarVLA's own head ``forward`` bit-for-bit.

    Both draw ``noise`` then ``t`` from the global generator in the same order, so re-seeding before each
    call makes them comparable. Masks off and ``wrap_weight=0`` reduce ours to the plain velocity MSE.
    """
    from vlact_ext.multihead_framework import flow_matching_loss

    model.eval()  # dropout is already 0 in these heads; eval makes the comparison independent of that
    B = 2
    target = torch.randn(B, HORIZON, ACTION_DIM)
    last_hidden = torch.randn(B, 11, HIDDEN)
    mask = torch.ones(B, 11, dtype=torch.bool)
    mask[1, :3] = False
    pi_cond = [proj(last_hidden) for proj in model.project_layers]

    for name, cond, layerwise in (("gr00t", last_hidden, False), ("pi", pi_cond, True)):
        head = model.heads[name]
        torch.manual_seed(1234)
        reference = head(cond, target, None, encoder_attention_mask=mask)
        torch.manual_seed(1234)
        ours = flow_matching_loss(head, cond, target, layerwise=layerwise, encoder_attention_mask=mask)
        assert torch.allclose(ours["velocity_loss"], reference.float(), atol=1e-6), (
            f"{name}: flow_matching_loss={ours['velocity_loss'].item():.6f} vs StarVLA head forward={reference.item():.6f}"
        )
        assert float(ours["wrap_loss"]) == 0.0
    model.train()
    print("[ok] flow_matching_loss == StarVLA FlowmatchingActionHead / LayerwiseFlowmatchingActionHead forward (same seed, atol 1e-6)")


# ──────────────────────────────────────────────────────────────────────
#  4. starvla_lab training wiring on a mock trainer
# ──────────────────────────────────────────────────────────────────────
def check_lab_wiring(model, workdir: Path) -> None:
    from starvla_lab.probes import read_jsonl
    from starvla_lab.train import LabConfig, LabHooks, attach_to_trainer, build_optimizer_and_scheduler
    from vlact_ext.freeze_rules import freeze_by_rules

    steps = 8
    cfg = {
        "seed": 0,
        "trainer": {
            "max_train_steps": steps,
            "weight_decay": 0.0,
            "freeze_modules": "qwen_vl_interface.model.model.visual,llm_layers_below:1",
            "learning_rate": {"base": 1e-3, "qwen_vl_interface": 1e-3, "action_model": 1e-2},
            "loss_scale": {"vla": 1.0, "vlm": 0.5},
            "lab": {
                "llrd": {"enabled": True, "decay": 0.7, "drift_driven": True, "drift_high": 0.05, "drift_low": 0.01, "down_factor": 0.5},
                "aux_scheduler": {"enabled": True, "strategy": "linear", "ratio_min": 0.1, "ratio_max": 0.5, "loss_scale_min": 0.2, "loss_scale_max": 1.0},
                "probes": {"enabled": True, "every_n_steps": 2, "jsonl_path": str(workdir / "probes.jsonl")},
                "head_dropout": {"enabled": True, "p_all": 0.0, "seed": 0},
            },
        },
    }
    lab = LabConfig.from_cfg(cfg)
    lab.validate()

    # Same freeze spec as the optimizer: StarVLA's freeze_backbones equivalent (grads off), LLRD excludes them.
    report = freeze_by_rules(model, cfg["trainer"]["freeze_modules"])
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, lab, lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    )
    layer_lrs = {g["layer_index"]: g["lr"] for g in optimizer.param_groups if "layer_index" in g}
    assert sorted(layer_lrs) == [1, 2], f"layer 0 must be frozen out of the optimizer, got {sorted(layer_lrs)}"
    assert layer_lrs[1] < layer_lrs[2], "LLRD must give deeper layers a larger lr"
    visual_ids = {id(p) for p in model.qwen_vl_interface.model.model.visual.parameters()}
    assert visual_ids.isdisjoint({id(p) for g in optimizer.param_groups for p in g["params"]}), "visual is frozen"

    class Trainer:
        def __init__(self):
            self.config, self.model, self.optimizer, self.lr_scheduler = cfg, model, optimizer, scheduler
            self.completed_steps = 0

        def _train_step(self, batch_vla, batch_vlm):
            self.optimizer.zero_grad()
            out = self.model(batch_vla)
            out["action_loss"].backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.completed_steps += 1
            return {"action_dit_loss": float(out["action_loss"].detach())}

    def extract_fn(m, batch):  # simplified mean-pool view of starvla_lab.probes.QwenBackboneProbe on the tiny VLM
        inputs = m.qwen_vl_interface.build_qwenvl_inputs([ex["image"] for ex in batch], m._prepare_instructions(batch))
        with torch.no_grad():
            out = m.qwen_vl_interface.model(**inputs, output_hidden_states=True)
        mask = inputs["attention_mask"].unsqueeze(-1).to(out.hidden_states[0].dtype)
        return [((h * mask).sum(1) / mask.sum(1).clamp(min=1)).float() for h in out.hidden_states[1:]]

    trainer = Trainer()
    hooks = LabHooks(trainer, lab, extract_fn=extract_fn, probe_batch=make_batch(4, seed=1))
    attach_to_trainer(trainer, hooks)

    seen_heads = set()
    metrics = None
    batch = make_batch(2, seed=2)
    for _ in range(steps):
        metrics = trainer._train_step(batch, None)
        seen_heads.add(metrics["lab/active_heads"])
    assert seen_heads <= {"oft", "gr00t", "pi"} and len(seen_heads) >= 2, f"head dropout did not rotate heads: {seen_heads}"
    assert abs(cfg["trainer"]["loss_scale"]["vlm"] - metrics["lab/vlm_loss_scale"]) < 1e-9, "aux scheduler must write loss_scale.vlm back"
    records = read_jsonl(workdir / "probes.jsonl")
    # step = completed updates: the initial reference-vs-itself record (step 0, drift 0) plus one every 2 updates
    expected_steps = [0] + list(range(2, steps + 1, 2))
    assert [r["step"] for r in records] == expected_steps, f"expected probe steps {expected_steps}, got {[r['step'] for r in records]}"
    assert records[0]["drift"]["mean"] == 0.0, "reference vs. itself must have zero drift"
    assert all("drift" in r for r in records) and hooks.last_drift is not None and hooks.last_drift >= 0
    assert "llrd_multipliers" in records[-1], "drift-driven LLRD did not act on the probe"

    print(
        f"[ok] lab wiring: frozen {report.num_frozen} params (visual + layer 0); LLRD lrs "
        f"{ {k: round(v, 6) for k, v in sorted(layer_lrs.items())} }; head dropout saw {sorted(seen_heads)}; "
        f"{len(records)} probe records, last drift={hooks.last_drift:.4f}; loss_scale.vlm -> {metrics['lab/vlm_loss_scale']:.3f}"
    )


def main() -> int:
    torch.manual_seed(0)
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  starVLA_code present: {STARVLA_ROOT.exists()}")
    try:
        check_starvla_imports()
    except Exception as exc:  # wrong interpreter / PYTHONPATH -> tell the reader instead of a traceback
        print(f"[fail] StarVLA is not importable ({type(exc).__name__}: {exc}).", file=sys.stderr)
        print("       Use the Python >= 3.10 venv from scripts/setup_cpu_env.sh and PYTHONPATH=code:../starVLA_code", file=sys.stderr)
        return 2

    from vlact_ext.multihead_framework import Qwen_MultiHead

    heads, project_layers = build_real_heads()
    model = Qwen_MultiHead(framework_config(), vlm=TinyQwenVLInterface(), heads=heads, project_layers=project_layers)
    model.train()
    check_framework(model)
    check_flow_matching_equivalence(model)
    with tempfile.TemporaryDirectory() as tmp:
        check_lab_wiring(model, Path(tmp))
    print("[done] smoke_starvla_integration passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
