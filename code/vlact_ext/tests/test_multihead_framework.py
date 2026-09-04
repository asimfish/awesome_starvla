"""CPU tests for Qwen_MultiHead with an injected mock backbone and tiny heads.

The mock VLM / heads only reproduce the *interfaces* the framework relies on:
  VLM   : .model.config.{hidden_size,text_config.num_hidden_layers}, .processor.tokenizer(...),
          .build_qwenvl_inputs(images, instructions) -> {"input_ids", "attention_mask"},
          __call__(**inputs, output_hidden_states=True, ...) -> .hidden_states (num_layers + 1 tensors)
  OFT   : .predict_action(queries[B, T, H]) -> [B, T, D]
  FM    : .sample_time, .num_timestep_buckets, .action_encoder, .future_tokens, .position_embedding,
          .model(hidden_states, encoder_hidden_states, timestep, encoder_attention_mask, return_pre_output),
          .action_decoder, .config.add_pos_embed, .predict_action(cond, state, encoder_attention_mask)
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta

from vlact_ext import multihead_framework as mh
from vlact_ext.multihead_framework import ACTION_QUERY_TOKEN, FRAMEWORK_REGISTRY, Qwen_MultiHead, flow_matching_loss

HIDDEN = 16
NUM_LAYERS = 2
PI_WIDTH = 8
HORIZON = 4
ACTION_DIM = 20
PAD_ID, ACTION_ID = 0, 1


class _NS(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def make_config(heads=("oft", "gr00t", "pi"), weights=None, wrap=True, layout=False, predict_head="oft", mask_queries=False, repeats=2):
    weights = weights or {}
    heads_cfg = {
        name: _NS(enabled=name in heads, loss_weight=weights.get(name, 1.0), action_model=_NS(repeated_diffusion_steps=repeats))
        for name in ("oft", "gr00t", "pi")
    }
    framework = _NS(
        name="QwenMultiHead",
        qwenvl=_NS(base_vlm="mock", vl_hidden_dim=HIDDEN, num_vl_layers=NUM_LAYERS),
        action_model=_NS(action_dim=ACTION_DIM, state_dim=0, action_horizon=HORIZON, repeated_diffusion_steps=1),
        heads=_NS(**heads_cfg),
        predict_head=predict_head,
        mask_oft_queries_for_fm_heads=mask_queries,
        wrap_aware=_NS(enabled=wrap, period=2.0, fm_sample_loss_weight=1.0),
        unified_layout=_NS(enabled=layout, unified_dim=20, layouts=None),
    )
    return _NS(framework=framework, datasets=_NS(vla_data=_NS()), trainer=_NS())


class MockTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ACTION_ID if c == ACTION_QUERY_TOKEN else 2 + (ord(c) % 60) for c in text]}


class MockVLM(nn.Module):
    def __init__(self, hidden=HIDDEN, num_layers=NUM_LAYERS, vocab=64):
        super().__init__()
        self.model = nn.Module()
        self.model.config = SimpleNamespace(hidden_size=hidden, text_config=SimpleNamespace(num_hidden_layers=num_layers))
        self.processor = SimpleNamespace(tokenizer=MockTokenizer())
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
        self.calls = 0

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        seqs = [self.processor.tokenizer(s)["input_ids"] for s in instructions]
        length = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), length), PAD_ID, dtype=torch.long)
        attn = torch.zeros((len(seqs), length), dtype=torch.long)
        for i, s in enumerate(seqs):  # left padding, as StarVLA's processors are configured
            ids[i, length - len(s) :] = torch.tensor(s)
            attn[i, length - len(s) :] = 1
        return {"input_ids": ids, "attention_mask": attn}

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.calls += 1
        h = self.embed(input_ids)
        hidden_states = [h]
        for layer in self.layers:
            h = torch.tanh(layer(h))
            hidden_states.append(h)
        return SimpleNamespace(hidden_states=tuple(hidden_states))


class TinyOFTHead(nn.Module):
    def __init__(self, hidden, action_dim):
        super().__init__()
        self.proj = nn.Linear(hidden, action_dim)
        self.calls = []

    def predict_action(self, queries):
        self.calls.append(tuple(queries.shape))
        return self.proj(queries)


def _masked_mean_pool(cond, mask):
    if mask is None:
        return cond.mean(dim=1)
    m = mask.to(cond.dtype)[..., None]
    return (cond * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class TinyDiT(nn.Module):
    def __init__(self, width, cond_dim, num_layers):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(num_layers)])
        self.cond_proj = nn.Linear(cond_dim, width)
        self.calls = []

    def forward(self, hidden_states, encoder_hidden_states, timestep=None, encoder_attention_mask=None, return_pre_output=False, return_all_hidden_states=False):
        layerwise = isinstance(encoder_hidden_states, (list, tuple))
        self.calls.append({"layerwise": layerwise, "return_pre_output": return_pre_output, "mask": encoder_attention_mask, "batch": hidden_states.shape[0]})
        conds = list(encoder_hidden_states) if layerwise else [encoder_hidden_states] * len(self.transformer_blocks)
        h = hidden_states
        for block, cond in zip(self.transformer_blocks, conds):
            pooled = _masked_mean_pool(cond, encoder_attention_mask)
            h = torch.tanh(block(h) + self.cond_proj(pooled)[:, None, :])
        return h


class TinyActionEncoder(nn.Module):
    def __init__(self, action_dim, width, buckets):
        super().__init__()
        self.lin = nn.Linear(action_dim, width)
        self.temb = nn.Embedding(buckets + 1, width)

    def forward(self, actions, timesteps):
        return self.lin(actions) + self.temb(timesteps.clamp(min=0))[:, None, :]


class TinyFMHead(nn.Module):
    def __init__(self, cond_dim, action_dim=ACTION_DIM, horizon=HORIZON, width=PI_WIDTH, num_layers=NUM_LAYERS, num_inference_timesteps=2):
        super().__init__()
        self.action_dim, self.action_horizon = action_dim, horizon
        self.num_timestep_buckets = 1000
        self.num_inference_timesteps = num_inference_timesteps
        self.config = SimpleNamespace(add_pos_embed=True, noise_s=0.999)
        self.action_encoder = TinyActionEncoder(action_dim, width, self.num_timestep_buckets)
        self.future_tokens = nn.Embedding(2, width)
        self.position_embedding = nn.Embedding(64, width)
        self.model = TinyDiT(width, cond_dim, num_layers)
        self.action_decoder = nn.Linear(width, action_dim)
        self.beta_dist = Beta(1.5, 1.0)
        self.predict_calls = 0

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    def predict_action(self, cond, state=None, encoder_attention_mask=None):
        self.predict_calls += 1
        first = cond[0] if isinstance(cond, (list, tuple)) else cond
        B, device = first.shape[0], first.device
        actions = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        dt = 1.0 / self.num_inference_timesteps
        for k in range(self.num_inference_timesteps):
            t = torch.full((B,), int(k / self.num_inference_timesteps * self.num_timestep_buckets), dtype=torch.long, device=device)
            feats = self.action_encoder(actions, t) + self.position_embedding(torch.arange(self.action_horizon, device=device))[None]
            sa = torch.cat([self.future_tokens.weight[None].expand(B, -1, -1), feats], dim=1)
            velocity = self.action_decoder(self.model(sa, cond, t, encoder_attention_mask))[:, -self.action_horizon :]
            actions = actions + dt * velocity
        return actions


def build_model(**cfg_kwargs):
    names = cfg_kwargs.get("heads", ("oft", "gr00t", "pi"))
    cfg = make_config(**cfg_kwargs)
    heads, project_layers = {}, None
    if "oft" in names:
        heads["oft"] = TinyOFTHead(HIDDEN, ACTION_DIM)
    if "gr00t" in names:
        heads["gr00t"] = TinyFMHead(cond_dim=HIDDEN)
    if "pi" in names:
        heads["pi"] = TinyFMHead(cond_dim=PI_WIDTH)
        project_layers = nn.ModuleList([nn.Linear(HIDDEN, PI_WIDTH) for _ in range(NUM_LAYERS)])
    return Qwen_MultiHead(cfg, vlm=MockVLM(), heads=heads, project_layers=project_layers)


def make_examples(batch=2, chunk=HORIZON + 2, dim=ACTION_DIM, seed=0, with_mask=False, robot_tag="franka", state=False):
    rng = np.random.default_rng(seed)
    examples = []
    for i in range(batch):
        ex = {
            "action": rng.uniform(-1, 1, size=(chunk, dim)).astype(np.float16),
            "image": ["img"],
            "lang": f"pick up object number {i}",
            "robot_tag": robot_tag,
        }
        if with_mask:
            mask = np.ones(dim, dtype=bool)
            mask[12:18] = False
            ex["action_mask"] = np.broadcast_to(mask, (chunk, dim)).copy()
            periodic = np.zeros(dim, dtype=bool)
            periodic[:12] = True
            ex["periodic_mask"] = np.broadcast_to(periodic, (chunk, dim)).copy()
        if state:
            ex["state"] = rng.uniform(-1, 1, size=(1, 7)).astype(np.float16)
        examples.append(ex)
    return examples


class TestConstruction(unittest.TestCase):
    def test_registered_under_expected_key(self):
        self.assertIs(FRAMEWORK_REGISTRY["QwenMultiHead"], Qwen_MultiHead)

    def test_requires_injection_without_starvla(self):
        if mh._STARVLA_AVAILABLE:
            self.skipTest("StarVLA importable; default constructors are exercised on GPU instead")
        with self.assertRaises(ImportError):
            Qwen_MultiHead(make_config())

    def test_config_plumbing(self):
        model = build_model(weights={"oft": 1.0, "gr00t": 0.5, "pi": 2.0}, repeats=3)
        self.assertEqual(model.head_weights, {"oft": 1.0, "gr00t": 0.5, "pi": 2.0})
        self.assertEqual(model.repeated_diffusion_steps, {"gr00t": 3, "pi": 3})
        self.assertEqual(model.action_horizon, HORIZON)
        self.assertEqual(model.action_token_id, ACTION_ID)
        self.assertEqual(model.predict_head, "oft")
        self.assertEqual(list(model.heads), ["oft", "gr00t", "pi"])
        self.assertEqual(len(model.project_layers), NUM_LAYERS)
        self.assertEqual(mh._deep_merge({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"b": 9}}), {"a": {"b": 9, "c": 2}, "d": 1})

    def test_predict_head_falls_back_when_disabled(self):
        with self.assertWarns(UserWarning):
            model = build_model(heads=("gr00t", "pi"), predict_head="oft")
        self.assertEqual(model.predict_head, "gr00t")
        self.assertIsNone(model.action_token_id)

    def test_rejects_unknown_head_and_bad_fm_head(self):
        cfg = make_config()
        with self.assertRaises(ValueError):
            Qwen_MultiHead(cfg, vlm=MockVLM(), heads={"fast": TinyOFTHead(HIDDEN, ACTION_DIM)})
        with self.assertRaises(TypeError):
            Qwen_MultiHead(cfg, vlm=MockVLM(), heads={"gr00t": TinyOFTHead(HIDDEN, ACTION_DIM)})


class TestForward(unittest.TestCase):
    def test_three_heads_share_one_backbone_forward_and_sum_weighted_losses(self):
        torch.manual_seed(0)
        model = build_model(weights={"oft": 1.0, "gr00t": 0.5, "pi": 2.0})
        out = model(make_examples())
        for key in ("action_loss", "loss_oft", "loss_pi", "loss_gr00t"):
            self.assertIn(key, out)
            self.assertEqual(out[key].ndim, 0)
            self.assertTrue(torch.isfinite(out[key]), key)
        torch.testing.assert_close(out["action_loss"], 1.0 * out["loss_oft"] + 0.5 * out["loss_gr00t"] + 2.0 * out["loss_pi"])
        self.assertEqual(model.qwen_vl_interface.calls, 1)
        out["action_loss"].backward()
        self.assertIsNotNone(model.qwen_vl_interface.embed.weight.grad)
        self.assertTrue(torch.any(model.qwen_vl_interface.embed.weight.grad != 0))
        for name in ("oft", "gr00t", "pi"):
            grads = [p.grad for p in model.heads[name].parameters() if p.grad is not None]
            self.assertTrue(grads and any(torch.any(g != 0) for g in grads), name)
        self.assertTrue(all(p.grad is not None for p in model.project_layers.parameters()))

    def test_head_routing_inside_forward(self):
        model = build_model(repeats=3)
        batch = 2
        model(make_examples(batch=batch))
        pi_call = model.heads["pi"].model.calls[-1]
        gr00t_call = model.heads["gr00t"].model.calls[-1]
        self.assertTrue(pi_call["layerwise"] and pi_call["return_pre_output"])
        self.assertFalse(gr00t_call["layerwise"] or gr00t_call["return_pre_output"])
        self.assertEqual(pi_call["batch"], batch * 3)
        self.assertEqual(gr00t_call["batch"], batch * 3)
        self.assertEqual(model.heads["oft"].calls[-1], (batch, HORIZON, HIDDEN))
        self.assertEqual(pi_call["mask"].shape[0], batch * 3)

    def test_disabled_head_reports_zero_and_is_excluded(self):
        model = build_model(heads=("oft", "pi"))
        out = model(make_examples())
        self.assertNotIn("gr00t", model.heads)
        self.assertEqual(out["loss_gr00t"].item(), 0.0)
        torch.testing.assert_close(out["action_loss"], out["loss_oft"] + out["loss_pi"])
        only_oft = build_model(heads=("oft",))
        out2 = only_oft(make_examples())
        torch.testing.assert_close(out2["action_loss"], out2["loss_oft"])
        self.assertEqual(out2["loss_pi"].item(), 0.0)

    def test_action_mask_removes_masked_dims_from_oft_loss(self):
        model = build_model(heads=("oft",), wrap=False)
        clean = make_examples(with_mask=True, seed=1)
        garbage = [dict(ex) for ex in clean]
        for ex in garbage:
            action = ex["action"].astype(np.float32)
            action[:, 12:18] = 1e3  # only masked slots differ
            ex["action"] = action
        loss_clean = model(clean)["loss_oft"]
        loss_garbage = model(garbage)["loss_oft"]
        torch.testing.assert_close(loss_clean, loss_garbage)
        unmasked = [{k: v for k, v in ex.items() if k not in ("action_mask", "periodic_mask")} for ex in garbage]
        self.assertGreater(model(unmasked)["loss_oft"].item(), loss_clean.item() * 10)

    def test_masks_reach_fm_heads_and_all_false_mask_gives_zero(self):
        torch.manual_seed(0)
        model = build_model()
        examples = make_examples(with_mask=True)
        out = model(examples)
        self.assertTrue(all(torch.isfinite(out[k]) for k in ("loss_oft", "loss_gr00t", "loss_pi")))
        for ex in examples:
            ex["action_mask"][:] = False
        out_zero = model(examples)
        self.assertEqual(out_zero["action_loss"].item(), 0.0)
        self.assertFalse(torch.isnan(out_zero["action_loss"]))
        out_zero["action_loss"].backward()

    def test_partial_masks_in_batch_are_rejected(self):
        model = build_model(heads=("oft",))
        examples = make_examples(with_mask=True)
        del examples[1]["action_mask"]
        with self.assertRaises(ValueError):
            model(examples)

    def test_state_is_discretised_into_the_instruction(self):
        model = build_model()
        instructions = model._prepare_instructions(make_examples(state=True))
        self.assertIn("[STATE]", instructions[0])
        self.assertIn("[ACTION]", instructions[0])
        self.assertEqual(instructions[0].count(ACTION_QUERY_TOKEN), HORIZON)
        out = model(make_examples(state=True))
        self.assertTrue(torch.isfinite(out["action_loss"]))
        no_oft = build_model(heads=("pi",))
        self.assertNotIn(ACTION_QUERY_TOKEN, no_oft._prepare_instructions(make_examples())[0])

    def test_oft_query_tokens_can_be_hidden_from_fm_heads(self):
        model = build_model(mask_queries=True, repeats=1)
        examples = make_examples()
        inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=None, instructions=model._prepare_instructions(examples))
        model(examples)
        mask = model.heads["gr00t"].model.calls[-1]["mask"]
        self.assertEqual(mask.dtype, torch.bool)
        torch.testing.assert_close(mask.sum(dim=1), inputs["attention_mask"].sum(dim=1) - HORIZON)
        plain = build_model(mask_queries=False, repeats=1)
        plain(examples)
        plain_mask = plain.heads["gr00t"].model.calls[-1]["mask"]
        torch.testing.assert_close(plain_mask.sum(dim=1), inputs["attention_mask"].sum(dim=1))

    def test_wrong_action_dim_without_layout_is_a_clear_error(self):
        model = build_model(heads=("oft",))
        with self.assertRaises(ValueError) as ctx:
            model(make_examples(dim=7))
        self.assertIn("unified_layout", str(ctx.exception))


class TestFlowMatchingLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.head = TinyFMHead(cond_dim=HIDDEN)
        self.cond = torch.randn(3, 5, HIDDEN)
        self.target = torch.rand(3, HORIZON, ACTION_DIM) * 2 - 1

    def _run(self, **kwargs):
        torch.manual_seed(123)
        return flow_matching_loss(self.head, self.cond, self.target, layerwise=False, **kwargs)

    def test_all_true_mask_equals_no_mask(self):
        ref = self._run()
        full = self._run(active_mask=torch.ones(ACTION_DIM, dtype=torch.bool))
        torch.testing.assert_close(ref["velocity_loss"], full["velocity_loss"])
        self.assertEqual(ref["wrap_loss"].item(), 0.0)
        torch.testing.assert_close(ref["loss"], ref["velocity_loss"])

    def test_wrap_term_only_with_periodic_dims(self):
        periodic = torch.zeros(ACTION_DIM, dtype=torch.bool)
        periodic[:12] = True
        out = self._run(active_mask=torch.ones(ACTION_DIM, dtype=torch.bool), periodic_mask=periodic, wrap_weight=0.5, period=2.0)
        self.assertGreater(out["wrap_loss"].item(), 0.0)
        torch.testing.assert_close(out["loss"], out["velocity_loss"] + 0.5 * out["wrap_loss"])
        no_wrap = self._run(active_mask=torch.ones(ACTION_DIM, dtype=torch.bool), periodic_mask=periodic, wrap_weight=0.0)
        self.assertEqual(no_wrap["wrap_loss"].item(), 0.0)

    def test_wrap_term_is_bounded_by_half_period(self):
        periodic = torch.ones(ACTION_DIM, dtype=torch.bool)
        out = self._run(periodic_mask=periodic, wrap_weight=1.0, period=2.0)
        self.assertLessEqual(out["wrap_loss"].item(), 1.0 + 1e-6)

    def test_layerwise_path_passes_return_pre_output(self):
        head = TinyFMHead(cond_dim=PI_WIDTH)
        conds = [torch.randn(3, 5, PI_WIDTH) for _ in range(NUM_LAYERS)]
        out = flow_matching_loss(head, conds, self.target, layerwise=True)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(head.model.calls[-1]["layerwise"] and head.model.calls[-1]["return_pre_output"])


class TestPredictAction(unittest.TestCase):
    def test_routes_to_requested_head(self):
        model = build_model(predict_head="gr00t")
        examples = make_examples(chunk=HORIZON)
        for head, counter in (("pi", lambda: model.heads["pi"].predict_calls), ("gr00t", lambda: model.heads["gr00t"].predict_calls), ("oft", lambda: len(model.heads["oft"].calls))):
            before = counter()
            out = model.predict_action(examples, head=head)
            self.assertEqual(out["normalized_actions"].shape, (2, HORIZON, ACTION_DIM))
            self.assertEqual(out["normalized_actions"].dtype, np.float32)
            self.assertEqual(out["head"], head)
            self.assertEqual(counter(), before + 1)
        self.assertEqual(model.predict_action(examples)["head"], "gr00t")
        single = model.predict_action(examples[0], head="oft")
        self.assertEqual(single["normalized_actions"].shape, (1, HORIZON, ACTION_DIM))
        with self.assertRaises(ValueError):
            model.predict_action(examples, head="fast")
        # trainer-side kwargs (use_ddim, num_ddim_steps) are accepted and ignored
        model.predict_action(examples, use_ddim=True, num_ddim_steps=20)


class TestUnifiedLayoutIntegration(unittest.TestCase):
    def test_mixed_embodiment_batch_trains_and_predicts_native_dims(self):
        torch.manual_seed(0)
        model = build_model(layout=True)
        franka = make_examples(batch=1, dim=7, robot_tag="franka", seed=1)
        agilex = make_examples(batch=1, dim=14, robot_tag="agilex", seed=2)
        out = model(franka + agilex)
        self.assertTrue(torch.isfinite(out["action_loss"]))
        target, active, periodic = model._collect_targets(franka + agilex)
        self.assertEqual(target.shape, (2, HORIZON, ACTION_DIM))
        self.assertEqual(active[0, 0].sum(), 7)
        self.assertEqual(active[1, 0].sum(), 14)
        self.assertEqual(periodic[1, 0].sum(), 12)
        self.assertEqual(periodic[0, 0].sum(), 0)
        self.assertEqual(model.predict_action(franka, robot_tag="franka")["normalized_actions"].shape, (1, HORIZON, 7))
        self.assertEqual(model.predict_action(agilex, robot_tag="agilex")["normalized_actions"].shape, (1, HORIZON, 14))
        self.assertEqual(model.predict_action(agilex)["normalized_actions"].shape, (1, HORIZON, ACTION_DIM))
        with self.assertRaises(KeyError):
            model(make_examples(batch=1, dim=7, robot_tag="ur5"))
        with self.assertRaises(ValueError):
            build_model(layout=False).predict_action(franka, robot_tag="franka")

    def test_pre_unified_samples_pass_through(self):
        model = build_model(layout=True)
        examples = make_examples(with_mask=True)
        target, active, _ = model._collect_targets(examples)
        self.assertEqual(target.shape, (2, HORIZON, ACTION_DIM))
        self.assertFalse(active[:, :, 12:18].any())
        self.assertTrue(torch.isfinite(model(examples)["action_loss"]))


if __name__ == "__main__":
    unittest.main()
