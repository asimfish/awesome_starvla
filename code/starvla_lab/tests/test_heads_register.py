"""CPU tests for Qwen_MultiHeadLab with the same mock backbone / tiny heads as vlact_ext's multihead tests."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from vlact_ext.multihead_framework import ACTION_QUERY_TOKEN, FRAMEWORK_REGISTRY, Qwen_MultiHead, gather_action_token_embeddings
from vlact_ext.tests.test_multihead_framework import (
    ACTION_DIM,
    HIDDEN,
    HORIZON,
    NUM_LAYERS,
    PI_WIDTH,
    MockVLM,
    TinyFMHead,
    TinyOFTHead,
    _NS,
    make_config,
    make_examples,
)

from starvla_lab.heads import (
    AUX_LOSS_KEYS,
    DEFAULT_AUX_HEADS,
    FutureFeaturePredictionHead,
    KeyframeHead,
    KeyframeWritePolicy,
    Qwen_MultiHeadLab,
    QwenMultiHeadLabDefaultConfig,
    keyframe_bce_loss,
    soft_keyframe_labels,
)

D_FEAT = 6
OFFSETS = (1, HORIZON)
ALL_LOSS_KEYS = ("action_loss", "loss_oft", "loss_pi", "loss_gr00t", "loss_featpred", "loss_keyframe")


def make_lab_config(featpred=True, keyframe=True, fp_weight=0.5, kf_weight=2.0, offsets=OFFSETS, d_feat=D_FEAT, sigma=1.0, horizon=None, aux_section=True, **cfg_kwargs):
    cfg = make_config(**cfg_kwargs)
    cfg.framework.name = "QwenMultiHeadLab"
    if aux_section:
        cfg.framework.aux_heads = _NS(
            featpred=_NS(enabled=featpred, weight=fp_weight, offsets=list(offsets), d_feat=d_feat, pooling="offset"),
            keyframe=_NS(enabled=keyframe, weight=kf_weight, sigma=sigma, horizon=horizon, threshold=0.6, nms_window=1, cooldown=3, max_events=2),
        )
    return cfg


def tiny_heads(names):
    heads, project_layers = {}, None
    if "oft" in names:
        heads["oft"] = TinyOFTHead(HIDDEN, ACTION_DIM)
    if "gr00t" in names:
        heads["gr00t"] = TinyFMHead(cond_dim=HIDDEN)
    if "pi" in names:
        heads["pi"] = TinyFMHead(cond_dim=PI_WIDTH)
        project_layers = nn.ModuleList([nn.Linear(HIDDEN, PI_WIDTH) for _ in range(NUM_LAYERS)])
    return heads, project_layers


def build_lab(aux_heads=None, **kwargs):
    names = kwargs.get("heads", ("oft", "gr00t", "pi"))
    cfg = make_lab_config(**kwargs)
    heads, project_layers = tiny_heads(names)
    return Qwen_MultiHeadLab(cfg, vlm=MockVLM(), heads=heads, project_layers=project_layers, aux_heads=aux_heads)


def build_parent(**kwargs):
    names = kwargs.get("heads", ("oft", "gr00t", "pi"))
    heads, project_layers = tiny_heads(names)
    return Qwen_MultiHead(make_config(**kwargs), vlm=MockVLM(), heads=heads, project_layers=project_layers)


def add_aux_fields(examples, featpred=True, keyframe=True, seed=0, offsets=OFFSETS, d_feat=D_FEAT):
    rng = np.random.default_rng(seed)
    for i, ex in enumerate(examples):
        if featpred:
            ex["future_features"] = rng.normal(size=(len(offsets), d_feat)).astype(np.float32)
        if keyframe:
            ex["keyframe_steps"] = [1 + i % HORIZON]
    return examples


class TestConstruction(unittest.TestCase):
    def test_registered_under_expected_key(self):
        self.assertIs(FRAMEWORK_REGISTRY["QwenMultiHeadLab"], Qwen_MultiHeadLab)
        self.assertTrue(issubclass(Qwen_MultiHeadLab, Qwen_MultiHead))
        self.assertEqual(QwenMultiHeadLabDefaultConfig().name, "QwenMultiHeadLab")
        self.assertEqual(QwenMultiHeadLabDefaultConfig().aux_heads, DEFAULT_AUX_HEADS)

    def test_config_plumbing(self):
        model = build_lab(fp_weight=0.25, kf_weight=3.0, sigma=0.7)
        self.assertEqual(list(model.aux_heads), ["featpred", "keyframe"])
        self.assertEqual(model.aux_weights, {"featpred": 0.25, "keyframe": 3.0})
        self.assertEqual(model.keyframe_sigma, 0.7)
        self.assertEqual(model.keyframe_horizon, HORIZON)
        self.assertEqual(model.aux_heads["featpred"].offsets, OFFSETS)
        self.assertEqual(model.aux_heads["featpred"].feat_dim, D_FEAT)
        self.assertIsInstance(model.keyframe_policy, KeyframeWritePolicy)
        self.assertEqual((model.keyframe_policy.threshold, model.keyframe_policy.nms_window, model.keyframe_policy.cooldown, model.keyframe_policy.max_events), (0.6, 1, 3, 2))
        # unspecified keys fall back to the module defaults
        self.assertEqual(model.aux_cfg["featpred"]["mse_weight"], DEFAULT_AUX_HEADS["featpred"]["mse_weight"])
        self.assertEqual(model.aux_cfg["keyframe"]["pos_weight"], None)
        # the parent's own plumbing is untouched
        self.assertEqual(list(model.heads), ["oft", "gr00t", "pi"])
        self.assertEqual(model.action_horizon, HORIZON)

    def test_missing_aux_section_disables_everything(self):
        model = build_lab(aux_section=False)
        self.assertEqual(len(model.aux_heads), 0)
        self.assertIsNone(model.keyframe_policy)
        self.assertIsNone(model.keyframe_horizon)

    def test_injected_aux_heads_and_unknown_names(self):
        injected = {"keyframe": KeyframeHead(HIDDEN, mlp_hidden=4), "featpred": FutureFeaturePredictionHead(HIDDEN, 3, offsets=[2], mlp_hidden=4)}
        model = build_lab(aux_heads=injected)
        self.assertIs(model.aux_heads["keyframe"], injected["keyframe"])
        self.assertEqual(model.aux_heads["featpred"].offsets, (2,))
        with self.assertRaises(ValueError):
            build_lab(aux_heads={"depth": KeyframeHead(HIDDEN)})
        cfg = make_lab_config()
        cfg.framework.aux_heads.segmentation = _NS(enabled=True)
        heads, project_layers = tiny_heads(("oft",))
        with self.assertRaises(ValueError):
            Qwen_MultiHeadLab(cfg, vlm=MockVLM(), heads=heads, project_layers=project_layers)


class TestForward(unittest.TestCase):
    def test_returns_all_losses_and_backprops_through_backbone_and_aux_heads(self):
        torch.manual_seed(0)
        model = build_lab(weights={"oft": 1.0, "gr00t": 0.5, "pi": 2.0}, fp_weight=0.5, kf_weight=2.0)
        out = model(add_aux_fields(make_examples()))
        for key in ALL_LOSS_KEYS:
            self.assertIn(key, out)
            self.assertEqual(out[key].ndim, 0)
            self.assertTrue(torch.isfinite(out[key]), key)
        self.assertGreater(out["loss_featpred"].item(), 0.0)
        self.assertGreater(out["loss_keyframe"].item(), 0.0)
        torch.testing.assert_close(
            out["action_loss"],
            1.0 * out["loss_oft"] + 0.5 * out["loss_gr00t"] + 2.0 * out["loss_pi"] + 0.5 * out["loss_featpred"] + 2.0 * out["loss_keyframe"],
        )
        self.assertEqual(model.qwen_vl_interface.calls, 1)
        out["action_loss"].backward()
        self.assertTrue(torch.any(model.qwen_vl_interface.embed.weight.grad != 0))
        for name in ("featpred", "keyframe"):
            grads = [p.grad for p in model.aux_heads[name].parameters()]
            self.assertTrue(all(g is not None for g in grads), name)
            self.assertTrue(any(torch.any(g != 0) for g in grads), name)
        for name in ("oft", "gr00t", "pi"):
            self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for p in model.heads[name].parameters()), name)

    def test_disabled_aux_heads_match_the_parent_exactly(self):
        examples = make_examples()
        torch.manual_seed(0)
        parent = build_parent()
        torch.manual_seed(0)
        lab = build_lab(featpred=False, keyframe=False)
        self.assertEqual(list(lab.state_dict()), list(parent.state_dict()))
        torch.manual_seed(1)
        ref = parent(examples)
        torch.manual_seed(1)
        out = lab(examples)
        for key in ("action_loss", "loss_oft", "loss_pi", "loss_gr00t"):
            torch.testing.assert_close(out[key], ref[key])
        self.assertEqual(out["loss_featpred"].item(), 0.0)
        self.assertEqual(out["loss_keyframe"].item(), 0.0)
        self.assertEqual(lab._prepare_instructions(examples), parent._prepare_instructions(examples))
        self.assertEqual(lab.predict_action(examples)["normalized_actions"].shape, (2, HORIZON, ACTION_DIM))
        self.assertNotIn("keyframe_probs", lab.predict_action(examples))

    def test_single_aux_head_leaves_the_other_at_zero(self):
        torch.manual_seed(0)
        model = build_lab(featpred=False)
        out = model(add_aux_fields(make_examples()))
        self.assertNotIn("featpred", model.aux_heads)
        self.assertEqual(out["loss_featpred"].item(), 0.0)
        self.assertGreater(out["loss_keyframe"].item(), 0.0)
        torch.testing.assert_close(out["action_loss"], out["loss_oft"] + out["loss_gr00t"] + out["loss_pi"] + 2.0 * out["loss_keyframe"])

    def test_aux_losses_match_a_manual_computation(self):
        torch.manual_seed(0)
        model = build_lab(sigma=0.8)
        examples = add_aux_fields(make_examples())
        out = model(examples)
        inputs, last_hidden, _ = model._encode([ex["image"] for ex in examples], model._prepare_instructions(examples))
        queries = gather_action_token_embeddings(last_hidden, inputs["input_ids"], model.action_token_id, HORIZON)
        featpred = model.aux_heads["featpred"]
        target = torch.as_tensor(np.stack([ex["future_features"] for ex in examples]))
        torch.testing.assert_close(out["loss_featpred"], featpred.loss(featpred(queries), target, torch.ones(2, len(OFFSETS), dtype=torch.bool))["loss"])
        keyframe = model.aux_heads["keyframe"]
        labels = soft_keyframe_labels([ex["keyframe_steps"] for ex in examples], HORIZON, 0.8)
        torch.testing.assert_close(out["loss_keyframe"], keyframe_bce_loss(keyframe(queries), labels))

    def test_samples_without_aux_fields_are_masked(self):
        torch.manual_seed(0)
        model = build_lab()
        # no sample carries the fields -> both losses are exactly zero, still differentiable
        out = model(make_examples())
        self.assertEqual(out["loss_featpred"].item(), 0.0)
        self.assertEqual(out["loss_keyframe"].item(), 0.0)
        self.assertFalse(torch.isnan(out["action_loss"]))
        out["action_loss"].backward()
        # one annotated sample next to an unannotated one -> loss equals that of the annotated sample alone
        # (the mock backbone is position-wise, so the query states of sample 0 do not depend on the batch)
        model.zero_grad()
        annotated = add_aux_fields(make_examples(batch=1, seed=3))
        mixed = annotated + make_examples(batch=1, seed=4)
        mixed_out = model(mixed)
        alone_out = model(annotated)
        torch.testing.assert_close(mixed_out["loss_featpred"], alone_out["loss_featpred"])
        torch.testing.assert_close(mixed_out["loss_keyframe"], alone_out["loss_keyframe"])
        # partial per-offset mask coming from targets_from_sequence-style dataloaders
        annotated[0]["future_features_mask"] = np.array([True, False])
        garbage = [dict(annotated[0])]
        garbage[0]["future_features"] = annotated[0]["future_features"].copy()
        garbage[0]["future_features"][1] = 1e3
        torch.testing.assert_close(model(garbage)["loss_featpred"], model(annotated)["loss_featpred"])

    def test_bad_future_feature_shape_is_a_clear_error(self):
        model = build_lab()
        examples = add_aux_fields(make_examples())
        examples[0]["future_features"] = np.zeros((1, D_FEAT), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            model(examples)
        self.assertIn("future_features", str(ctx.exception))

    def test_aux_heads_without_oft_add_the_query_tokens(self):
        torch.manual_seed(0)
        model = build_lab(heads=("pi",), predict_head="pi")
        examples = add_aux_fields(make_examples())
        instructions = model._prepare_instructions(examples)
        self.assertEqual(instructions[0].count(ACTION_QUERY_TOKEN), HORIZON)
        self.assertIsNotNone(model.action_token_id)
        out = model(examples)
        self.assertTrue(torch.isfinite(out["action_loss"]))
        self.assertGreater(out["loss_featpred"].item(), 0.0)
        self.assertGreater(out["loss_keyframe"].item(), 0.0)
        self.assertEqual(out["loss_oft"].item(), 0.0)
        self.assertEqual(model.predict_action(examples)["keyframe_probs"].shape, (2, HORIZON))

    def test_custom_keyframe_horizon(self):
        torch.manual_seed(0)
        model = build_lab(horizon=2 * HORIZON, featpred=False)
        self.assertEqual(model.keyframe_horizon, 2 * HORIZON)
        examples = add_aux_fields(make_examples())
        examples[0]["keyframe_steps"] = [2 * HORIZON - 1]
        self.assertTrue(torch.isfinite(model(examples)["loss_keyframe"]))
        self.assertEqual(model.predict_action(examples)["keyframe_probs"].shape, (2, 2 * HORIZON))


class TestPredictAction(unittest.TestCase):
    def test_returns_keyframe_probs_next_to_actions(self):
        torch.manual_seed(0)
        model = build_lab()
        examples = make_examples(chunk=HORIZON)
        for head in ("oft", "gr00t", "pi"):
            out = model.predict_action(examples, head=head)
            self.assertEqual(out["normalized_actions"].shape, (2, HORIZON, ACTION_DIM))
            self.assertEqual(out["head"], head)
            probs = out["keyframe_probs"]
            self.assertIsInstance(probs, np.ndarray)
            self.assertEqual(probs.shape, (2, HORIZON))
            self.assertEqual(probs.dtype, np.float32)
            self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1))
        single = model.predict_action(examples[0])
        self.assertEqual(single["keyframe_probs"].shape, (1, HORIZON))
        writes = model.keyframe_policy.decide(single["keyframe_probs"][0], t0=10)
        self.assertTrue(all(10 <= t < 10 + HORIZON for t in writes))
        self.assertIsNone(model._encoded)


if __name__ == "__main__":
    unittest.main()
