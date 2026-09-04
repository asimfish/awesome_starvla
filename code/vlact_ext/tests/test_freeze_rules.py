import os
import unittest
import warnings
from types import SimpleNamespace

import torch.nn as nn

from vlact_ext.freeze_rules import (
    DEFAULT_LLM_LAYERS_PATH,
    build_param_lr_groups,
    expand_to_exact_paths,
    freeze_backbones,
    freeze_by_rules,
    freeze_llm_layers_below,
    get_submodule,
    parse_rule,
    parse_rules,
    resolve_frozen_param_ids,
    resolve_frozen_params,
    split_rules,
)


class _Block(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.input_layernorm = nn.LayerNorm(dim)


class _LanguageModel(nn.Module):
    def __init__(self, dim=4, num_layers=6):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, dim)
        self.layers = nn.ModuleList([_Block(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)


class _Container(nn.Module):
    """Generic holder so the toy mirrors StarVLA's `qwen_vl_interface.model.model.{visual,language_model}` nesting."""

    def __init__(self, **children):
        super().__init__()
        for name, child in children.items():
            setattr(self, name, child)


def make_toy(num_layers=6, dim=4):
    inner = _Container(visual=nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)), language_model=_LanguageModel(dim, num_layers))
    hf_model = _Container(model=inner, lm_head=nn.Linear(dim, 10))
    interface = _Container(model=hf_model)
    return _Container(
        qwen_vl_interface=interface,
        heads=nn.ModuleDict({"oft": nn.Linear(dim, 7), "pi": nn.Linear(dim, 7)}),
        project_layers=nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)]),
    )


def starvla_style_ids(model, freeze_modules: str):
    """Re-implementation of StarVLA's exact-path parser (getattr chain) used as the reference."""
    ids = set()
    for path in [p.strip() for p in freeze_modules.split(",") if p.strip()]:
        module = model
        for attr in path.split("."):
            module = getattr(module, attr)
        ids |= {id(p) for p in module.parameters()}
    return ids


LAYERS = DEFAULT_LLM_LAYERS_PATH  # qwen_vl_interface.model.model.language_model.layers
VISUAL = "qwen_vl_interface.model.model.visual"


class TestParsing(unittest.TestCase):
    def test_split_rules(self):
        self.assertEqual(split_rules(""), [])
        self.assertEqual(split_rules(None), [])
        self.assertEqual(split_rules(" a.b , c ,"), ["a.b", "c"])
        self.assertEqual(split_rules(["a", "b,c"]), ["a", "b", "c"])
        with self.assertRaises(TypeError):
            split_rules(3)

    def test_parse_kinds(self):
        self.assertEqual(parse_rule("a.b.c").kind, "exact")
        rx = parse_rule(r"re:^a\.b\.\d+\.")
        self.assertEqual(rx.kind, "regex")
        self.assertTrue(rx.pattern.search("a.b.3.weight"))
        rg = parse_rule("a.layers[2:5].mlp")
        self.assertEqual((rg.kind, rg.path, rg.start, rg.stop, rg.suffix), ("range", "a.layers", 2, 5, "mlp"))
        rg2 = parse_rule("a.layers[:3]")
        self.assertEqual((rg2.start, rg2.stop, rg2.suffix), (None, 3, ""))
        rg3 = parse_rule("a.layers[-2:]")
        self.assertEqual((rg3.start, rg3.stop), (-2, None))
        with self.assertRaises(ValueError):
            parse_rule("a.layers[3]")

    def test_sugar_expands_to_range_rule(self):
        self.assertEqual(freeze_llm_layers_below(18), f"{LAYERS}[0:18]")
        rule = parse_rule("llm_layers_below:18")
        self.assertEqual((rule.kind, rule.path, rule.start, rule.stop), ("range", LAYERS, 0, 18))
        rule2 = parse_rule("llm_layers_below=4", llm_layers_path="llm.blocks")
        self.assertEqual((rule2.path, rule2.stop), ("llm.blocks", 4))
        rules = parse_rules("a.b", freeze_llm_layers_below_n=3)
        self.assertEqual([r.kind for r in rules], ["exact", "range"])
        self.assertEqual(parse_rules("a.b", freeze_llm_layers_below_n=0)[-1].kind, "exact")


class TestResolution(unittest.TestCase):
    def setUp(self):
        self.model = make_toy(num_layers=6)

    def test_exact_regex_range_and_sugar_select_the_same_params(self):
        exact = ",".join(f"{LAYERS}.{i}" for i in range(3))
        ref = starvla_style_ids(self.model, exact)
        self.assertTrue(ref)
        self.assertEqual(resolve_frozen_param_ids(self.model, exact), ref)
        escaped_layers = LAYERS.replace(".", "\\.")
        self.assertEqual(resolve_frozen_param_ids(self.model, "re:^" + escaped_layers + r"\.[0-2]\."), ref)
        self.assertEqual(resolve_frozen_param_ids(self.model, f"{LAYERS}[0:3]"), ref)
        self.assertEqual(resolve_frozen_param_ids(self.model, f"{LAYERS}[:3]"), ref)
        self.assertEqual(resolve_frozen_param_ids(self.model, "llm_layers_below:3"), ref)
        self.assertEqual(resolve_frozen_param_ids(self.model, "", freeze_llm_layers_below_n=3), ref)

    def test_vlact_shallow_freeze_set(self):
        rules = f"{VISUAL},qwen_vl_interface.model.model.language_model.embed_tokens,llm_layers_below:3"
        ids = resolve_frozen_param_ids(self.model, rules)
        exact = f"{VISUAL},qwen_vl_interface.model.model.language_model.embed_tokens," + ",".join(f"{LAYERS}.{i}" for i in range(3))
        self.assertEqual(ids, starvla_style_ids(self.model, exact))
        # upper layers, norm, lm_head and heads stay trainable
        for name in (f"{LAYERS}.3", f"{LAYERS}.5", "qwen_vl_interface.model.model.language_model.norm", "qwen_vl_interface.model.lm_head", "heads"):
            self.assertTrue(all(id(p) not in ids for p in get_submodule(self.model, name).parameters()), name)

    def test_range_with_suffix_and_negative_slice(self):
        ids = resolve_frozen_param_ids(self.model, f"{LAYERS}[0:2].mlp")
        ref = starvla_style_ids(self.model, f"{LAYERS}.0.mlp,{LAYERS}.1.mlp")
        self.assertEqual(ids, ref)
        ids_tail = resolve_frozen_param_ids(self.model, f"{LAYERS}[-2:]")
        self.assertEqual(ids_tail, starvla_style_ids(self.model, f"{LAYERS}.4,{LAYERS}.5"))

    def test_regex_can_target_parameter_names(self):
        params = resolve_frozen_params(self.model, r"re:\.bias$")
        self.assertTrue(params)
        self.assertTrue(all(name.endswith(".bias") for name in params))

    def test_missing_path_warns_or_raises(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ids = resolve_frozen_param_ids(self.model, "does.not.exist,heads.oft")
        self.assertEqual(ids, starvla_style_ids(self.model, "heads.oft"))
        self.assertTrue(any("does.not.exist" in str(w.message) for w in caught))
        with self.assertRaises(AttributeError):
            resolve_frozen_param_ids(self.model, "does.not.exist", strict=True)
        with self.assertRaises(ValueError):
            resolve_frozen_param_ids(self.model, r"re:nothing_matches_this", strict=True)
        with self.assertRaises(TypeError):
            resolve_frozen_param_ids(self.model, f"{VISUAL}.0[0:1]", strict=True)  # a Linear has no length

    def test_freeze_by_rules_sets_requires_grad(self):
        report = freeze_by_rules(self.model, f"{VISUAL},llm_layers_below:2")
        frozen_ids = starvla_style_ids(self.model, f"{VISUAL},{LAYERS}.0,{LAYERS}.1")
        for name, p in self.model.named_parameters():
            self.assertEqual(p.requires_grad, id(p) not in frozen_ids, name)
        self.assertEqual(len(report.frozen), len(frozen_ids))
        self.assertEqual(report.unmatched, [])
        self.assertTrue(all(n.startswith(("qwen_vl_interface.model.model.visual", LAYERS)) for n in report.frozen))

    def test_expand_to_exact_paths_for_unmodified_starvla(self):
        expanded = expand_to_exact_paths(self.model, f"{VISUAL},llm_layers_below:3,{LAYERS}[4:5].mlp")
        self.assertEqual(
            expanded,
            f"{VISUAL},{LAYERS}.0,{LAYERS}.1,{LAYERS}.2,{LAYERS}.4.mlp",
        )
        # the expanded string is consumable by StarVLA's own getattr-chain parser
        self.assertEqual(starvla_style_ids(self.model, expanded), resolve_frozen_param_ids(self.model, expanded))
        with self.assertRaises(ValueError):
            expand_to_exact_paths(self.model, r"re:\.bias$")


class _Cfg(dict):
    """dict with attribute access and .get, standing in for OmegaConf in tests."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class TestTrainerReplacements(unittest.TestCase):
    def test_build_param_lr_groups_excludes_frozen_and_folds_sugar(self):
        model = make_toy(num_layers=6)
        cfg = SimpleNamespace(
            trainer=_Cfg(
                learning_rate=_Cfg(base=2.5e-5, qwen_vl_interface=1e-5, heads=1e-4),
                freeze_modules=VISUAL,
                freeze_llm_layers_below=3,
            )
        )
        groups = build_param_lr_groups(model, cfg)
        by_name = {g["name"]: g for g in groups}
        self.assertEqual(set(by_name), {"qwen_vl_interface", "heads", "base"})
        self.assertEqual(by_name["qwen_vl_interface"]["lr"], 1e-5)
        self.assertEqual(by_name["heads"]["lr"], 1e-4)
        frozen = starvla_style_ids(model, f"{VISUAL}," + ",".join(f"{LAYERS}.{i}" for i in range(3)))
        all_grouped = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(all_grouped), len(set(all_grouped)))
        self.assertTrue(frozen.isdisjoint(all_grouped))
        expected_trainable = {id(p) for p in model.parameters()} - frozen
        self.assertEqual(set(all_grouped), expected_trainable)
        self.assertEqual({id(p) for p in by_name["base"]["params"]}, {id(p) for p in model.project_layers.parameters()})
        # the sugar key was folded into freeze_modules so the later freeze_backbones() call agrees
        self.assertIn(f"{LAYERS}[0:3]", cfg.trainer.freeze_modules)
        freeze_backbones(model, cfg.trainer.freeze_modules)
        self.assertEqual({id(p) for p in model.parameters() if not p.requires_grad}, frozen)

    def test_freeze_backbones_ignores_bool_like_starvla(self):
        model = make_toy()
        freeze_backbones(model, True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        freeze_backbones(model, "")
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


@unittest.skipUnless(os.environ.get("VLACT_TEST_QWEN3VL") == "1", "set VLACT_TEST_QWEN3VL=1 (needs transformers; slow import)")
class TestRealQwen3VLPaths(unittest.TestCase):
    def test_default_paths_exist_on_tiny_qwen3vl(self):
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        cfg = Qwen3VLConfig(
            text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, vocab_size=512, max_position_embeddings=128, rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]}),
            vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=32, patch_size=14, spatial_merge_size=2, temporal_patch_size=2, in_channels=3, deepstack_visual_indexes=[0, 1]),
        )
        interface = _Container(model=Qwen3VLForConditionalGeneration(cfg))
        model = _Container(qwen_vl_interface=interface, heads=nn.ModuleDict({"oft": nn.Linear(32, 7)}))
        ids = resolve_frozen_param_ids(model, f"{VISUAL},qwen_vl_interface.model.model.language_model.embed_tokens,llm_layers_below:2", strict=True)
        expected = starvla_style_ids(model, f"{VISUAL},qwen_vl_interface.model.model.language_model.embed_tokens,{LAYERS}.0,{LAYERS}.1")
        self.assertEqual(ids, expected)


if __name__ == "__main__":
    unittest.main()
