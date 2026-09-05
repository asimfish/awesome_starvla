from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starvla_lab.data.mixtures import parse_mixture_spec, register_mixture
from starvla_lab.probes import DriftTracker, QwenBackboneProbe, gather_probe_batch, layerwise_cka, stratified_probe_batch
from starvla_lab.probes.qwen_extract import framework_of

IMAGE_ID = 7


class _HFLike(nn.Module):
    """Stands in for the HF Qwen-VL model: ``device``, ``get_input_embeddings`` and ``forward(..., output_hidden_states)``."""

    def __init__(self, vocab=64, dim=8, n_layers=3):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    @property
    def device(self):
        return self.embed.weight.device

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask, output_hidden_states=False, return_dict=True):
        h = self.embed(input_ids)
        hidden = [h]
        for layer in self.layers:
            h = h + torch.tanh(layer(h))
            hidden.append(h)
        return SimpleNamespace(hidden_states=tuple(hidden), last_hidden_state=h)


class _Interface(nn.Module):
    """Stands in for StarVLA's ``qwen_vl_interface``: ``model`` + ``build_qwenvl_inputs``."""

    def __init__(self):
        super().__init__()
        self.model = _HFLike()

    def build_qwenvl_inputs(self, images, instructions):
        seqs = [[IMAGE_ID] * (4 * len(imgs)) + [10 + (ord(c) % 40) for c in text] for imgs, text in zip(images, instructions)]
        length = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), length, dtype=torch.long)
        mask = torch.zeros(len(seqs), length, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _Framework(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = _Interface()


class _Wrapper(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.module = inner


def _batch():
    return [
        {"image": ["img", "img"], "lang": "put the bowl on the plate"},
        {"image": ["img", "img"], "lang": "open the drawer"},
        {"image": ["img", "img"], "lang": "push the plate to the front"},
        {"image": ["img", "img"], "lang": "turn on the stove"},
    ]


def test_token_and_pooled_views_have_expected_shapes_and_fixed_token_positions():
    fw = _Framework()
    batch = _batch()
    token = QwenBackboneProbe(representation="token", max_tokens=10, image_token_id=IMAGE_ID)
    reps = token(fw, batch)
    assert len(reps) == 3 and all(r.shape == (10, 8) and r.dtype == torch.float32 for r in reps)
    n_valid = sum(len(ex["lang"]) + 8 for ex in batch)
    assert token.last_token_counts == {"all": n_valid, "image": 32, "text": n_valid - 32}
    again = token(fw, batch)
    assert all(torch.equal(a, b) for a, b in zip(reps, again))  # same positions, same model -> identical
    pooled = QwenBackboneProbe(representation="pooled", image_token_id=IMAGE_ID)
    assert all(r.shape == (4, 8) for r in pooled(fw, batch))
    # fewer valid tokens than max_tokens: all of them are used
    assert QwenBackboneProbe(representation="token", max_tokens=4096, image_token_id=IMAGE_ID)(fw, batch)[0].shape == (n_valid, 8)


def test_token_subset_selects_only_those_tokens():
    fw = _Framework()
    batch = _batch()
    inputs = fw.qwen_vl_interface.build_qwenvl_inputs([ex["image"] for ex in batch], [ex["lang"] for ex in batch])
    ids = inputs["input_ids"].flatten()
    text = QwenBackboneProbe(representation="token", token_subset="text", max_tokens=8, image_token_id=IMAGE_ID)
    text(fw, batch)
    assert torch.all(ids[text._token_index] != IMAGE_ID) and torch.all(ids[text._token_index] != 0)
    image = QwenBackboneProbe(representation="token", token_subset="image", max_tokens=8, image_token_id=IMAGE_ID)
    image(fw, batch)
    assert torch.all(ids[image._token_index] == IMAGE_ID)
    with pytest.raises(ValueError):
        QwenBackboneProbe(representation="cls")
    with pytest.raises(ValueError):
        QwenBackboneProbe(token_subset="patches")


def test_restoring_pretrained_embeddings_removes_embedding_drift_and_leaves_live_weights_intact():
    fw = _Framework()
    batch = _batch()
    restoring = QwenBackboneProbe(representation="pooled", restore_pretrained_embeddings=True, image_token_id=IMAGE_ID)
    naive = QwenBackboneProbe(representation="pooled", restore_pretrained_embeddings=False, image_token_id=IMAGE_ID)
    ref_restoring, ref_naive = restoring(fw, batch), naive(fw, batch)
    assert restoring.embed_stats == {"changed_rows": 0, "relative_frobenius_change": 0.0}

    weight = fw.qwen_vl_interface.model.embed.weight
    rows = sorted({10 + (ord(c) % 40) for ex in batch for c in ex["lang"]})[:5]
    with torch.no_grad():
        weight[rows] += 0.5  # "training" moved a few prompt-token rows, as in F0
    live_after_training = weight.detach().clone()

    cur = restoring(fw, batch)
    assert all(torch.allclose(a, b) for a, b in zip(ref_restoring, cur)), "with the snapshot swapped in, the layers are unchanged"
    assert restoring.embed_stats["changed_rows"] == len(rows) and restoring.embed_stats["relative_frobenius_change"] > 0
    assert torch.equal(weight.detach(), live_after_training), "the probe must put the live embeddings back"

    cur_naive = naive(fw, batch)
    assert not all(torch.allclose(a, b) for a, b in zip(ref_naive, cur_naive)), "the naive view sees the embedding update"

    # and a real layer change is still visible through the restoring probe
    with torch.no_grad():
        fw.qwen_vl_interface.model.layers[1].weight.mul_(3.0)
    drift = 1.0 - layerwise_cka(ref_restoring, restoring(fw, batch))
    assert drift[0] < 1e-9 and drift[1] > 1e-4 and drift[2] > 1e-4


def test_probe_works_through_ddp_like_wrapper_and_with_drift_tracker_compute_device():
    fw = _Framework()
    assert framework_of(_Wrapper(fw)) is fw
    with pytest.raises(AttributeError):
        framework_of(nn.Linear(2, 2))
    probe = QwenBackboneProbe(representation="token", max_tokens=16, image_token_id=IMAGE_ID)
    tracker = DriftTracker(probe, _batch(), reference=_Wrapper(fw), compute_device="cpu")
    assert torch.all(tracker.update(fw, step=0) < 1e-9)
    with torch.no_grad():
        fw.qwen_vl_interface.model.layers[2].weight.mul_(-2.0)
    drift = tracker.update(fw, step=1)
    assert drift[0] < 1e-9 and drift[1] < 1e-9 and drift[2] > 1e-4


def test_stratified_probe_batch_round_robins_over_instructions():
    pool = [{"lang": "A", "i": i} for i in range(5)] + [{"lang": "B", "i": i} for i in range(2)] + [{"lang": "C", "i": i} for i in range(3)]
    pool = [pool[0], pool[5], pool[7], *pool[1:5], pool[6], *pool[8:]]  # first appearance order A, B, C
    picked = stratified_probe_batch(pool, 6)
    assert [ex["lang"] for ex in picked] == ["A", "B", "C", "A", "B", "C"]
    picked = stratified_probe_batch(pool, 9)
    assert [ex["lang"] for ex in picked] == ["A", "B", "C", "A", "B", "C", "A", "C", "A"]  # B exhausted after 2
    assert [ex["i"] for ex in picked if ex["lang"] == "A"] == [0, 1, 2, 3]  # order within a group preserved
    assert len(stratified_probe_batch(pool, 50)) == 10  # never more than the pool


def test_gather_probe_batch_accumulates_loader_batches():
    samples = [{"lang": f"task {i % 3}", "i": i} for i in range(24)]
    loader = [samples[k : k + 4] for k in range(0, 24, 4)]  # StarVLA collate_fn: list of raw dicts
    batch = gather_probe_batch(loader, 6, stratify=True, pool_factor=2)
    assert len(batch) == 6 and [ex["lang"] for ex in batch] == ["task 0", "task 1", "task 2"] * 2
    assert max(ex["i"] for ex in batch) < 12  # only 6 * 2 samples were pulled from the loader
    assert [ex["i"] for ex in gather_probe_batch(loader, 6, stratify=False)] == list(range(6))
    with pytest.raises(ValueError):
        gather_probe_batch(loader, 1)


def test_register_mixture_inline_spec_and_existing_name():
    registry = {"libero_goal": [("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka")]}
    assert register_mixture("libero_goal", registry) == "libero_goal"
    spec = "libero_goal_no_noops_1.0.0_lerobot:libero_franka, libero_spatial_no_noops_1.0.0_lerobot:libero_franka:2"
    name = register_mixture(spec, registry)
    assert name.startswith("lab_probe_") and registry[name] == [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 2.0, "libero_franka"),
    ]
    assert register_mixture(spec, registry) == name and len(registry) == 2
    assert parse_mixture_spec("a:b") == [("a", 1.0, "b")]
    for bad in ("", "onlyname", "a:b:c:d", "a:b:-1"):
        with pytest.raises(ValueError):
            parse_mixture_spec(bad)
