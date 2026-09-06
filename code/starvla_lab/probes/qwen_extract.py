"""Per-layer backbone representations of a StarVLA Qwen-VL framework for the WP1 drift probe.

The F0 diagnostics (``experiments/results/f0_libero_goal_smoke``) showed that a naive drift probe measures
the wrong thing: mean-pooled ``1 - CKA`` on a single-scene probe batch was 98% due to the ~45 prompt-token rows
of ``embed_tokens`` that fine-tuning updates (relative change 2e-5), carried into every layer's output by the
residual stream. :class:`QwenBackboneProbe` is the corrected extractor:

* the prompt is the plain VLM prompt (images + instruction), identical for every framework and free of
  framework-specific learnable tokens such as OFT's action queries;
* ``restore_pretrained_embeddings`` snapshots ``embed_tokens`` on the first call (the reference extraction,
  before training) and swaps the snapshot back in for every later forward, so what is measured is the drift
  of the transformer layers themselves;
* ``representation="token"`` treats every valid token as a CKA sample (evenly subsampled to ``max_tokens``
  positions fixed on the first call); ``"pooled"`` is the masked per-sample mean, kept as the secondary view.

:func:`stratified_probe_batch` / :func:`gather_probe_batch` build the probe batch round-robin over instructions
so a mixture of several suites (e.g. LIBERO-goal + LIBERO-spatial) covers every task of every scene.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from torch import Tensor, nn

__all__ = ["QwenBackboneProbe", "QWEN_IMAGE_TOKEN_ID", "REPRESENTATIONS", "TOKEN_SUBSETS", "framework_of", "stratified_probe_batch", "gather_probe_batch"]

QWEN_IMAGE_TOKEN_ID = 151655  # <|image_pad|> in Qwen2.5-VL / Qwen3-VL (StarVLA ``QWen3.IMAGE_TOKEN_INDEX``)
REPRESENTATIONS = ("token", "pooled")
TOKEN_SUBSETS = ("all", "image", "text")


def framework_of(model: nn.Module) -> nn.Module:
    """The StarVLA framework behind DDP / accelerate wrappers (anything exposing ``qwen_vl_interface``)."""
    if hasattr(model, "qwen_vl_interface"):
        return model
    for attr in ("module", "model"):
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module) and hasattr(inner, "qwen_vl_interface"):
            return inner
    raise AttributeError("QwenBackboneProbe needs a framework with a `qwen_vl_interface` (StarVLA Qwen-VL family)")


class QwenBackboneProbe:
    """Callable ``extract_fn(model, batch) -> [L] tensors`` for :class:`~starvla_lab.probes.DriftTracker`.

    ``batch`` is a list of raw StarVLA samples (``ex["image"]``: list of PIL images, ``ex["lang"]``: instruction).
    Returns one tensor per decoder layer (the embedding output ``hidden_states[0]`` is excluded): ``[T, d]`` for
    ``representation="token"`` (``T <= max_tokens`` fixed token positions) or ``[N, d]`` for ``"pooled"``, in fp32
    on the model device. ``token_subset`` restricts both views to image tokens, text tokens or all valid tokens.

    ``restore_pretrained_embeddings`` requires the full ``embed_tokens`` weight on this rank (single process, DDP,
    DeepSpeed ZeRO-1/2; not ZeRO-3 where parameters are partitioned). The snapshot lives in host memory
    (~780 MB for Qwen3-VL-4B); each probe forward temporarily copies it onto the device.
    """

    def __init__(
        self,
        representation: str = "token",
        token_subset: str = "all",
        max_tokens: int = 4096,
        restore_pretrained_embeddings: bool = True,
        image_token_id: int = QWEN_IMAGE_TOKEN_ID,
    ) -> None:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}, got {representation!r}")
        if token_subset not in TOKEN_SUBSETS:
            raise ValueError(f"token_subset must be one of {TOKEN_SUBSETS}, got {token_subset!r}")
        if max_tokens < 2:
            raise ValueError("max_tokens must be >= 2 (CKA needs at least two samples)")
        self.representation = representation
        self.token_subset = token_subset
        self.max_tokens = int(max_tokens)
        self.restore_pretrained_embeddings = restore_pretrained_embeddings
        self.image_token_id = image_token_id
        self._token_index: Optional[Tensor] = None
        self._embed_snapshot: Optional[Tensor] = None
        self.last_token_counts: Dict[str, int] = {}
        # Filled on every call after the snapshot exists: how far the live embed_tokens has moved from it.
        self.embed_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ embeddings
    @staticmethod
    def embedding_weight(fw: nn.Module) -> Tensor:
        return fw.qwen_vl_interface.model.get_input_embeddings().weight

    def _update_embed_stats(self, live: Tensor, snapshot: Tensor, rows_per_chunk: int = 16384) -> None:
        changed, sq_diff, sq_ref = 0, 0.0, 0.0
        for lo in range(0, live.shape[0], rows_per_chunk):
            a = live[lo : lo + rows_per_chunk].float()
            b = snapshot[lo : lo + rows_per_chunk].float()
            changed += int((a != b).any(dim=1).sum())
            sq_diff += float((a - b).pow(2).sum())
            sq_ref += float(b.pow(2).sum())
        self.embed_stats = {"changed_rows": changed, "relative_frobenius_change": (sq_diff ** 0.5) / max(sq_ref ** 0.5, 1e-12)}

    @contextmanager
    def _pretrained_embeddings(self, fw: nn.Module) -> Iterator[None]:
        if not self.restore_pretrained_embeddings:
            yield
            return
        weight = self.embedding_weight(fw).data
        if self._embed_snapshot is None:
            self._embed_snapshot = weight.detach().to("cpu", copy=True)
            self.embed_stats = {"changed_rows": 0, "relative_frobenius_change": 0.0}
            yield
            return
        snapshot = self._embed_snapshot.to(weight.device)
        live = weight.clone()
        self._update_embed_stats(live, snapshot)
        weight.copy_(snapshot)
        del snapshot
        try:
            yield
        finally:
            weight.copy_(live)

    # ------------------------------------------------------------------ extraction
    @torch.no_grad()
    def __call__(self, model: nn.Module, batch: Sequence[dict]) -> List[Tensor]:
        fw = framework_of(model)
        vlm = fw.qwen_vl_interface
        inputs = vlm.build_qwenvl_inputs(images=[ex["image"] for ex in batch], instructions=[ex["lang"] for ex in batch])
        device = vlm.model.device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        # Same compute path as training: StarVLA wraps the backbone in bf16 autocast, which is also what makes a
        # backbone with mixed fp32 (trainable, `backbone_fp32`) and bf16 (frozen) parameters run at all.
        with self._pretrained_embeddings(fw), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            try:  # only the hidden states are needed; skip the full-vocabulary logits when the model supports it
                out = vlm.model(**inputs, output_hidden_states=True, return_dict=True, logits_to_keep=1)
            except TypeError:
                out = vlm.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[1:]

        valid = inputs["attention_mask"].bool()
        image = valid & (inputs["input_ids"] == self.image_token_id)
        masks = {"all": valid, "image": image, "text": valid & ~image}
        self.last_token_counts = {k: int(v.sum()) for k, v in masks.items()}
        mask = masks[self.token_subset]

        if self.representation == "pooled":
            m = mask.unsqueeze(-1).float()
            # Pool in fp32: a bf16 mean over ~150 tokens keeps ~3 significant digits and would hide small drifts.
            return [((h.float() * m).sum(1) / m.sum(1).clamp(min=1.0)) for h in hidden]

        if self._token_index is None:
            idx = mask.flatten().nonzero(as_tuple=False).flatten()
            if idx.numel() < 2:
                raise ValueError(f"token_subset={self.token_subset!r} selects {idx.numel()} tokens; CKA needs at least two")
            if idx.numel() > self.max_tokens:
                pick = torch.linspace(0, idx.numel() - 1, self.max_tokens, device=idx.device).round().long()
                idx = idx[pick]
            self._token_index = idx.cpu()
        idx = self._token_index.to(device)
        return [h.flatten(0, 1)[idx].float() for h in hidden]


# ---------------------------------------------------------------------- probe batch construction
def stratified_probe_batch(samples: Sequence[dict], n: int, key: str = "lang") -> List[dict]:
    """Pick ``n`` samples round-robin over the distinct values of ``samples[i][key]`` (first-appearance order).

    With ``key="lang"`` every instruction (= LIBERO task) gets a sample before any instruction gets a second one,
    so a pool drawn from several suites yields a batch that covers every task of every scene. Returns fewer than
    ``n`` samples only when the pool is smaller than ``n``.
    """
    groups: Dict[str, List[dict]] = {}
    for ex in samples:
        groups.setdefault(str(ex.get(key, "")), []).append(ex)
    order = list(groups)
    out: List[dict] = []
    i = 0
    while len(out) < n and any(groups.values()):
        g = groups[order[i % len(order)]]
        if g:
            out.append(g.pop(0))
        i += 1
    return out


def gather_probe_batch(loader: Iterable[Sequence[dict]], n: int, stratify: bool = True, pool_factor: int = 4, key: str = "lang") -> List[dict]:
    """Collect ``n`` probe samples from a StarVLA loader whose ``collate_fn`` yields lists of raw sample dicts.

    Loader batches are accumulated until ``n * pool_factor`` samples (``n`` when not stratifying) are available,
    then :func:`stratified_probe_batch` picks ``n`` of them spread over instructions.
    """
    if n < 2:
        raise ValueError("a probe batch needs at least two samples")
    want = n * (max(1, int(pool_factor)) if stratify else 1)
    pool: List[dict] = []
    it = iter(loader)
    while len(pool) < want:
        pool.extend(next(it))
    return stratified_probe_batch(pool, n, key) if stratify else pool[:n]
