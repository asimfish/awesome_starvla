#!/usr/bin/env python3
"""Probe-methodology diagnostics for the WP1 drift metric on a real fine-tuned checkpoint (GPU, ~5 min).

Questions answered on one fixed probe batch drawn from the training dataloader:

1. noise floor        -- drift (1 - CKA) between two extractions of the *same* pretrained VLM;
2. pooling variants   -- drift pretrained -> fine-tuned when pooling all tokens / image tokens only /
                         text tokens only, and token-level CKA (every valid token is a sample);
3. embedding leakage  -- the same drift after copying the pretrained ``embed_tokens`` back into the
                         fine-tuned model: what remains is drift caused by the trained layers themselves.

Motivation: in F0 the mean-pooled drift of *frozen* layers 0-17 was as large as 1e-2 and non-monotonic.
The residual stream carries ``embed_tokens`` straight into every layer's output, and on a probe batch whose
images are near-identical (one LIBERO scene) the centred Gram matrix is dominated by the few instruction
tokens, so CKA reacts to word-embedding updates rather than to the visual-language representation.

    PYTHONPATH=<awesome_starvla>/code:<StarVLA> python scripts/probe_diagnostics.py \
        --config code/starvla_lab/configs/f0_libero_goal_smoke.yaml \
        --checkpoint <run_dir>/final_model/pytorch_model.pt --out <dir> [--probe_batch_size 32]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "code") not in sys.path:
    sys.path.insert(0, str(REPO / "code"))

from starvla_lab.probes.cka import layerwise_cka, linear_cka  # noqa: E402

IMAGE_TOKEN_ID = 151655  # Qwen3-VL <|image_pad|> (StarVLA QWen3.IMAGE_TOKEN_INDEX)


def load_probe_batch(cfg, n: int) -> List[dict]:
    from starVLA.dataloader import build_dataloader

    cfg.datasets.vla_data.per_device_batch_size = min(n, 8)
    cfg.output_dir = str(Path(cfg.run_root_dir) / "_probe_diag")
    loader = build_dataloader(cfg=cfg, dataset_py="lerobot_datasets")
    batch, it = [], iter(loader)
    while len(batch) < n:
        batch.extend(next(it))
    return batch[:n]


@torch.no_grad()
def hidden_states(vlm, batch: List[dict]):
    inputs = vlm.build_qwenvl_inputs(images=[ex["image"] for ex in batch], instructions=[ex["lang"] for ex in batch])
    device = vlm.model.device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = vlm.model(**inputs, output_hidden_states=True, return_dict=True)
    valid = inputs["attention_mask"].bool()
    image = valid & (inputs["input_ids"] == IMAGE_TOKEN_ID)
    text = valid & ~image
    return [h.float() for h in out.hidden_states[1:]], {"all": valid, "image": image, "text": text}


def pooled(hs: List[torch.Tensor], mask: torch.Tensor) -> List[torch.Tensor]:
    m = mask.unsqueeze(-1).float()
    return [((h * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu() for h in hs]


def token_level(hs: List[torch.Tensor], mask: torch.Tensor, max_tokens: int = 4096) -> List[torch.Tensor]:
    idx = mask.flatten().nonzero(as_tuple=False).flatten()
    if idx.numel() > max_tokens:
        idx = idx[torch.linspace(0, idx.numel() - 1, max_tokens).long()]
    return [h.flatten(0, 1)[idx].cpu() for h in hs]


def drift(a: List[torch.Tensor], b: List[torch.Tensor]) -> Dict[str, float]:
    d = (1.0 - layerwise_cka(a, b)).clamp(0.0, 1.0)
    return {
        "mean": float(d.mean()), "max": float(d.max()), "argmax": int(d.argmax()),
        "frozen_0_17_mean": float(d[:18].mean()), "trainable_18_35_mean": float(d[18:].mean()),
        "per_layer": [round(float(x), 6) for x in d],
    }


def load_finetuned_backbone(vlm, checkpoint: Path) -> int:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = "qwen_vl_interface."
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = vlm.load_state_dict(sub, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading backbone: {unexpected[:5]}")
    return len(sub)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="StarVLA final_model/pytorch_model.pt of a fine-tuned run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe_batch_size", type=int, default=32)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from starVLA.model.modules.vlm import get_vlm_model

    cfg = OmegaConf.load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    batch = load_probe_batch(cfg, args.probe_batch_size)
    print(f"[diag] probe batch: {len(batch)} samples, {len({ex['lang'] for ex in batch})} distinct instructions")

    vlm = get_vlm_model(config=cfg).to(device).eval()
    hs_ref, masks = hidden_states(vlm, batch)
    n_tok = {k: int(v.sum()) for k, v in masks.items()}
    print(f"[diag] tokens per batch: {n_tok}")
    ref = {k: pooled(hs_ref, masks[k]) for k in ("all", "image", "text")}
    ref["token"] = token_level(hs_ref, masks["all"])

    report: Dict[str, dict] = {"n_samples": len(batch), "tokens": n_tok}

    hs_again, _ = hidden_states(vlm, batch)
    report["noise_floor_all_tokens"] = drift(ref["all"], pooled(hs_again, masks["all"]))
    report["noise_floor_token_level"] = drift(ref["token"], token_level(hs_again, masks["all"]))
    del hs_again
    print(f"[diag] noise floor (same model twice): pooled mean {report['noise_floor_all_tokens']['mean']:.2e}, "
          f"token-level mean {report['noise_floor_token_level']['mean']:.2e}")

    embed_pretrained = vlm.model.model.language_model.embed_tokens.weight.detach().clone()
    n_loaded = load_finetuned_backbone(vlm, Path(args.checkpoint))
    print(f"[diag] loaded {n_loaded} backbone tensors from {args.checkpoint}")
    embed_ft = vlm.model.model.language_model.embed_tokens.weight.detach()
    changed_rows = int((embed_ft != embed_pretrained).any(dim=1).sum())
    rel = ((embed_ft - embed_pretrained).norm() / embed_pretrained.norm()).item()
    report["embed_tokens"] = {"changed_rows": changed_rows, "relative_frobenius_change": rel}
    print(f"[diag] embed_tokens: {changed_rows} rows changed, relative Frobenius change {rel:.2e}")

    hs_ft, _ = hidden_states(vlm, batch)
    for k in ("all", "image", "text"):
        report[f"finetuned_pooled_{k}"] = drift(ref[k], pooled(hs_ft, masks[k]))
    report["finetuned_token_level"] = drift(ref["token"], token_level(hs_ft, masks["all"]))
    del hs_ft

    vlm.model.model.language_model.embed_tokens.weight.data.copy_(embed_pretrained)
    hs_swap, _ = hidden_states(vlm, batch)
    for k in ("all", "image", "text"):
        report[f"finetuned_pretrained_embed_pooled_{k}"] = drift(ref[k], pooled(hs_swap, masks[k]))
    report["finetuned_pretrained_embed_token_level"] = drift(ref["token"], token_level(hs_swap, masks["all"]))

    (out / "probe_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = [k for k in report if k.startswith("finetuned") or k.startswith("noise")]
    print("\n| measurement | mean drift | frozen L0-17 | trainable L18-35 | max (layer) |\n|---|---:|---:|---:|---|")
    for k in rows:
        r = report[k]
        print(f"| {k} | {r['mean']:.4f} | {r['frozen_0_17_mean']:.4f} | {r['trainable_18_35_mean']:.4f} | {r['max']:.4f} (L{r['argmax']}) |")
    print(f"\n[done] wrote {out / 'probe_diagnostics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
