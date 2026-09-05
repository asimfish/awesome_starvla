#!/usr/bin/env python3
"""WP1 cross-head probe on fine-tuned backbones: did the heads write action information into the backbone, or
erode what the pretrained VLM encoded? (GPU, ~10 min for one pretrained + five fine-tuned backbones.)

Two head-agnostic measurements on one fixed sample set (stratified over instructions, drawn from a LeRobot
mixture that may include suites the models never trained on):

1. action readability -- ridge probe from per-sample pooled hidden states (layer ``l``, all or image tokens)
   to the raw action chunk ``[K, D]``; held-out R^2 / MAE per suite (fit and evaluate inside the same suite)
   and cross-suite (fit on suite A, evaluate on suite B). Higher than the pretrained VLM = the fine-tuning
   made actions more linearly decodable from the backbone ("written in").
2. retention -- token-level ridge map from the fine-tuned layer-``l`` token states to the *pretrained* states
   of the same tokens, held-out over samples; R^2 close to 1 = the pretrained representation is still
   linearly recoverable ("rewritten, not erased"); a drop = information lost. The reverse map (pretrained ->
   fine-tuned) says how much of the new representation the old one explains.

Features are standardised per dimension with fit-set statistics; the ridge strength is chosen on an inner
split of the fit set from a fixed grid. All variants share the same samples, splits, token positions and grid.

    PYTHONPATH=<awesome_starvla>/code:<StarVLA> python scripts/cross_head_probe.py \
        --config code/starvla_lab/configs/f0_libero_goal_smoke.yaml \
        --variant oft=<run>/final_model/pytorch_model.pt --variant multihead=... \
        --data_mix "dirA:robot,dirB:robot" --n_samples 1024 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "code") not in sys.path:
    sys.path.insert(0, str(REPO / "code"))

from starvla_lab.data.mixtures import parse_mixture_spec, register_mixture  # noqa: E402
from starvla_lab.probes.action_probe import DEFAULT_RIDGE_GRID, fit_ridge_probe_cv, split_indices_by_group  # noqa: E402
from starvla_lab.probes.qwen_extract import QWEN_IMAGE_TOKEN_ID, stratified_probe_batch  # noqa: E402

RIDGE_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ridge(X_fit, Y_fit, X_eval, Y_eval, seed: int) -> Dict[str, float]:
    return fit_ridge_probe_cv(X_fit, Y_fit, X_eval, Y_eval, lambdas=DEFAULT_RIDGE_GRID, seed=seed, device=RIDGE_DEVICE)


# ---------------------------------------------------------------------------------------------------------------- data
def registries() -> List[dict]:
    out = []
    for module in ("starVLA.dataloader.gr00t_lerobot.mixtures", "starVLA.dataloader.gr00t_lerobot.registry"):
        try:
            mod = __import__(module, fromlist=["DATASET_NAMED_MIXTURES"])
        except ImportError:
            continue
        reg = getattr(mod, "DATASET_NAMED_MIXTURES", None)
        if isinstance(reg, dict) and all(reg is not r for r in out):
            out.append(reg)
    return out


def load_samples(cfg, data_mix: str, n: int, pool_factor: int, out_dir: Path, seed: int) -> List[dict]:
    from omegaconf import OmegaConf
    from starVLA.dataloader import build_dataloader

    probe_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    name = None
    for reg in registries():
        name = register_mixture(data_mix, reg)
    probe_cfg.datasets.vla_data.data_mix = name
    probe_cfg.datasets.vla_data.per_device_batch_size = 16
    probe_cfg.datasets.vla_data.num_workers = 4
    probe_cfg.seed = seed
    probe_cfg.output_dir = str(out_dir / "_data")
    (out_dir / "_data").mkdir(parents=True, exist_ok=True)
    loader = build_dataloader(cfg=probe_cfg, dataset_py="lerobot_datasets")
    pool, it = [], iter(loader)
    while len(pool) < n * pool_factor:
        pool.extend(next(it))
    return stratified_probe_batch(pool, n, key="lang")


def mixture_entries(data_mix: str) -> list:
    for reg in registries():
        if data_mix in reg:
            return list(reg[data_mix])
    return parse_mixture_spec(data_mix)


def suite_lookup(cfg, data_mix: str) -> Dict[str, str]:
    """instruction -> short suite name, from meta/tasks.jsonl of every dataset in the mixture."""
    root = Path(cfg.datasets.vla_data.data_root_dir)
    mapping: Dict[str, str] = {}
    for dataset_dir, _, _ in mixture_entries(data_mix):
        short = dataset_dir.replace("_no_noops_1.0.0_lerobot", "").replace("libero_", "")
        tasks = root / dataset_dir / "meta" / "tasks.jsonl"
        if tasks.exists():
            for line in tasks.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    mapping[json.loads(line)["task"]] = short
    return mapping


# ------------------------------------------------------------------------------------------------------------ features
@torch.no_grad()
def extract_features(vlm, samples: Sequence[dict], layers: Sequence[int], token_layers: Sequence[int],
                     tokens_per_sample: int, batch_size: int, token_positions: Optional[List[torch.Tensor]] = None):
    """Pooled features ``{layer: {"all": [N, d], "image": [N, d]}}`` (fp32, cpu) and token features
    ``{layer: [N, T, d]}`` (fp16, cpu) at ``T`` fixed positions per sample (chosen on the first call)."""
    device = vlm.model.device
    pooled = {l: {"all": [], "image": []} for l in layers}
    tokens = {l: [] for l in token_layers}
    positions: List[torch.Tensor] = [] if token_positions is None else token_positions
    for lo in range(0, len(samples), batch_size):
        batch = samples[lo : lo + batch_size]
        inputs = vlm.build_qwenvl_inputs(images=[ex["image"] for ex in batch], instructions=[ex["lang"] for ex in batch])
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        out = vlm.model(**inputs, output_hidden_states=True, return_dict=True)
        valid = inputs["attention_mask"].bool()
        image = valid & (inputs["input_ids"] == QWEN_IMAGE_TOKEN_ID)
        for l in layers:
            h = out.hidden_states[l + 1].float()
            for name, m in (("all", valid), ("image", image)):
                mf = m.unsqueeze(-1).float()
                pooled[l][name].append(((h * mf).sum(1) / mf.sum(1).clamp(min=1.0)).cpu())
        for i in range(len(batch)):
            if token_positions is None:
                idx = valid[i].nonzero(as_tuple=False).flatten()
                pick = torch.linspace(0, idx.numel() - 1, tokens_per_sample, device=idx.device).round().long()
                positions.append(idx[pick].cpu())
            pos = positions[lo + i].to(device)
            for l in token_layers:
                tokens[l].append(out.hidden_states[l + 1][i, pos].to(torch.float16).cpu())
    pooled_t = {l: {k: torch.cat(v) for k, v in d.items()} for l, d in pooled.items()}
    tokens_t = {l: torch.stack(v) for l, v in tokens.items()}
    return pooled_t, tokens_t, positions


def load_backbone(vlm, checkpoint: Path) -> int:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = "qwen_vl_interface."
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = vlm.load_state_dict(sub, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected backbone keys: {unexpected[:5]}")
    return len(sub)


# -------------------------------------------------------------------------------------------------------------- probes
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="F0 yaml (base_vlm, data_root_dir, action config)")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=CHECKPOINT",
                    help="fine-tuned StarVLA final_model/pytorch_model.pt; the pretrained VLM is always included as 'pretrained'")
    ap.add_argument("--data_mix", required=True, help="StarVLA mixture name or inline dir:robot[,...] spec for the probe samples")
    ap.add_argument("--data_root_dir", default=None, help="override datasets.vla_data.data_root_dir")
    ap.add_argument("--n_samples", type=int, default=1024)
    ap.add_argument("--pool_factor", type=int, default=2)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--layers", type=str, default="17,26,30,33,34,35")
    ap.add_argument("--token_layers", type=str, default="34,35")
    ap.add_argument("--tokens_per_sample", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from starVLA.model.modules.vlm import get_vlm_model

    cfg = OmegaConf.load(args.config)
    if args.data_root_dir:
        cfg.datasets.vla_data.data_root_dir = args.data_root_dir
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    token_layers = [int(x) for x in args.token_layers.split(",")]
    variants = [v.split("=", 1) for v in args.variant]

    t0 = time.time()
    samples = load_samples(cfg, args.data_mix, args.n_samples, args.pool_factor, out, args.seed)
    lookup = suite_lookup(cfg, args.data_mix)
    suites = [lookup.get(ex["lang"], "unknown") for ex in samples]
    actions = torch.stack([torch.as_tensor(ex["action"], dtype=torch.float32) for ex in samples])  # [N, K, D]
    counts = {s: suites.count(s) for s in sorted(set(suites))}
    print(f"[probe] {len(samples)} samples, {len({ex['lang'] for ex in samples})} instructions, suites {counts}, "
          f"actions {tuple(actions.shape)} ({time.time() - t0:.0f}s)")
    splits = split_indices_by_group(suites, args.holdout, args.seed)

    device = torch.device("cuda:0")
    vlm = get_vlm_model(config=cfg).to(device).eval()
    feats: Dict[str, tuple] = {}
    t1 = time.time()
    pooled, tokens, positions = extract_features(vlm, samples, layers, token_layers, args.tokens_per_sample, args.batch_size)
    feats["pretrained"] = (pooled, tokens)
    print(f"[probe] pretrained features in {time.time() - t1:.0f}s; token positions per sample {args.tokens_per_sample}")
    for name, ckpt in variants:
        t1 = time.time()
        n_loaded = load_backbone(vlm, Path(ckpt))
        pooled, tokens, _ = extract_features(vlm, samples, layers, token_layers, args.tokens_per_sample, args.batch_size, positions)
        feats[name] = (pooled, tokens)
        print(f"[probe] {name}: {n_loaded} backbone tensors loaded, features in {time.time() - t1:.0f}s")

    report = {
        "config": vars(args) | {"n_samples_actual": len(samples), "suites": counts, "action_shape": list(actions.shape[1:]),
                                "layers": layers, "token_layers": token_layers, "lambdas": list(DEFAULT_RIDGE_GRID)},
        "action_readability": {}, "retention": {},
    }
    # 1. action readability: within-suite and cross-suite
    suite_names = sorted(splits)
    for name, (pooled, _) in feats.items():
        report["action_readability"][name] = {}
        for l in layers:
            for pool in ("all", "image"):
                H = pooled[l][pool]
                entry = {}
                for s in suite_names:
                    fit_i, eval_i = splits[s]
                    entry[f"{s}->{s}"] = ridge(H[fit_i], actions[fit_i], H[eval_i], actions[eval_i], args.seed)
                if len(suite_names) >= 2:
                    for s_fit in suite_names:
                        for s_eval in suite_names:
                            if s_fit == s_eval:
                                continue
                            fit_i = torch.cat(splits[s_fit])  # all samples of the fit suite
                            eval_i = torch.cat(splits[s_eval])
                            entry[f"{s_fit}->{s_eval}"] = ridge(H[fit_i], actions[fit_i], H[eval_i], actions[eval_i], args.seed)
                report["action_readability"][name][f"L{l}/{pool}"] = entry
    # 2. retention: fine-tuned tokens -> pretrained tokens (and reverse), split by sample over all suites
    all_fit = torch.cat([splits[s][0] for s in suite_names])
    all_eval = torch.cat([splits[s][1] for s in suite_names])
    pre_tokens = feats["pretrained"][1]
    for name, (_, tokens) in feats.items():
        report["retention"][name] = {}
        for l in token_layers:
            X, Y = tokens[l].float(), pre_tokens[l].float()  # [N, T, d]
            xf, xe = X[all_fit].flatten(0, 1), X[all_eval].flatten(0, 1)
            yf, ye = Y[all_fit].flatten(0, 1), Y[all_eval].flatten(0, 1)
            report["retention"][name][f"L{l}"] = {
                "finetuned_to_pretrained": ridge(xf, yf, xe, ye, args.seed),
                "pretrained_to_finetuned": ridge(yf, xf, ye, xe, args.seed),
            }

    (out / "cross_head_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # summary tables
    lines = ["| variant | " + " | ".join(f"L{l} {k}" for l in layers for k in [f"{s}->{s}" for s in suite_names]) + " |"]
    lines.append("|---|" + "---:|" * (len(layers) * len(suite_names)))
    for name in feats:
        cells = [f"{report['action_readability'][name][f'L{l}/all'][f'{s}->{s}']['r2']:.3f}" for l in layers for s in suite_names]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    print("\naction readability, held-out R^2 (all-token pool, within suite):\n" + "\n".join(lines))
    if len(suite_names) >= 2:
        lines = ["| variant | " + " | ".join(f"L{l} {a}->{b}" for l in layers for a in suite_names for b in suite_names if a != b) + " |"]
        lines.append("|---|" + "---:|" * (len(layers) * len(suite_names) * (len(suite_names) - 1)))
        for name in feats:
            cells = [f"{report['action_readability'][name][f'L{l}/all'][f'{a}->{b}']['r2']:.3f}" for l in layers for a in suite_names for b in suite_names if a != b]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        print("\naction readability, cross-suite R^2 (all-token pool):\n" + "\n".join(lines))
    lines = ["| variant | " + " | ".join(f"L{l} ft->pre | L{l} pre->ft" for l in token_layers) + " |", "|---|" + "---:|" * (2 * len(token_layers))]
    for name in feats:
        cells = []
        for l in token_layers:
            r = report["retention"][name][f"L{l}"]
            cells += [f"{r['finetuned_to_pretrained']['r2']:.4f}", f"{r['pretrained_to_finetuned']['r2']:.4f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    print("\nretention, held-out token-level R^2:\n" + "\n".join(lines))
    print(f"\n[done] wrote {out / 'cross_head_probe.json'} ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
