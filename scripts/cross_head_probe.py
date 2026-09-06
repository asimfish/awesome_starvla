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
import gc
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


def load_samples(cfg, data_mix: str, n: int, pool_factor: int, out_dir: Path, seed: int, num_workers: int = 4) -> List[dict]:
    from omegaconf import OmegaConf
    from starVLA.dataloader import build_dataloader

    probe_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    name = None
    for reg in registries():
        name = register_mixture(data_mix, reg)
    probe_cfg.datasets.vla_data.data_mix = name
    probe_cfg.datasets.vla_data.per_device_batch_size = 16
    probe_cfg.datasets.vla_data.num_workers = int(num_workers)  # 0 = decode in-process (nodes where forked workers cannot allocate)
    probe_cfg.seed = seed
    probe_cfg.output_dir = str(out_dir / "_data")
    (out_dir / "_data").mkdir(parents=True, exist_ok=True)
    loader = build_dataloader(cfg=probe_cfg, dataset_py="lerobot_datasets")
    pool, it, failures = [], iter(loader), 0
    while len(pool) < n * pool_factor:
        try:
            pool.extend(next(it))
        except StopIteration:
            it = iter(loader)
        except Exception as exc:  # PyAV ENOMEM on memory-starved nodes: release decoders, back off, keep going
            failures += 1
            if failures > 20:
                raise
            print(f"[probe] loader error ({type(exc).__name__}: {str(exc)[:80]}); retry {failures}/20 after gc", flush=True)
            gc.collect()
            time.sleep(5)
            continue
        if len(pool) % 256 < 16:
            gc.collect()  # PyAV containers held by decoded samples' frames are released promptly
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


QUERY_TOKEN = "🔍"  # StarVLA QwenOFT's action query token (one existing vocab entry, repeated chunk_len times)


def oft_prompt(instruction: str, k: int) -> str:
    """The exact prompt QwenOFT trains with: instruction + a suffix carrying ``k`` query tokens."""
    return f"{instruction} Please predict the next {k} robot actions: <action>{QUERY_TOKEN * k}<action>."


@torch.no_grad()
def extract_query_features(vlm, samples: Sequence[dict], layers: Sequence[int], k: int, batch_size: int) -> Dict[int, torch.Tensor]:
    """Hidden states at the ``k`` OFT action-query positions, ``{layer: [N, k, d]}`` (fp32, cpu).

    This is the pre-registered H1 feature: what QwenOFT's MLP head reads (layer 35). The pretrained VLM has never
    seen the suffix, so its query states are those of an emoji attending to image + instruction -- the baseline.
    """
    token_id = vlm.processor.tokenizer(QUERY_TOKEN, add_special_tokens=False)["input_ids"][0]
    device = vlm.model.device
    feats = {l: [] for l in layers}
    for lo in range(0, len(samples), batch_size):
        batch = samples[lo : lo + batch_size]
        inputs = vlm.build_qwenvl_inputs(images=[ex["image"] for ex in batch], instructions=[oft_prompt(ex["lang"], k) for ex in batch])
        inputs = {kk: (v.to(device) if hasattr(v, "to") else v) for kk, v in inputs.items()}
        out = vlm.model(**inputs, output_hidden_states=True, return_dict=True)
        mask = inputs["input_ids"] == token_id
        counts = mask.sum(1)
        if not bool((counts == k).all()):
            raise RuntimeError(f"expected {k} query tokens per sample, got {counts.tolist()}")
        for l in layers:
            feats[l].append(out.hidden_states[l + 1].float()[mask].view(len(batch), k, -1).cpu())
    return {l: torch.cat(v) for l, v in feats.items()}


def per_position_probe(H: torch.Tensor, A: torch.Tensor, fit_i, eval_i, seed: int) -> Dict[str, object]:
    """One ridge probe per query position k (query state k -> action step k), as the OFT head is wired; mean R^2 over k."""
    per_k = [ridge(H[fit_i, k], A[fit_i, k], H[eval_i, k], A[eval_i, k], seed) for k in range(H.shape[1])]
    return {"r2": sum(r["r2"] for r in per_k) / len(per_k), "mae_std": sum(r["mae_std"] for r in per_k) / len(per_k),
            "per_position_r2": [round(r["r2"], 4) for r in per_k], "lambdas": [r["lambda"] for r in per_k],
            "n_fit": per_k[0]["n_fit"], "n_eval": per_k[0]["n_eval"]}


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
    ap.add_argument("--query_layers", type=str, default="", help="layers for the OFT query-position probe (H1 primary metric), e.g. 33,34,35; empty = skip")
    ap.add_argument("--no_pooled", action="store_true", help="skip the pooled-feature action probe")
    ap.add_argument("--no_retention", action="store_true", help="skip the token-level retention probe")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for the sample pool (0 = in-process)")
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
    samples = load_samples(cfg, args.data_mix, args.n_samples, args.pool_factor, out, args.seed, args.num_workers)
    lookup = suite_lookup(cfg, args.data_mix)
    suites = [lookup.get(ex["lang"], "unknown") for ex in samples]
    actions = torch.stack([torch.as_tensor(ex["action"], dtype=torch.float32) for ex in samples])  # [N, K, D]
    counts = {s: suites.count(s) for s in sorted(set(suites))}
    print(f"[probe] {len(samples)} samples, {len({ex['lang'] for ex in samples})} instructions, suites {counts}, "
          f"actions {tuple(actions.shape)} ({time.time() - t0:.0f}s)")
    splits = split_indices_by_group(suites, args.holdout, args.seed)

    device = torch.device("cuda:0")
    vlm = get_vlm_model(config=cfg).to(device).eval()
    query_layers = [int(x) for x in args.query_layers.split(",") if x.strip()]
    k_queries = int(actions.shape[1])
    do_pooled, do_retention, do_query = not args.no_pooled, not args.no_retention, bool(query_layers)
    need_tokens = do_pooled or do_retention

    feats: Dict[str, tuple] = {}
    qfeats: Dict[str, Dict[int, torch.Tensor]] = {}
    positions = None

    def extract_all(name: str) -> None:
        nonlocal positions
        t1 = time.time()
        if need_tokens:
            pooled, tokens, pos = extract_features(vlm, samples, layers, token_layers, args.tokens_per_sample, args.batch_size, positions)
            positions = pos
            feats[name] = (pooled, tokens)
        if do_query:
            qfeats[name] = extract_query_features(vlm, samples, query_layers, k_queries, args.batch_size)
        print(f"[probe] {name}: features in {time.time() - t1:.0f}s")

    extract_all("pretrained")
    for name, ckpt in variants:
        n_loaded = load_backbone(vlm, Path(ckpt))
        print(f"[probe] {name}: {n_loaded} backbone tensors loaded")
        extract_all(name)
    names = ["pretrained"] + [n for n, _ in variants]

    report = {
        "config": vars(args) | {"n_samples_actual": len(samples), "suites": counts, "action_shape": list(actions.shape[1:]),
                                "layers": layers, "token_layers": token_layers, "query_layers": query_layers, "lambdas": list(DEFAULT_RIDGE_GRID)},
        "action_readability": {}, "retention": {}, "query_readability": {},
    }
    suite_names = sorted(splits)

    def suite_entries(H, probe_fn):
        entry = {}
        for s_ in suite_names:
            fit_i, eval_i = splits[s_]
            entry[f"{s_}->{s_}"] = probe_fn(H, fit_i, eval_i)
        for s_fit in suite_names:
            for s_eval in suite_names:
                if s_fit != s_eval:
                    entry[f"{s_fit}->{s_eval}"] = probe_fn(H, torch.cat(splits[s_fit]), torch.cat(splits[s_eval]))
        return entry

    # 1. action readability from pooled features: within-suite and cross-suite
    if do_pooled:
        for name in names:
            pooled = feats[name][0]
            report["action_readability"][name] = {}
            for l in layers:
                for pool in ("all", "image"):
                    H = pooled[l][pool]
                    report["action_readability"][name][f"L{l}/{pool}"] = suite_entries(
                        H, lambda H_, fi, ei: ridge(H_[fi], actions[fi], H_[ei], actions[ei], args.seed))
    # 2. retention: fine-tuned tokens -> pretrained tokens (and reverse), split by sample over all suites
    if do_retention:
        all_fit = torch.cat([splits[s_][0] for s_ in suite_names])
        all_eval = torch.cat([splits[s_][1] for s_ in suite_names])
        pre_tokens = feats["pretrained"][1]
        for name in names:
            tokens = feats[name][1]
            report["retention"][name] = {}
            for l in token_layers:
                X, Y = tokens[l].float(), pre_tokens[l].float()  # [N, T, d]
                xf, xe = X[all_fit].flatten(0, 1), X[all_eval].flatten(0, 1)
                yf, ye = Y[all_fit].flatten(0, 1), Y[all_eval].flatten(0, 1)
                report["retention"][name][f"L{l}"] = {
                    "finetuned_to_pretrained": ridge(xf, yf, xe, ye, args.seed),
                    "pretrained_to_finetuned": ridge(yf, xf, ye, xe, args.seed),
                }
    # 3. H1 primary metric: per-position ridge probe on the OFT query states (query k -> action step k)
    if do_query:
        for name in names:
            report["query_readability"][name] = {}
            for l in query_layers:
                report["query_readability"][name][f"L{l}"] = suite_entries(
                    qfeats[name][l], lambda H_, fi, ei: per_position_probe(H_, actions, fi, ei, args.seed))

    (out / "cross_head_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def table(title, section, keys, cell):
        pairs = [f"{a}->{b}" for a in suite_names for b in suite_names]
        lines = ["| variant | " + " | ".join(f"{k} {pr}" for k in keys for pr in pairs) + " |", "|---|" + "---:|" * (len(keys) * len(pairs))]
        for name in names:
            lines.append(f"| {name} | " + " | ".join(f"{cell(report[section][name][k][pr]):.3f}" for k in keys for pr in pairs) + " |")
        print(f"\n{title}\n" + "\n".join(lines))

    if do_pooled:
        table("action readability, held-out R^2 (all-token pool; within suite and cross-suite):", "action_readability",
              [f"L{l}/all" for l in layers], lambda r: r["r2"])
    if do_retention:
        lines = ["| variant | " + " | ".join(f"L{l} ft->pre | L{l} pre->ft" for l in token_layers) + " |", "|---|" + "---:|" * (2 * len(token_layers))]
        for name in names:
            cells = []
            for l in token_layers:
                r = report["retention"][name][f"L{l}"]
                cells += [f"{r['finetuned_to_pretrained']['r2']:.4f}", f"{r['pretrained_to_finetuned']['r2']:.4f}"]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        print("\nretention, held-out token-level R^2:\n" + "\n".join(lines))
    if do_query:
        table("H1 query-position readability, held-out R^2 (mean over the K query positions):", "query_readability",
              [f"L{l}" for l in query_layers], lambda r: r["r2"])
    print(f"\n[done] wrote {out / 'cross_head_probe.json'} ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
