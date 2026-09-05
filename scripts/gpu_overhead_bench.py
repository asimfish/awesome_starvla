#!/usr/bin/env python3
"""WP6 on a real GPU: training-step cost of one vs three action heads on a real Qwen3-VL backbone.

Builds ``vlact_ext.Qwen_MultiHead`` through its *production* constructor (``get_vlm_model`` loads the
Qwen3-VL weights, the StarVLA head factories build full-size heads: OFT MLP, GR00T DiT-B x16, PI
layer-wise DiT x36) and measures, per head configuration, forward+backward seconds/step, samples/s
and peak allocated memory, plus ``predict_action`` latency per head. Samples are synthetic (random
224x224 images, random 16x7 action chunks, a 7-D state injected as text), so no dataset is needed.

Configurations (``--configs``):
    oft / gr00t / pi   single head   (QwenOFT / QwenGR00T / QwenPI_v3 equivalents)
    three              all three heads every step (VLAct recipe (c))
    three_dropout      three heads, one active per step (``active_heads`` rotation, p_all = 0)
    three_masked       three heads, ``mask_oft_queries_for_fm_heads = true`` (R2+ setting)
``three*`` share one loaded model; single-head configs each load the backbone once.

Freeze (``--freeze``): ``vlact`` = visual encoder + LLM layers below 18 frozen (VLAct recipe (a),
``freeze_rules`` syntax); ``none`` = full fine-tuning. No optimizer step is taken, so the numbers
exclude optimizer-state memory (identical across head configurations).

Run (one GPU, ~10-15 min for the default matrix)::

    PYTHONPATH=<awesome_starvla>/code:<StarVLA> python scripts/gpu_overhead_bench.py \
        --base_vlm /path/to/Qwen3-VL-4B-Instruct --out results/overhead_$(date +%Y%m%d_%H%M)
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "code") not in sys.path:
    sys.path.insert(0, str(REPO / "code"))

from starvla_lab.bench.overhead_bench import OverheadResult, measure_step_overhead, write_overhead_csv  # noqa: E402
from vlact_ext.freeze_rules import freeze_by_rules  # noqa: E402
from vlact_ext.multihead_framework import HEAD_NAMES, Qwen_MultiHead  # noqa: E402

VLACT_FREEZE = "qwen_vl_interface.model.model.visual,llm_layers_below:18"


def build_config(base_vlm: str, heads: List[str], args, mask_queries: bool = False):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "framework": {
                "name": "QwenMultiHead",
                "qwenvl": {"base_vlm": base_vlm},
                "action_model": {
                    "action_dim": args.action_dim,
                    "state_dim": 0,
                    "action_horizon": args.horizon,
                    "repeated_diffusion_steps": args.repeated_diffusion_steps,
                    "num_inference_timesteps": args.num_inference_timesteps,
                },
                "heads": {name: {"enabled": name in heads, "loss_weight": 1.0} for name in HEAD_NAMES},
                "predict_head": heads[0],
                "mask_oft_queries_for_fm_heads": mask_queries,
                # keep the bench about head cost: no wrap term, native 7-D actions
                "wrap_aware": {"enabled": False},
                "unified_layout": {"enabled": False},
            },
            "datasets": {"vla_data": {"obs_image_size": [args.image_size, args.image_size]}},
            "trainer": {},
        }
    )


def make_batch(n: int, args, seed: int = 0) -> List[dict]:
    rng = np.random.default_rng(seed)
    batch = []
    for i in range(n):
        img = Image.fromarray(rng.integers(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8))
        batch.append(
            {
                "action": rng.uniform(-1, 1, size=(args.horizon, args.action_dim)).astype(np.float32),
                "image": [img],
                "lang": f"put the {['red', 'blue', 'green', 'yellow'][i % 4]} block into the bowl",
                "state": rng.uniform(-1, 1, size=(1, args.action_dim)).astype(np.float32),
            }
        )
    return batch


def count_params(model: Qwen_MultiHead) -> Dict[str, int]:
    out = {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    out["backbone"] = sum(p.numel() for p in model.qwen_vl_interface.parameters())
    for name, head in model.heads.items():
        out[f"head_{name}"] = sum(p.numel() for p in head.parameters())
    out["project_layers"] = sum(p.numel() for p in model.project_layers.parameters())
    return out


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_model(base_vlm: str, heads: List[str], args, device: torch.device, mask_queries: bool = False):
    _empty_cache(device)
    t0 = time.perf_counter()
    model = Qwen_MultiHead(build_config(base_vlm, heads, args, mask_queries))
    model.to(device)
    model.train()
    frozen = 0
    if args.freeze == "vlact":
        frozen = freeze_by_rules(model, VLACT_FREEZE).num_frozen
    load_s = time.perf_counter() - t0
    params = count_params(model)
    print(
        f"[load] heads={heads} in {load_s:.0f}s | params total {params['total'] / 1e9:.2f}B, trainable {params['trainable'] / 1e9:.2f}B "
        f"({frozen} tensors frozen) | heads: " + ", ".join(f"{k[5:]}={v / 1e6:.0f}M" for k, v in params.items() if k.startswith("head_")),
        flush=True,
    )
    return model, params, load_s


def train_step_fn(model: Qwen_MultiHead, batch: List[dict], rotate_heads: bool = False):
    heads = list(model.heads.keys())

    def step(i: int) -> torch.Tensor:
        model.active_heads = [heads[i % len(heads)]] if rotate_heads else None
        model.zero_grad(set_to_none=True)
        return model(batch)["action_loss"]

    return step


@torch.inference_mode()
def predict_latency(model: Qwen_MultiHead, example: dict, device: torch.device, reps: int = 5) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for head in model.heads:
        model.predict_action(examples=[example], head=head)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            model.predict_action(examples=[example], head=head)
        _sync(device)
        out[head] = round((time.perf_counter() - t0) / reps, 4)
    model.train()
    return out


def env_info(args) -> dict:
    def sha(path: Path) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            return "unknown"

    import transformers

    cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    info = {
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
        "gpu_total_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if cuda else 0.0,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "awesome_starvla": sha(REPO),
        "args": vars(args),
    }
    try:
        import starVLA

        info["starVLA"] = sha(Path(starVLA.__file__).resolve().parents[1])
    except Exception:
        pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_vlm", required=True)
    ap.add_argument("--out", required=True, help="output directory (overhead.csv, results.json)")
    ap.add_argument("--configs", default="oft,gr00t,pi,three,three_dropout,three_masked")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--freeze", choices=["vlact", "none"], default="vlact")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--action_dim", type=int, default=7)
    ap.add_argument("--repeated_diffusion_steps", type=int, default=4, help="StarVLA default for the FM heads")
    ap.add_argument("--num_inference_timesteps", type=int, default=4)
    ap.add_argument("--predict_reps", type=int, default=5)
    ap.add_argument("--device", default="cuda", help="cuda (numbers) or cpu (pipeline dry run with a tiny checkpoint, no memory numbers)")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; pass --device cpu for a dry run", file=sys.stderr)
        return 2
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = env_info(args)
    print(f"[env] {info['gpu']} ({info['gpu_total_mem_gb']} GB) torch {info['torch']} transformers {info['transformers']} "
          f"starVLA {info.get('starVLA')} awesome_starvla {info['awesome_starvla']}", flush=True)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    batch = make_batch(args.batch_size, args)
    results: List[OverheadResult] = []
    extra: Dict[str, dict] = {}

    single = [c for c in configs if c in HEAD_NAMES]
    for head in single:
        model, params, load_s = load_model(args.base_vlm, [head], args, device)
        res = measure_step_overhead(head, train_step_fn(model, batch), args.batch_size, args.steps, args.warmup, device)
        results.append(res)
        extra[head] = {"params": params, "load_s": load_s, "predict_s": predict_latency(model, batch[0], device, args.predict_reps)}
        print(f"[bench] {res.name:14s} {res.sec_per_step:.3f} s/step  {res.samples_per_sec:.2f} samples/s  peak {res.peak_mem_mb / 1024:.1f} GB  "
              f"predict {extra[head]['predict_s']}", flush=True)
        del model
        _empty_cache(device)

    three = [c for c in configs if c.startswith("three")]
    if three:
        model, params, load_s = load_model(args.base_vlm, list(HEAD_NAMES), args, device)
        for name in three:
            model.mask_oft_queries_for_fm_heads = name == "three_masked"
            res = measure_step_overhead(name, train_step_fn(model, batch, rotate_heads=(name == "three_dropout")), args.batch_size, args.steps, args.warmup, device)
            results.append(res)
            extra[name] = {"params": params, "load_s": load_s}
            print(f"[bench] {res.name:14s} {res.sec_per_step:.3f} s/step  {res.samples_per_sec:.2f} samples/s  peak {res.peak_mem_mb / 1024:.1f} GB", flush=True)
        model.mask_oft_queries_for_fm_heads = False
        model.active_heads = None
        extra[three[0]]["predict_s"] = predict_latency(model, batch[0], device, args.predict_reps)
        print(f"[bench] three-head predict_action latency per head: {extra[three[0]]['predict_s']}", flush=True)
        del model
        _empty_cache(device)

    write_overhead_csv(results, out_dir / "overhead.csv")
    (out_dir / "results.json").write_text(json.dumps({"env": info, "results": [r.to_row() for r in results], "extra": extra}, indent=2), encoding="utf-8")

    base = next((r for r in results if r.name == "oft"), results[0])
    print("\n| config | s/step | samples/s | peak GB | vs oft (time) |\n|---|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r.name} | {r.sec_per_step:.3f} | {r.samples_per_sec:.2f} | {r.peak_mem_mb / 1024:.1f} | {r.sec_per_step / base.sec_per_step:.2f}x |")
    print(f"\n[done] wrote {out_dir / 'overhead.csv'} and results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
