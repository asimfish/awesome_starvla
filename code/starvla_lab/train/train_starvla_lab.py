"""StarVLA training entry with the lab hooks attached (WP1/WP2/WP4/WP6 integration, F1 data fractions).

Mirrors ``starVLA/training/train_starvla.py::main`` (single VLA loader) or
``train_starvla_cotrain.py::main`` (VLA + VLM loaders) and changes three things only:

* ``datasets.vla_data.data_fraction < 1`` wraps every LeRobot dataset in a seeded ``TrajectorySubset``;
* ``trainer.lab.llrd.enabled`` builds layer-wise decayed parameter groups instead of StarVLA's default;
* ``VLA(M)Trainer._train_step`` is wrapped with :class:`starvla_lab.train.LabHooks`; with ``trainer.lab.probes``
  the drift probe uses :class:`starvla_lab.probes.QwenBackboneProbe` (token-level CKA on the plain VLM prompt
  with the pretrained ``embed_tokens`` swapped in, pooled view as secondary) on a probe batch that is stratified
  over instructions and may come from a separate mixture (``probes.probe_data_mix``).

``trainer.lab.mode`` selects the base script: ``single`` / ``cotrain`` / ``auto`` (cotrain when
``datasets.vlm_data`` is configured). ``single`` honours StarVLA's ``STARVLA_DISABLE_DEEPSPEED=1`` for
one-GPU runs without DeepSpeed; ``cotrain`` imports ``train_starvla_cotrain`` whose module-level
``Accelerator(DeepSpeedPlugin())`` requires DeepSpeed. Launch as a module from the StarVLA repo root with
this package on ``PYTHONPATH``::

    PYTHONPATH=<awesome_starvla>/code accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \\
        --num_processes 16 -m starvla_lab.train.train_starvla_lab --config_yaml <yaml> --trainer.lab.mode cotrain \\
        --trainer.lab.llrd.enabled true --trainer.lab.probes.enabled true
"""
from __future__ import annotations

import argparse
import os
from typing import Any, List, Optional

import torch

from ..data.mixtures import register_mixture
from ..data.subsample import install_fraction_hook
from ..probes.qwen_extract import QwenBackboneProbe, framework_of, gather_probe_batch
from .integration import LabHooks, apply_backbone_fp32, attach_to_trainer, build_optimizer_and_scheduler
from .lab_config import LabConfig, ProbesConfig, cfg_get


def build_probe_extractors(probes: ProbesConfig) -> tuple[QwenBackboneProbe, Optional[QwenBackboneProbe]]:
    """Primary (drives the schedulers) and optional secondary (recorded only) extractors from the probe config."""
    kwargs = dict(token_subset=probes.token_subset, max_tokens=probes.max_tokens,
                  restore_pretrained_embeddings=probes.restore_pretrained_embeddings)
    primary = QwenBackboneProbe(representation=probes.representation, **kwargs)
    secondary = QwenBackboneProbe(representation=probes.secondary_representation, **kwargs) if probes.secondary_representation else None
    return primary, secondary


def mixture_registries() -> List[dict]:
    """Every dict StarVLA may resolve ``data_mix`` through. ``lerobot_datasets`` reads
    ``gr00t_lerobot.registry.DATASET_NAMED_MIXTURES``, which is a *copy* of ``gr00t_lerobot.mixtures``' dict made
    at import time, so an ad-hoc mixture has to go into the registry copy (and into the base dict for older
    StarVLA checkouts without ``registry.py``)."""
    out: List[dict] = []
    for module in ("starVLA.dataloader.gr00t_lerobot.mixtures", "starVLA.dataloader.gr00t_lerobot.registry"):
        try:
            mod = __import__(module, fromlist=["DATASET_NAMED_MIXTURES"])
        except ImportError:
            continue
        registry = getattr(mod, "DATASET_NAMED_MIXTURES", None)
        if isinstance(registry, dict) and all(registry is not r for r in out):
            out.append(registry)
    if not out:
        raise ImportError("StarVLA mixture registry not importable (starVLA.dataloader.gr00t_lerobot.{mixtures,registry})")
    return out


def resolve_inline_mixture(data_mix: Any, registries: List[dict]) -> str:
    """Return a registry name for ``data_mix``: unchanged when it is already a registered mixture name, otherwise the
    inline ``dataset_dir:robot_type[,...]`` spec is registered under a synthetic name in every registry."""
    spec = str(data_mix)
    if any(spec in reg for reg in registries):
        return spec
    name = spec
    for reg in registries:
        name = register_mixture(spec, reg)
    return name


def build_probe_loader(cfg: Any, probe_data_mix: str):
    """A separate LeRobot loader for the probe batch (``trainer.lab.probes.probe_data_mix``).

    ``probe_data_mix`` is a StarVLA mixture name or an inline ``dataset_dir:robot_type[,...]`` list (registered at
    runtime, see :mod:`starvla_lab.data.mixtures`). Everything else (action type, image size, root dir) is copied from
    the training data config. StarVLA's ``build_dataloader`` writes ``dataset_statistics.json`` into ``cfg.output_dir``,
    so the copy points at ``<run>/probe_data/`` to leave the training run's statistics untouched.
    """
    from omegaconf import OmegaConf
    from starVLA.dataloader import build_dataloader

    # StarVLA hands main() an AccessTrackedConfig; copy the underlying OmegaConf tree so the probe loader's
    # edits never touch the training config (nor its accessed-keys bookkeeping).
    base = cfg.unwrap() if hasattr(cfg, "unwrap") else cfg
    probe_cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    probe_cfg.datasets.vla_data.data_mix = resolve_inline_mixture(probe_data_mix, mixture_registries())
    probe_cfg.datasets.vla_data.num_workers = 0  # one-off loader: no worker pool next to the training loader
    probe_cfg.output_dir = os.path.join(str(cfg.output_dir), "probe_data")
    os.makedirs(probe_cfg.output_dir, exist_ok=True)
    return build_dataloader(cfg=probe_cfg, dataset_py="lerobot_datasets")


def build_probe_batch(cfg: Any, probes: ProbesConfig, train_loader: Any) -> List[dict]:
    loader = build_probe_loader(cfg, probes.probe_data_mix) if probes.probe_data_mix else train_loader
    batch = gather_probe_batch(loader, probes.probe_batch_size, stratify=probes.stratify_by_instruction, pool_factor=probes.pool_factor)
    n_instr = len({str(ex.get("lang", "")) for ex in batch})
    print(f"[starvla_lab] probe batch: {len(batch)} samples, {n_instr} distinct instructions, "
          f"source={'probe_data_mix=' + str(probes.probe_data_mix) if probes.probe_data_mix else 'training loader'}")
    return batch


def apply_gradient_accumulation(accelerator: Any, cfg: Any) -> int:
    """Make ``trainer.gradient_accumulation_steps`` effective on the no-DeepSpeed path.

    StarVLA constructs its module-level ``Accelerator`` without ``gradient_accumulation_steps``; with DeepSpeed the
    value comes from the DeepSpeed config, but with ``STARVLA_DISABLE_DEEPSPEED=1`` the accelerator keeps 1 and the
    yaml key only changes the printed batch size. Setting the accelerator's property makes ``accumulate()`` /
    ``sync_gradients`` behave (the trainer already advances ``completed_steps`` and the LR scheduler on sync only).
    """
    steps = int(cfg_get(cfg, "trainer.gradient_accumulation_steps", 1) or 1)
    if steps > 1 and int(getattr(accelerator, "gradient_accumulation_steps", 1)) != steps:
        accelerator.gradient_accumulation_steps = steps
        print(f"[starvla_lab] gradient accumulation: {steps} micro-steps per optimizer step (accelerator updated)")
    return int(getattr(accelerator, "gradient_accumulation_steps", 1))


def select_mode(cfg: Any) -> str:
    mode = str(cfg_get(cfg, "trainer.lab.mode", "auto")).lower()
    if mode == "auto":
        return "cotrain" if cfg_get(cfg, "datasets.vlm_data", None) is not None else "single"
    if mode not in ("single", "cotrain"):
        raise ValueError(f"trainer.lab.mode must be auto|single|cotrain, got {mode!r}")
    return mode


def register_extension_frameworks() -> None:
    """Make ``QwenMultiHead`` / ``QwenMultiHeadLab`` visible to StarVLA's ``build_framework`` without copying
    files into ``starVLA/model/framework/``: importing the modules registers them in ``FRAMEWORK_REGISTRY``."""
    for module in ("vlact_ext.multihead_framework", "starvla_lab.heads.register"):
        try:
            __import__(module)
        except ImportError as exc:  # vlact_ext not on PYTHONPATH: native frameworks still work
            print(f"[starvla_lab] {module} not importable ({exc}); only StarVLA-native frameworks are available")


def main(cfg: Any) -> None:
    mode = select_mode(cfg)
    if mode == "cotrain":
        from starVLA.training import train_starvla_cotrain as base  # StarVLA runtime only
    else:
        from starVLA.training import train_starvla as base
    register_extension_frameworks()

    cfg = base.wrap_config(cfg)
    lab = LabConfig.from_cfg(cfg)
    lab.validate()
    fraction = float(cfg_get(cfg, "datasets.vla_data.data_fraction", 1.0))
    install_fraction_hook(fraction, seed=int(cfg_get(cfg, "seed", 0)))

    output_dir = base.setup_directories(cfg=cfg)
    apply_gradient_accumulation(base.accelerator, cfg)
    vla = base.build_framework(cfg)
    # datasets.vla_data.data_mix may be an inline "dataset_dir:robot_type[,...]" spec (e.g. a LIBERO suite StarVLA
    # has no named mixture for); it is registered at runtime and replaced by the generated name.
    cfg.datasets.vla_data.data_mix = resolve_inline_mixture(cfg.datasets.vla_data.data_mix, mixture_registries())
    loaders = base.prepare_data(cfg=cfg, accelerator=base.accelerator, output_dir=output_dir)
    vla_loader, vlm_loader = (loaders if mode == "cotrain" else (loaders, None))

    def scheduler_factory(optimizer):
        from transformers import get_scheduler

        return get_scheduler(
            name=cfg.trainer.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=cfg.trainer.num_warmup_steps,
            num_training_steps=cfg.trainer.max_train_steps,
            scheduler_specific_kwargs=dict(cfg_get(cfg, "trainer.scheduler_specific_kwargs", {}) or {}),
        )

    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        vla, cfg, lab, scheduler_factory, fallback=lambda m, c: base.setup_optimizer_and_scheduler(model=m, cfg=c)
    )
    if lab.backbone_fp32:
        counts = apply_backbone_fp32(vla, cfg_get(cfg, "trainer.freeze_modules", "") or "")
        print(f"[starvla_lab] backbone_fp32: {counts['converted'] / 1e6:.0f}M backbone params upcast to fp32 "
              f"({counts['frozen'] / 1e6:.0f}M frozen left as loaded, {counts['already_fp32'] / 1e6:.0f}M already fp32); compute stays bf16 (autocast)")
    if lab.llrd.enabled:
        # The scheduler has already stepped once (warmup starts at 0), so report the base lr the scheduler scales, not g["lr"].
        def base_lr(g):
            return g.get("initial_lr", g["lr"])

        layer_groups = [g for g in optimizer.param_groups if "layer_index" in g]
        others = ", ".join(f"{g.get('name', '?')}={base_lr(g):.2e}" for g in optimizer.param_groups if "layer_index" not in g)
        span = (f"layer {layer_groups[0]['layer_index']} lr {base_lr(layer_groups[0]):.2e} ... "
                f"layer {layer_groups[-1]['layer_index']} lr {base_lr(layer_groups[-1]):.2e}") if layer_groups else "no trainable layers"
        print(f"[starvla_lab] LLRD groups (base lr): {len(layer_groups)} decoder layers ({span}); {others}")
    if mode == "cotrain":
        trainer = base.VLAMTrainer(cfg=cfg, model=vla, vla_train_dataloader=vla_loader, vlm_train_dataloader=vlm_loader,
                                   optimizer=optimizer, lr_scheduler=lr_scheduler, accelerator=base.accelerator)
    else:
        trainer = base.VLATrainer(cfg=cfg, model=vla, vla_train_dataloader=vla_loader,
                                  optimizer=optimizer, lr_scheduler=lr_scheduler, accelerator=base.accelerator)
    trainer.prepare_training()

    if lab.any_enabled():
        probe_batch, primary, secondary, compute_device = None, None, None, None
        if lab.probes.enabled:
            probe_batch = build_probe_batch(cfg, lab.probes, vla_loader)
            primary, secondary = build_probe_extractors(lab.probes)
            compute_device = framework_of(trainer.model).qwen_vl_interface.model.device
        hooks = LabHooks(trainer, lab, extract_fn=primary, probe_batch=probe_batch, secondary_extract_fn=secondary,
                         compute_device=compute_device)
        attach_to_trainer(trainer, hooks)
    trainer.train()

    import torch.distributed as dist  # same shutdown as StarVLA's own main()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    # Imported from where they are defined: importing train_starvla_cotrain here would construct its
    # module-level Accelerator(DeepSpeedPlugin()) and fail on single-GPU machines without DeepSpeed.
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(normalize_dotlist_args(clipargs)))
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    main(cfg)
