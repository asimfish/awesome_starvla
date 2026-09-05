"""StarVLA training entry with the lab hooks attached (WP1/WP2/WP4/WP6 integration, F1 data fractions).

Mirrors ``starVLA/training/train_starvla.py::main`` (single VLA loader) or
``train_starvla_cotrain.py::main`` (VLA + VLM loaders) and changes three things only:

* ``datasets.vla_data.data_fraction < 1`` wraps every LeRobot dataset in a seeded ``TrajectorySubset``;
* ``trainer.lab.llrd.enabled`` builds layer-wise decayed parameter groups instead of StarVLA's default;
* ``VLA(M)Trainer._train_step`` is wrapped with :class:`starvla_lab.train.LabHooks`.

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
from typing import Any, List

import torch

from ..data.subsample import install_fraction_hook
from .integration import LabHooks, attach_to_trainer, build_optimizer_and_scheduler, unwrap_model
from .lab_config import LabConfig, cfg_get


def qwen_layer_extract_fn(model: torch.nn.Module, batch: List[dict]) -> List[torch.Tensor]:
    """Per-layer mean-pooled hidden states of a StarVLA Qwen-VL framework on raw samples.

    The probe prompt is the plain VLM prompt (images + instruction), identical for every framework and free
    of framework-specific learnable tokens: OFT's ``<action>🔍…`` query embeddings are trained from scratch
    and would otherwise dominate the pooled features of *frozen* layers. What is measured is therefore the
    drift of the VLM's own visual-language representation. Returns one ``[N, d]`` tensor per decoder layer
    (embedding output excluded).
    """
    fw = unwrap_model(model)
    instructions = [ex["lang"] for ex in batch]
    images = [ex["image"] for ex in batch]
    inputs = fw.qwen_vl_interface.build_qwenvl_inputs(images, instructions)
    device = fw.qwen_vl_interface.model.device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = fw.qwen_vl_interface.model(**inputs, output_hidden_states=True, return_dict=True)
    # Pool in fp32: the backbone emits bf16 hidden states, and a bf16 mean over ~200 tokens is rounded to ~3
    # significant digits, which would hide small drifts and add a noise floor to 1 - CKA.
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    return [((h.float() * mask).sum(1) / mask.sum(1).clamp(min=1.0)).cpu() for h in out.hidden_states[1:]]


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
    vla = base.build_framework(cfg)
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
    if mode == "cotrain":
        trainer = base.VLAMTrainer(cfg=cfg, model=vla, vla_train_dataloader=vla_loader, vlm_train_dataloader=vlm_loader,
                                   optimizer=optimizer, lr_scheduler=lr_scheduler, accelerator=base.accelerator)
    else:
        trainer = base.VLATrainer(cfg=cfg, model=vla, vla_train_dataloader=vla_loader,
                                  optimizer=optimizer, lr_scheduler=lr_scheduler, accelerator=base.accelerator)
    trainer.prepare_training()

    if lab.any_enabled():
        probe_batch = None
        if lab.probes.enabled:
            # StarVLA's LeRobot collate_fn returns the raw list of sample dicts; gather loader batches until the
            # requested probe size is reached (per-device batch is usually smaller than probe_batch_size).
            probe_batch, it = [], iter(vla_loader)
            while len(probe_batch) < lab.probes.probe_batch_size:
                probe_batch.extend(next(it))
            probe_batch = probe_batch[: lab.probes.probe_batch_size]
        hooks = LabHooks(trainer, lab, extract_fn=qwen_layer_extract_fn if lab.probes.enabled else None, probe_batch=probe_batch)
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
