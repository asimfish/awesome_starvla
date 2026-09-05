"""StarVLA training entry with the lab hooks attached (WP1/WP2/WP4/WP6 integration, F1 data fractions).

Mirrors ``starVLA/training/train_starvla.py::main`` (single VLA loader) or
``train_starvla_cotrain.py::main`` (VLA + VLM loaders) and changes three things only:

* ``datasets.vla_data.data_fraction < 1`` wraps every LeRobot dataset in a seeded ``TrajectorySubset``;
* ``trainer.lab.llrd.enabled`` builds layer-wise decayed parameter groups instead of StarVLA's default;
* ``VLA(M)Trainer._train_step`` is wrapped with :class:`starvla_lab.train.LabHooks`.

``trainer.lab.mode`` selects the base script: ``single`` / ``cotrain`` / ``auto`` (cotrain when
``datasets.vlm_data`` is configured). Launch as a module from the StarVLA repo root with this package on
``PYTHONPATH``::

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

    Uses the framework's own preprocessing so the probe sees the training token layout; returns one
    ``[N, d]`` tensor per decoder layer (embedding output excluded).
    """
    fw = unwrap_model(model)
    instructions = fw._prepare_instructions(batch)
    images = [ex["image"] for ex in batch]
    inputs = fw.qwen_vl_interface.build_qwenvl_inputs(images, instructions)
    device = fw.qwen_vl_interface.model.device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = fw.qwen_vl_interface.model(**inputs, output_hidden_states=True, return_dict=True)
    mask = inputs["attention_mask"].unsqueeze(-1).to(out.hidden_states[0].dtype)
    return [((h * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu() for h in out.hidden_states[1:]]


def select_mode(cfg: Any) -> str:
    mode = str(cfg_get(cfg, "trainer.lab.mode", "auto")).lower()
    if mode == "auto":
        return "cotrain" if cfg_get(cfg, "datasets.vlm_data", None) is not None else "single"
    if mode not in ("single", "cotrain"):
        raise ValueError(f"trainer.lab.mode must be auto|single|cotrain, got {mode!r}")
    return mode


def main(cfg: Any) -> None:
    mode = select_mode(cfg)
    if mode == "cotrain":
        from starVLA.training import train_starvla_cotrain as base  # StarVLA runtime only
    else:
        from starVLA.training import train_starvla as base

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
        probe_batch = next(iter(vla_loader))[: lab.probes.probe_batch_size] if lab.probes.enabled else None
        hooks = LabHooks(trainer, lab, extract_fn=qwen_layer_extract_fn if lab.probes.enabled else None, probe_batch=probe_batch)
        attach_to_trainer(trainer, hooks)
    trainer.train()


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from starVLA.training.train_starvla_cotrain import apply_config_compat, normalize_dotlist_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(normalize_dotlist_args(clipargs)))
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    main(cfg)
