"""Wire probes, schedules and head dropout into a StarVLA-style trainer.

Two entry points:

* :func:`build_optimizer_and_scheduler` replaces StarVLA's ``setup_optimizer_and_scheduler`` when
  ``trainer.lab.llrd.enabled``: parameter groups come from
  :func:`starvla_lab.schedules.layerwise_lr_decay_groups` (frozen params excluded, layers tagged with
  ``layer_index``) and the lr scheduler is created by an injected factory (``transformers.get_scheduler``
  in StarVLA, any ``LambdaLR`` in tests).
* :class:`LabHooks` runs before/after every optimizer step: it writes the auxiliary-data scheduler outputs
  into the config (``trainer.loss_scale.vlm`` is read per step by StarVLA's ``_train_step``), picks the
  active heads for head dropout, and every ``probes.every_n_steps`` measures drift with a
  :class:`starvla_lab.probes.DriftTracker`, logs to JSONL and feeds the schedulers.
  :func:`attach_to_trainer` wraps ``trainer._train_step`` so the StarVLA loop stays untouched. Probe records
  are labelled with the number of completed optimizer updates (``step 0`` = reference, before training).

Everything here works on the minimal trainer surface ``config / model / optimizer / lr_scheduler /
completed_steps / _train_step`` so it can be tested with a mock trainer on CPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
from torch import nn

from ..bench.overhead_bench import HeadDropoutSchedule
from ..probes.drift import DriftTracker, ExtractFn
from ..probes.hooks import ProbeRunner, ProbeSchedule
from ..schedules.aux_scheduler import AuxDataScheduler
from ..schedules.llrd import DriftDrivenLLRD, layer_group_index, layerwise_lr_decay_groups
from .lab_config import LabConfig, cfg_get

__all__ = ["build_optimizer_and_scheduler", "LabHooks", "attach_to_trainer", "unwrap_model", "default_reference_reps", "completed_updates_after"]

SchedulerFactory = Callable[[torch.optim.Optimizer], Any]


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying framework behind DeepSpeed / DDP / accelerate wrappers."""
    for attr in ("module", "model"):
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module) and hasattr(inner, "heads"):
            return inner
    return model


def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: Any,
    lab: LabConfig,
    scheduler_factory: SchedulerFactory,
    fallback: Optional[Callable[[nn.Module, Any], Any]] = None,
    optimizer_cls: type = torch.optim.AdamW,
):
    """LLRD-aware replacement for StarVLA's ``setup_optimizer_and_scheduler``.

    With ``lab.llrd.enabled`` false the ``fallback`` (StarVLA's original function) is used unchanged.
    Optimizer hyper-parameters mirror StarVLA: ``trainer.optimizer.{betas, weight_decay, eps}`` when
    present (default weight decay 0.0); the backbone lr is ``trainer.learning_rate.qwen_vl_interface``
    (falling back to ``trainer.learning_rate.base``) and the head lr ``trainer.learning_rate.action_model``.
    """
    if not lab.llrd.enabled:
        if fallback is None:
            raise ValueError("lab.llrd is disabled and no fallback optimizer builder was given")
        return fallback(model, cfg)
    lr_node = cfg_get(cfg, "trainer.learning_rate", {})
    base_lr = float(cfg_get(lr_node, "qwen_vl_interface", cfg_get(lr_node, "base", 1e-5)))
    head_lr = float(lab.llrd.head_lr if lab.llrd.head_lr is not None else cfg_get(lr_node, "action_model", base_lr))
    groups = layerwise_lr_decay_groups(
        unwrap_model(model),
        base_lr=base_lr,
        decay=float(lab.llrd.decay),
        head_lr=head_lr,
        freeze_rules_spec=cfg_get(cfg, "trainer.freeze_modules", "") or "",
    )
    # StarVLA's setup_optimizer_and_scheduler reads trainer.optimizer.{betas, weight_decay, eps}; the flat
    # trainer.weight_decay form is kept as a fallback for hand-written configs.
    wd = cfg_get(cfg, "trainer.optimizer.weight_decay", cfg_get(cfg, "trainer.weight_decay", 0.0))
    kwargs: Dict[str, Any] = {"weight_decay": float(wd)}
    betas = cfg_get(cfg, "trainer.optimizer.betas", None)
    eps = cfg_get(cfg, "trainer.optimizer.eps", None)
    if betas is not None:
        kwargs["betas"] = tuple(float(b) for b in betas)
    if eps is not None:
        kwargs["eps"] = float(eps)
    optimizer = optimizer_cls(groups, lr=base_lr, **kwargs)
    return optimizer, scheduler_factory(optimizer)


def default_reference_reps(extract_fn: ExtractFn, model: nn.Module, batch: Any) -> List[torch.Tensor]:
    """Snapshot the per-layer representations of ``model`` on ``batch`` (call before training starts)."""
    with torch.no_grad():
        return [r.detach().to("cpu") for r in extract_fn(model, batch)]


@dataclass
class _Step:
    step: int
    drift: Optional[float] = None


class LabHooks:
    """Per-step controller: aux-data scheduler, head dropout, drift probes and drift-driven LLRD."""

    def __init__(
        self,
        trainer: Any,
        lab: LabConfig,
        *,
        extract_fn: Optional[ExtractFn] = None,
        probe_batch: Any = None,
        reference_reps: Optional[Sequence[torch.Tensor]] = None,
        layer_names: Optional[Sequence[str]] = None,
        vlm_capability_fn: Optional[Callable[[nn.Module], Mapping[str, float]]] = None,
        secondary_extract_fn: Optional[ExtractFn] = None,
        compute_device: Optional[Any] = None,
    ) -> None:
        """``extract_fn`` feeds the primary :class:`DriftTracker` (drives ``last_drift`` and the schedulers);
        ``secondary_extract_fn`` (e.g. the pooled view next to the token-level one) is only recorded, under
        ``drift_secondary``. ``compute_device`` is where the CKA Gram products run (default: where the
        representations are)."""
        lab.validate()
        self.trainer = trainer
        self.lab = lab
        self.model = unwrap_model(trainer.model)
        self.records: List[Dict[str, Any]] = []
        self.last_drift: Optional[float] = None
        # StarVLA's _train_step only forwards output["action_loss"] to the logs; capture the per-head /
        # auxiliary losses (loss_oft, loss_pi, loss_gr00t, loss_featpred, ...) from the framework's output dict.
        # StarVLA calls ``self.model.forward(batch)`` directly, which bypasses nn.Module hooks, so the bound
        # ``forward`` is wrapped on the instance instead (also covers ``model(batch)`` and DDP wrappers).
        self.last_losses: Dict[str, float] = {}
        self._wrap_forward(self.model)

        self.aux: Optional[AuxDataScheduler] = None
        if lab.aux_scheduler.enabled:
            a = lab.aux_scheduler
            total = int(cfg_get(trainer.config, "trainer.max_train_steps", 0)) or None
            self.aux = AuxDataScheduler(
                strategy=a.strategy, ratio_min=a.ratio_min, ratio_max=a.ratio_max,
                loss_scale_min=a.loss_scale_min, loss_scale_max=a.loss_scale_max,
                total_steps=total, init_u=a.init_u, drift_high=a.drift_high, drift_low=a.drift_low,
                gain=a.gain, max_step=a.max_step,
            )

        self.head_dropout: Optional[HeadDropoutSchedule] = None
        if lab.head_dropout.enabled:
            heads = list(getattr(self.model, "heads", {}).keys())
            if len(heads) > 1:
                self.head_dropout = HeadDropoutSchedule(heads, p_all=lab.head_dropout.p_all, seed=lab.head_dropout.seed)

        self.tracker: Optional[DriftTracker] = None
        self.secondary_tracker: Optional[DriftTracker] = None
        self.runner: Optional[ProbeRunner] = None
        self.extract_fn = extract_fn
        if lab.probes.enabled:
            if extract_fn is None or probe_batch is None:
                raise ValueError("probes are enabled: LabHooks needs extract_fn and probe_batch")
            was_training = self.model.training
            self.model.eval()
            try:
                reference = list(reference_reps) if reference_reps is not None else default_reference_reps(extract_fn, self.model, probe_batch)
                self.tracker = DriftTracker(extract_fn, probe_batch, reference=reference, layers=lab.probes.layers,
                                            layer_names=layer_names, compute_device=compute_device)
                if secondary_extract_fn is not None:
                    self.secondary_tracker = DriftTracker(secondary_extract_fn, probe_batch, reference=self.model,
                                                          layers=lab.probes.layers, layer_names=layer_names, compute_device=compute_device)
            finally:
                self.model.train(was_training)
            self.vlm_capability_fn = vlm_capability_fn
            self.runner = ProbeRunner(
                [ProbeSchedule(lab.probes.every_n_steps, self._measure, name="drift")],
                jsonl_path=lab.probes.jsonl_path,
            )

        self.llrd: Optional[DriftDrivenLLRD] = None
        if lab.llrd.enabled and lab.llrd.drift_driven:
            index = layer_group_index(trainer.optimizer.param_groups)
            if not index:
                raise ValueError("drift-driven LLRD needs optimizer groups tagged with layer_index (build_optimizer_and_scheduler)")
            self.llrd = DriftDrivenLLRD(
                trainer.optimizer, index, lr_scheduler=getattr(trainer, "lr_scheduler", None),
                drift_high=lab.llrd.drift_high, drift_low=lab.llrd.drift_low,
                down_factor=lab.llrd.down_factor, up_factor=lab.llrd.up_factor, min_scale=lab.llrd.min_scale,
            )

        if self.runner is not None and lab.probes.record_initial:
            # Reference vs. itself before any update: exactly zero unless the extractor is non-deterministic
            # (noise-floor check). Zero drift leaves the drift-driven schedulers at their initial state.
            self.runner.run(0)

    def _wrap_forward(self, model: nn.Module) -> None:
        if getattr(model, "_lab_forward_wrapped", False):
            return
        original = model.forward

        def forward(*args, **kwargs):
            output = original(*args, **kwargs)
            self._capture_losses(model, args, output)
            return output

        model.forward = forward
        model._lab_forward_wrapped = True

    def _capture_losses(self, module: nn.Module, inputs: Any, output: Any) -> None:
        if not isinstance(output, Mapping):
            return
        losses: Dict[str, float] = {}
        for key, value in output.items():
            if key == "action_loss" or not (key.startswith("loss_") or key.endswith("_loss")):
                continue
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                losses[key] = float(value.detach())
            elif isinstance(value, (int, float)):
                losses[key] = float(value)
        if losses:
            self.last_losses = losses

    # ------------------------------------------------------------------ measurement
    def _measure(self, step: int) -> Dict[str, Any]:
        assert self.tracker is not None
        was_training = self.model.training
        self.model.eval()
        try:
            drift = self.tracker.update(self.model, step=step)
            secondary = self.secondary_tracker.update(self.model, step=step) if self.secondary_tracker is not None else None
        finally:
            self.model.train(was_training)
        summary = self.tracker.summary()
        record: Dict[str, Any] = {"step": step, "time": time.time(), "drift": summary}
        if secondary is not None:
            record["drift_secondary"] = self.secondary_tracker.summary()
        # QwenBackboneProbe exposes how far embed_tokens moved from the pretrained snapshot; useful next to the drift.
        embed_stats = getattr(self.extract_fn, "embed_stats", None)
        if embed_stats:
            record["embed_tokens"] = dict(embed_stats)
        token_counts = getattr(self.extract_fn, "last_token_counts", None)
        if token_counts:
            record["probe_tokens"] = dict(token_counts)
        if self.vlm_capability_fn is not None:
            record["vlm_capability"] = dict(self.vlm_capability_fn(self.model))
        scalar = summary.get(self.lab.probes.drift_summary, summary.get("mean"))
        self.last_drift = float(scalar) if scalar is not None else None
        if not self.lab.probes.calibrate_only:
            if self.llrd is not None:
                record["llrd_multipliers"] = self.llrd.step(drift)
            if self.aux is not None and self.aux.strategy == "drift":
                prob, scale = self.aux.step(step, drift=self.last_drift)
                record["aux"] = {"vlm_sample_prob": prob, "vlm_loss_scale": scale}
        self.records.append(record)
        return record

    # ------------------------------------------------------------------ per-step hooks
    def before_step(self, step: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if self.aux is not None:
            if self.aux.strategy != "drift":
                self.aux.step(step)
            prob, scale = self.aux.apply_to_cfg(
                self.trainer.config, loss_scale_key=self.lab.aux_scheduler.loss_scale_key,
                sample_prob_key=self.lab.aux_scheduler.sample_prob_key,
            )
            info["vlm_sample_prob"], info["vlm_loss_scale"] = prob, scale
        if self.head_dropout is not None:
            active = self.head_dropout.active(step)
            self.model.active_heads = list(active)
            info["active_heads"] = list(active)
        return info

    def after_step(self, step: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if self.runner is not None:
            fired = self.runner.maybe_run(step)
            if fired:
                info["probe"] = fired[-1] if isinstance(fired, list) else fired
        if self.last_drift is not None:
            info["drift"] = self.last_drift
        if self.last_losses:
            info["losses"] = dict(self.last_losses)
        return info


def completed_updates_after(trainer: Any, step_before: int) -> int:
    """Number of optimizer updates completed once ``_train_step`` has returned.

    StarVLA increments ``completed_steps`` in ``train()`` *after* ``_train_step`` returns (and only when
    ``accelerator.sync_gradients`` is true), so inside the wrapper the counter still shows the pre-step value;
    other trainers (and the test double) bump it inside ``_train_step``. Either way the probe fired after this
    call is labelled with the number of updates actually applied, so JSONL step ``k`` means "after k updates"
    and matches the reference (``k = 0``) and StarVLA's ``Step k`` log line.
    """
    now = int(getattr(trainer, "completed_steps", step_before))
    if now > step_before:
        return now
    synced = getattr(getattr(trainer, "accelerator", None), "sync_gradients", True)
    return step_before + 1 if synced else step_before


def attach_to_trainer(trainer: Any, hooks: LabHooks) -> Any:
    """Wrap ``trainer._train_step`` so hooks run around every step; returns the trainer."""
    original = trainer._train_step

    def wrapped(*args, **kwargs):
        step = int(getattr(trainer, "completed_steps", 0))
        before = hooks.before_step(step)
        metrics = original(*args, **kwargs)
        after = hooks.after_step(completed_updates_after(trainer, step))
        if isinstance(metrics, dict):
            for key in ("vlm_sample_prob", "vlm_loss_scale", "drift"):
                if key in before:
                    metrics[f"lab/{key}"] = before[key]
                if key in after:
                    metrics[f"lab/{key}"] = after[key]
            if "active_heads" in before:
                metrics["lab/active_heads"] = ",".join(before["active_heads"])
            for key, value in after.get("losses", {}).items():
                metrics[f"lab/{key}"] = value
        return metrics

    trainer._train_step = wrapped
    trainer.lab_hooks = hooks
    return trainer
