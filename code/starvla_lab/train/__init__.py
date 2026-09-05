"""Training-loop integration: ``trainer.lab.*`` config, LLRD-aware optimizer builder, per-step hooks."""
from .integration import LabHooks, attach_to_trainer, build_optimizer_and_scheduler, default_reference_reps, unwrap_model
from .lab_config import AuxSchedulerConfig, HeadDropoutConfig, LLRDConfig, LabConfig, ProbesConfig, cfg_get, cfg_set

__all__ = [
    "LabHooks", "attach_to_trainer", "build_optimizer_and_scheduler", "default_reference_reps", "unwrap_model",
    "AuxSchedulerConfig", "HeadDropoutConfig", "LLRDConfig", "LabConfig", "ProbesConfig", "cfg_get", "cfg_set",
]
