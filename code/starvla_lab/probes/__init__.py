"""WP1 diagnostic suite: cross-head action probes, linear CKA, drift tracking and step-triggered hooks."""

from .action_probe import (
    LinearProbe,
    MLPProbe,
    ProbeReport,
    VariantProbeResult,
    cross_head_probe_report,
    fit_linear_probe,
    fit_mlp_probe,
    mae_score,
    r2_score,
)
from .cka import layerwise_cka, linear_cka
from .drift import DriftRecord, DriftTracker, ExtractFn, drift_to_llrd_decay
from .hooks import ProbeRunner, ProbeSchedule, install_hook_example, read_jsonl, to_jsonable
from .qwen_extract import QwenBackboneProbe, gather_probe_batch, stratified_probe_batch

__all__ = [
    "QwenBackboneProbe",
    "gather_probe_batch",
    "stratified_probe_batch",
    "LinearProbe",
    "MLPProbe",
    "ProbeReport",
    "VariantProbeResult",
    "cross_head_probe_report",
    "fit_linear_probe",
    "fit_mlp_probe",
    "mae_score",
    "r2_score",
    "layerwise_cka",
    "linear_cka",
    "DriftRecord",
    "DriftTracker",
    "ExtractFn",
    "drift_to_llrd_decay",
    "ProbeRunner",
    "ProbeSchedule",
    "install_hook_example",
    "read_jsonl",
    "to_jsonable",
]
