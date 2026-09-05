"""starvla_lab: research package for the improvement plan in reports/10_improvement_plan.md.

Sub-packages (all CPU-testable, StarVLA-independent via dependency injection):
    probes     WP1  cross-head action probes, linear CKA, drift tracking, step-triggered hooks
    schedules  WP2/WP4  layer-wise lr decay (with freeze rules), drift-driven LLRD, auxiliary-data scheduler
    heads      WP3  future-feature prediction head, keyframe head, ``QwenMultiHeadLab`` framework
    bench      WP5/WP6  backbone-only benchmark protocol, training-overhead measurement
"""
from . import bench, heads, probes, schedules

__all__ = ["bench", "heads", "probes", "schedules"]
__version__ = "0.1.0"
