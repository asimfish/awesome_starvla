"""WP2 / WP4 schedules: layer-wise lr decay with freeze rules, drift-driven LLRD, auxiliary data scheduling."""

from .aux_scheduler import STRATEGIES, AuxDataScheduler
from .llrd import (
    DEFAULT_EMBED_PATH,
    DEFAULT_LLM_LAYERS_PATH,
    DEFAULT_VISUAL_PATH,
    DriftDrivenLLRD,
    layer_group_index,
    layerwise_lr_decay_groups,
)

__all__ = [
    "STRATEGIES",
    "AuxDataScheduler",
    "DEFAULT_EMBED_PATH",
    "DEFAULT_LLM_LAYERS_PATH",
    "DEFAULT_VISUAL_PATH",
    "DriftDrivenLLRD",
    "layer_group_index",
    "layerwise_lr_decay_groups",
]
