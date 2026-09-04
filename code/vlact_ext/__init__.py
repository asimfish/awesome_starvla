"""VLAct (arXiv 2608.27550) recipe extensions for StarVLA.

Modules:
    wrap_aware_loss        (e) wrap-aware L1 for periodic joints (torch + numpy)
    unified_action_layout  (d) 20-D partially unified cross-embodiment layout + masks
    freeze_rules           (a) regex / range / ``llm_layers_below`` freeze rules
    multihead_framework    (c) ``QwenMultiHead`` framework (OFT + PI + GR00T, shared backbone)

Importing the package registers ``QwenMultiHead`` when StarVLA is importable, so copying this
directory into ``starVLA/model/framework/VLM4A/`` is enough for ``build_framework`` to find it.
"""

from . import freeze_rules, multihead_framework, unified_action_layout, wrap_aware_loss
from .freeze_rules import (
    build_param_lr_groups,
    expand_to_exact_paths,
    freeze_backbones,
    freeze_by_rules,
    freeze_llm_layers_below,
    install_into_starvla,
    resolve_frozen_param_ids,
)
from .multihead_framework import Qwen_MultiHead, QwenMultiHeadDefaultConfig, flow_matching_loss
from .unified_action_layout import (
    DEFAULT_LAYOUTS,
    UNIFIED_DIM,
    EmbodimentLayout,
    TransformedDataset,
    UnifiedActionLayout,
    UnifiedActionTransform,
    make_dataset_hook,
)
from .wrap_aware_loss import (
    flow_matching_sample_estimate,
    masked_wrap_aware_l1,
    masked_wrap_aware_l1_np,
    wrap_aware_residual,
    wrap_to_pi,
)

__all__ = [
    "freeze_rules",
    "multihead_framework",
    "unified_action_layout",
    "wrap_aware_loss",
    "build_param_lr_groups",
    "expand_to_exact_paths",
    "freeze_backbones",
    "freeze_by_rules",
    "freeze_llm_layers_below",
    "install_into_starvla",
    "resolve_frozen_param_ids",
    "Qwen_MultiHead",
    "QwenMultiHeadDefaultConfig",
    "flow_matching_loss",
    "DEFAULT_LAYOUTS",
    "UNIFIED_DIM",
    "EmbodimentLayout",
    "TransformedDataset",
    "UnifiedActionLayout",
    "UnifiedActionTransform",
    "make_dataset_hook",
    "flow_matching_sample_estimate",
    "masked_wrap_aware_l1",
    "masked_wrap_aware_l1_np",
    "wrap_aware_residual",
    "wrap_to_pi",
]
