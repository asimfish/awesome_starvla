# Copyright 2026 awesome_starvla contributors. MIT License.
"""WP3 auxiliary heads: future visual-feature prediction, keyframe prediction, and their registration
into ``QwenMultiHead`` as ``QwenMultiHeadLab``.

Modules:
    feature_prediction_head  ``FutureFeaturePredictionHead`` (cosine + MSE on frozen-extractor features)
    keyframe_head            ``KeyframeHead``, soft labels + BCE, NMS / cooldown write policy, memory, curriculum
    register                 ``Qwen_MultiHeadLab`` (``framework.name: QwenMultiHeadLab``)
"""

from . import feature_prediction_head, keyframe_head, register
from .feature_prediction_head import FutureFeaturePredictionHead, feature_prediction_loss, masked_mean, targets_from_sequence
from .keyframe_head import (
    EvidenceMemory,
    KeyframeHead,
    KeyframeWritePolicy,
    TeacherStudentCurriculum,
    keyframe_bce_loss,
    nms_1d,
    soft_keyframe_labels,
)
from .register import AUX_HEAD_NAMES, AUX_LOSS_KEYS, DEFAULT_AUX_HEADS, Qwen_MultiHeadLab, QwenMultiHeadLabDefaultConfig

__all__ = [
    "feature_prediction_head",
    "keyframe_head",
    "register",
    "FutureFeaturePredictionHead",
    "feature_prediction_loss",
    "masked_mean",
    "targets_from_sequence",
    "EvidenceMemory",
    "KeyframeHead",
    "KeyframeWritePolicy",
    "TeacherStudentCurriculum",
    "keyframe_bce_loss",
    "nms_1d",
    "soft_keyframe_labels",
    "AUX_HEAD_NAMES",
    "AUX_LOSS_KEYS",
    "DEFAULT_AUX_HEADS",
    "Qwen_MultiHeadLab",
    "QwenMultiHeadLabDefaultConfig",
]
