# Copyright 2026 awesome_starvla contributors. MIT License.
"""``QwenMultiHeadLab``: ``Qwen_MultiHead`` (OFT + GR00T + PI) plus optional non-action auxiliary heads.

YAML (``framework.aux_heads``; every key optional, defaults shown)::

    framework:
      name: QwenMultiHeadLab
      aux_heads:
        featpred: {enabled: false, weight: 0.1, offsets: [1, 16], d_feat: 768, pooling: offset,
                   mlp_hidden: null, dropout: 0.0, cosine_weight: 1.0, mse_weight: 1.0}
        keyframe: {enabled: false, weight: 1.0, sigma: 1.0, horizon: null, mlp_hidden: null,
                   dropout: 0.0, pos_weight: null, threshold: 0.5, nms_window: 2, cooldown: 8,
                   max_events: 4}

Sample fields read on top of the parent's ``action`` / ``action_mask`` / ``periodic_mask``:
``future_features`` (``[len(offsets), d_feat]``, optional ``future_features_mask`` ``[len(offsets)]``)
and ``keyframe_steps`` (offsets in ``[0, horizon)`` relative to the chunk start). A sample without the
field is masked out of that head's loss; a batch without any is a zero loss.

``forward`` returns the parent's keys plus ``loss_featpred`` / ``loss_keyframe`` and folds
``weight * loss`` into ``action_loss`` so StarVLA's trainer trains the auxiliary heads unchanged.
The parent keeps all intermediates of ``forward`` local, so the subclass hooks ``_encode`` (the one
method both ``forward`` and ``predict_action`` route the backbone through) to capture the token
embeddings and extends the parent's result afterwards.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from vlact_ext.multihead_framework import (
        _STARVLA_AVAILABLE,
        FRAMEWORK_REGISTRY,
        Qwen_MultiHead,
        QwenMultiHeadDefaultConfig,
        _deep_merge,
        _select,
        _to_plain,
        _vlm_dims,
        gather_action_token_embeddings,
        merge_framework_config,
    )
except ImportError:  # installed as starVLA/vlact_ext next to StarVLA
    from starVLA.vlact_ext.multihead_framework import (
        _STARVLA_AVAILABLE,
        FRAMEWORK_REGISTRY,
        Qwen_MultiHead,
        QwenMultiHeadDefaultConfig,
        _deep_merge,
        _select,
        _to_plain,
        _vlm_dims,
        gather_action_token_embeddings,
        merge_framework_config,
    )

from .feature_prediction_head import FutureFeaturePredictionHead
from .keyframe_head import KeyframeHead, KeyframeWritePolicy, keyframe_bce_loss, soft_keyframe_labels

AUX_HEAD_NAMES: Tuple[str, ...] = ("featpred", "keyframe")
AUX_LOSS_KEYS: Dict[str, str] = {"featpred": "loss_featpred", "keyframe": "loss_keyframe"}

DEFAULT_AUX_HEADS: Dict[str, Dict[str, Any]] = {
    "featpred": {
        "enabled": False,
        "weight": 0.1,
        "offsets": [1, 16],
        "d_feat": 768,
        "pooling": "offset",
        "mlp_hidden": None,
        "dropout": 0.0,
        "cosine_weight": 1.0,
        "mse_weight": 1.0,
    },
    "keyframe": {
        "enabled": False,
        "weight": 1.0,
        "sigma": 1.0,
        # None -> action_horizon (one logit per action query)
        "horizon": None,
        "mlp_hidden": None,
        "dropout": 0.0,
        "pos_weight": None,
        # inference-side write policy (KeyframeWritePolicy / EvidenceMemory)
        "threshold": 0.5,
        "nms_window": 2,
        "cooldown": 8,
        "max_events": 4,
    },
}


@dataclass
class QwenMultiHeadLabDefaultConfig(QwenMultiHeadDefaultConfig):
    name: str = "QwenMultiHeadLab"
    aux_heads: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_AUX_HEADS))


@FRAMEWORK_REGISTRY.register("QwenMultiHeadLab")
class Qwen_MultiHeadLab(Qwen_MultiHead):
    """``Qwen_MultiHead`` with optional ``featpred`` / ``keyframe`` auxiliary heads on the action queries."""

    def __init__(
        self,
        config=None,
        vlm: Optional[nn.Module] = None,
        heads: Optional[Mapping[str, nn.Module]] = None,
        project_layers: Optional[nn.ModuleList] = None,
        aux_heads: Optional[Mapping[str, nn.Module]] = None,
        **kwargs,
    ) -> None:
        if _STARVLA_AVAILABLE and config is not None:
            config = merge_framework_config(QwenMultiHeadLabDefaultConfig, config)
        super().__init__(config, vlm=vlm, heads=heads, project_layers=project_layers, **kwargs)
        self._encoded: Optional[Tuple[Any, torch.Tensor]] = None

        aux_cfg = _deep_merge(DEFAULT_AUX_HEADS, _to_plain(_select(self.config.framework, "aux_heads")))
        unknown = [name for name in aux_cfg if name not in AUX_HEAD_NAMES]
        if unknown:
            raise ValueError(f"unknown aux head(s) {unknown} in framework.aux_heads; supported: {AUX_HEAD_NAMES}")
        self.aux_cfg = aux_cfg

        hidden_size, _ = _vlm_dims(self.qwen_vl_interface)
        if aux_heads is None:
            aux_heads = self._build_aux_heads(aux_cfg, hidden_size)
        unknown = [name for name in aux_heads if name not in AUX_HEAD_NAMES]
        if unknown:
            raise ValueError(f"unknown aux head(s) {unknown}; supported: {AUX_HEAD_NAMES}")
        self.aux_heads = nn.ModuleDict(dict(aux_heads))
        self.aux_weights: Dict[str, float] = {name: float(aux_cfg[name].get("weight", 1.0)) for name in self.aux_heads}

        keyframe_cfg = aux_cfg["keyframe"]
        self.keyframe_sigma = float(keyframe_cfg.get("sigma", 1.0))
        pos_weight = keyframe_cfg.get("pos_weight")
        self.keyframe_pos_weight: Optional[float] = None if pos_weight is None else float(pos_weight)
        self.keyframe_policy: Optional[KeyframeWritePolicy] = None
        if "keyframe" in self.aux_heads:
            self.keyframe_policy = KeyframeWritePolicy(
                threshold=float(keyframe_cfg.get("threshold", 0.5)),
                nms_window=int(keyframe_cfg.get("nms_window", 0)),
                cooldown=int(keyframe_cfg.get("cooldown", 0)),
                max_events=keyframe_cfg.get("max_events"),
            )

        # the auxiliary heads read the action-query states even when the OFT head is disabled
        if self.aux_heads and self.action_token_id is None:
            self.action_token_id = int(
                self.qwen_vl_interface.processor.tokenizer(self.action_token, add_special_tokens=False)["input_ids"][0]
            )

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _build_aux_heads(aux_cfg: Mapping[str, Mapping[str, Any]], hidden_size: int) -> Dict[str, nn.Module]:
        heads: Dict[str, nn.Module] = {}
        featpred = aux_cfg["featpred"]
        if featpred.get("enabled", False):
            heads["featpred"] = FutureFeaturePredictionHead(
                hidden_size,
                int(featpred["d_feat"]),
                offsets=featpred["offsets"],
                pooling=str(featpred.get("pooling", "offset")),
                mlp_hidden=featpred.get("mlp_hidden"),
                dropout=float(featpred.get("dropout", 0.0)),
                cosine_weight=float(featpred.get("cosine_weight", 1.0)),
                mse_weight=float(featpred.get("mse_weight", 1.0)),
            )
        keyframe = aux_cfg["keyframe"]
        if keyframe.get("enabled", False):
            horizon = keyframe.get("horizon")
            heads["keyframe"] = KeyframeHead(
                hidden_size,
                horizon=None if horizon is None else int(horizon),
                mlp_hidden=keyframe.get("mlp_hidden"),
                dropout=float(keyframe.get("dropout", 0.0)),
            )
        return heads

    @property
    def keyframe_horizon(self) -> Optional[int]:
        if "keyframe" not in self.aux_heads:
            return None
        return self.aux_heads["keyframe"].horizon or self.chunk_len

    # ------------------------------------------------------------------ parent hooks
    def _prepare_instructions(self, examples: Sequence[dict]) -> List[str]:
        instructions = super()._prepare_instructions(examples)
        if self.aux_heads and "oft" not in self.heads:
            suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{self.action_token * self.chunk_len}<action>."
            instructions = [instruction + suffix for instruction in instructions]
        return instructions

    def _encode(self, batch_images, instructions: List[str]):
        inputs, last_hidden, pi_embs = super()._encode(batch_images, instructions)
        self._encoded = (inputs, last_hidden)
        return inputs, last_hidden, pi_embs

    def _pop_encoded(self) -> Tuple[Any, torch.Tensor]:
        if self._encoded is None:
            raise RuntimeError("no cached backbone output; _encode must run before the auxiliary heads")
        encoded, self._encoded = self._encoded, None
        return encoded

    # ------------------------------------------------------------------ auxiliary targets
    @staticmethod
    def _collect_featpred_targets(examples: Sequence[dict], head: FutureFeaturePredictionHead) -> Tuple[np.ndarray, np.ndarray]:
        shape = (head.num_offsets, head.feat_dim)
        target = np.zeros((len(examples),) + shape, dtype=np.float32)
        valid = np.zeros((len(examples), head.num_offsets), dtype=bool)
        for row, example in enumerate(examples):
            feats = example.get("future_features")
            if feats is None:
                continue
            feats = np.asarray(feats, dtype=np.float32)
            if feats.shape != shape:
                raise ValueError(f"future_features must have shape {shape} (offsets x d_feat), got {feats.shape}")
            target[row] = feats
            mask = example.get("future_features_mask")
            valid[row] = True if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
        return target, valid

    @staticmethod
    def _collect_keyframe_steps(examples: Sequence[dict]) -> Tuple[List[List[int]], np.ndarray]:
        steps: List[List[int]] = []
        valid = np.zeros(len(examples), dtype=bool)
        for row, example in enumerate(examples):
            raw = example.get("keyframe_steps")
            if raw is None:
                steps.append([])
                continue
            steps.append([int(s) for s in np.asarray(raw).reshape(-1)])
            valid[row] = True
        return steps, valid

    # ------------------------------------------------------------------ auxiliary losses
    def _action_queries(self, inputs, last_hidden: torch.Tensor) -> torch.Tensor:
        return gather_action_token_embeddings(last_hidden, inputs["input_ids"], self.action_token_id, self.chunk_len)

    @staticmethod
    def _cast_for(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=next(head.parameters()).dtype)

    def _featpred_loss(self, queries: torch.Tensor, examples: Sequence[dict]) -> torch.Tensor:
        head = self.aux_heads["featpred"]
        target_np, valid_np = self._collect_featpred_targets(examples, head)
        pred = head(self._cast_for(head, queries))
        target = torch.as_tensor(target_np, device=pred.device)
        valid = torch.as_tensor(valid_np, device=pred.device)
        return head.loss(pred, target, valid)["loss"]

    def _keyframe_loss(self, queries: torch.Tensor, examples: Sequence[dict]) -> torch.Tensor:
        head = self.aux_heads["keyframe"]
        logits = head(self._cast_for(head, queries))
        steps, valid_np = self._collect_keyframe_steps(examples)
        labels = soft_keyframe_labels(steps, logits.shape[1], self.keyframe_sigma, device=logits.device)
        valid = torch.as_tensor(valid_np, device=logits.device)
        return keyframe_bce_loss(logits, labels, valid, pos_weight=self.keyframe_pos_weight)

    # ------------------------------------------------------------------ training / inference
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        result = super().forward(examples, **kwargs)
        inputs, last_hidden = self._pop_encoded()
        device = last_hidden.device

        losses: Dict[str, torch.Tensor] = {}
        if self.aux_heads:
            queries = self._action_queries(inputs, last_hidden)
            if "featpred" in self.aux_heads:
                losses["featpred"] = self._featpred_loss(queries, examples)
            if "keyframe" in self.aux_heads:
                losses["keyframe"] = self._keyframe_loss(queries, examples)
            result["action_loss"] = result["action_loss"] + sum(self.aux_weights[name] * loss for name, loss in losses.items())

        for name in AUX_HEAD_NAMES:
            result[AUX_LOSS_KEYS[name]] = losses.get(name, torch.zeros((), device=device, dtype=torch.float32))
        return result

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, head: Optional[str] = None, robot_tag: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        result = super().predict_action(examples, head=head, robot_tag=robot_tag, **kwargs)
        inputs, last_hidden = self._pop_encoded()
        if "keyframe" in self.aux_heads:
            keyframe = self.aux_heads["keyframe"]
            probs = keyframe.probabilities(self._cast_for(keyframe, self._action_queries(inputs, last_hidden)))
            result["keyframe_probs"] = probs.detach().float().cpu().numpy()
        return result
