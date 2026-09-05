# Copyright 2026 awesome_starvla contributors. MIT License.
"""QwenMultiHead: OFT + PI + GR00T heads sharing one Qwen-VL backbone forward (VLAct recipe (c)).

拷贝到 ``starVLA/model/framework/VLM4A/QwenMultiHead.py`` 即可被 ``build_framework`` 自动注册为
``framework.name: QwenMultiHead``（``_auto_import_framework_modules`` 会扫描 VLM4A/ 下所有模块）。
同时需要把 ``wrap_aware_loss.py`` 与 ``unified_action_layout.py`` 放到可导入的位置，推荐整包拷贝为
``starVLA/vlact_ext/``（本文件会依次尝试 ``.``、``starVLA.vlact_ext``、``vlact_ext`` 三个导入前缀）。
也可以只写一个一行 shim：``starVLA/model/framework/VLM4A/QwenMultiHead.py`` 内容为
``from starVLA.vlact_ext.multihead_framework import *  # noqa``。

Design
------
* One backbone forward (``output_hidden_states=True``) feeds three existing StarVLA heads:
    - ``oft``   ``MLP_ActionHeader.L1RegressionActionHead`` on the ``<action>`` query tokens (QwenOFT);
    - ``gr00t`` ``GR00T_ActionHeader.FlowmatchingActionHead`` cross-attending to ``hidden_states[-1]``;
    - ``pi``    ``LayerwiseFM_ActionHeader.LayerwiseFlowmatchingActionHead`` + per-layer projectors (QwenPI_v3).
  Head constructors are reused verbatim; each head gets its own ``framework.action_model`` view built
  from the shared block plus ``framework.heads.<name>.action_model`` overrides.
* ``forward(examples) -> {"action_loss": sum_h w_h * L_h, "loss_oft", "loss_pi", "loss_gr00t"}``.
  Every head can be switched off (``heads.<name>.enabled``) and weighted (``heads.<name>.loss_weight``).
* ``predict_action(examples, head=None, robot_tag=None) -> {"normalized_actions": np.ndarray[B, T, D]}``.
* Sample-level ``action_mask`` / ``periodic_mask`` (VLAct recipes (d)/(e)) are honoured by all heads:
  OFT uses ``masked_wrap_aware_l1``; the flow-matching heads use a masked velocity MSE plus an optional
  wrap-aware L1 on the one-step clean-sample estimate ``x1_hat = x_t + (1 - t) * v_hat``. That loss is
  computed here from the heads' own sub-modules (``action_encoder`` / ``future_tokens`` / ``model`` /
  ``action_decoder``) because ``FlowmatchingActionHead.forward`` does not accept a mask.
* State is injected as discretised text (``[STATE] ... [ACTION]``, QwenPI_v3 / QwenOFT style) for all
  heads, so the GR00T head is built with ``state_dim: 0`` (no ``state_encoder``).
* Dependency injection for CPU tests: ``Qwen_MultiHead(config, vlm=..., heads={...}, project_layers=...)``
  skips the StarVLA constructors entirely.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from .unified_action_layout import UnifiedActionLayout
    from .wrap_aware_loss import flow_matching_sample_estimate, masked_wrap_aware_l1
except ImportError:  # copied as a standalone file into starVLA/model/framework/VLM4A/
    try:
        from starVLA.vlact_ext.unified_action_layout import UnifiedActionLayout
        from starVLA.vlact_ext.wrap_aware_loss import flow_matching_sample_estimate, masked_wrap_aware_l1
    except ImportError:
        from vlact_ext.unified_action_layout import UnifiedActionLayout
        from vlact_ext.wrap_aware_loss import flow_matching_sample_estimate, masked_wrap_aware_l1

try:
    from starVLA.model.framework.base_framework import baseframework
    from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, merge_framework_config
    from starVLA.model.tools import FRAMEWORK_REGISTRY

    _STARVLA_AVAILABLE = True
except ImportError:  # stand-ins so the file imports (and the class can be unit-tested) without StarVLA
    _STARVLA_AVAILABLE = False

    class _Registry:
        def __init__(self) -> None:
            self._registry: Dict[str, type] = {}

        def register(self, key: str):
            def decorator(cls):
                self._registry[key] = cls
                return cls

            return decorator

        def __getitem__(self, key: str):
            return self._registry[key]

    FRAMEWORK_REGISTRY = _Registry()

    class baseframework(nn.Module):  # noqa: N801 - mirrors StarVLA's class name
        def __init__(self, hf_config=None) -> None:
            super().__init__()

    def merge_framework_config(default_config_cls, cfg):
        return cfg

    def add_discretized_state_to_instruction(instructions, states, num_bins: int = 256):
        bins = np.linspace(-1, 1, num_bins + 1)[:-1]
        out = []
        for instr, state in zip(instructions, states):
            ids = np.digitize(np.asarray(state)[0], bins=bins) - 1
            out.append(f"{instr} [STATE] {' '.join(map(str, ids))} [ACTION]")
        return out


try:
    from deployment.model_server.tools.image_tools import to_pil_preserve
except ImportError:

    def to_pil_preserve(images):
        return images


try:
    from starVLA.training.trainer_utils.trainer_tools import resize_images
except ImportError:

    def resize_images(images, target_size=(224, 224)):
        if isinstance(images, list):
            return [resize_images(img, target_size) for img in images]
        return images.resize(tuple(target_size)) if hasattr(images, "resize") else images


HEAD_NAMES: Tuple[str, ...] = ("oft", "gr00t", "pi")
LOSS_KEYS: Dict[str, str] = {"oft": "loss_oft", "gr00t": "loss_gr00t", "pi": "loss_pi"}
ACTION_QUERY_TOKEN = "\U0001F50D"  # the magnifier emoji QwenOFT uses as a single-token action query
_FM_HEAD_ATTRS = ("sample_time", "num_timestep_buckets", "action_encoder", "future_tokens", "model", "action_decoder", "predict_action", "config")


# ──────────────────────────────────────────────────────────────────────
#  Default config (merged with the YAML ``framework:`` section; YAML wins)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenMultiHeadDefaultConfig:
    name: str = "QwenMultiHead"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            # both auto-overridden from the loaded VLM
            "vl_hidden_dim": 2560,
            "num_vl_layers": 36,
        }
    )

    # Shared by every head (targets must be identical). Head-specific fields live in ``heads.<name>.action_model``.
    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 20,
            # state goes into the instruction text for all heads -> no GR00T state_encoder
            "state_dim": 0,
            "action_horizon": 16,
            # OFT MLP input dim, overwritten by the VLM hidden size at runtime
            "action_hidden_dim": 2560,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "repeated_diffusion_steps": 4,
        }
    )

    heads: dict = field(
        default_factory=lambda: {
            "oft": {"enabled": True, "loss_weight": 1.0, "action_model": {"action_model_type": "MLP"}},
            "gr00t": {
                "enabled": True,
                "loss_weight": 1.0,
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": 1024,
                    "diffusion_model_cfg": {
                        # aligned to the VLM hidden size at runtime
                        "cross_attention_dim": None,
                        "dropout": 0.2,
                        "final_dropout": True,
                        "interleave_self_attention": True,
                        "norm_type": "ada_norm",
                        "num_layers": 16,
                        "output_dim": 1024,
                        "positional_embeddings": None,
                    },
                },
            },
            "pi": {
                "enabled": True,
                "loss_weight": 1.0,
                "action_model": {
                    "action_model_type": "LayerwiseFM",
                    "diffusion_model_cfg": {
                        # DiT width; project_layers compress VLM hidden -> this dim (QwenPI_v3)
                        "action_dit_hidden_dim": 1024,
                        "dropout": 0.2,
                        "final_dropout": True,
                        "interleave_self_attention": False,
                        "norm_type": "ada_norm",
                        "positional_embeddings": None,
                        "attention_head_dim": 64,
                    },
                },
            },
        }
    )

    # which head ``predict_action`` uses when ``head=`` is not given
    predict_head: str = "oft"
    # hide the OFT ``<action>`` query positions from the flow-matching heads' cross-attention
    mask_oft_queries_for_fm_heads: bool = False

    # VLAct (e). ``period`` is measured in the (normalised) action space: 2.0 for [-pi, pi] -> [-1, 1].
    # May be a list of length action_dim for per-slot periods.
    wrap_aware: dict = field(
        default_factory=lambda: {"enabled": True, "period": 2.0, "fm_sample_loss_weight": 1.0}
    )

    # VLAct (d). When enabled, samples whose action dim != action_dim are mapped through the layout
    # (``robot_tag`` required) and masks are generated; samples already carrying ``action_mask`` pass through.
    unified_layout: dict = field(default_factory=lambda: {"enabled": False, "unified_dim": 20, "layouts": None})


# ──────────────────────────────────────────────────────────────────────
#  Config helpers (work for OmegaConf, AccessTrackedConfig, dict, SimpleNamespace)
# ──────────────────────────────────────────────────────────────────────
def _to_plain(node) -> Dict[str, Any]:
    """Whole sub-tree as a plain dict. Reading it as one leaf keeps AccessTrackedConfig saving all of it."""
    if node is None:
        return {}
    if hasattr(node, "to_dict") and callable(node.to_dict):
        return dict(node.to_dict())
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(node):
            return dict(OmegaConf.to_container(node, resolve=True))
    except ImportError:
        pass
    if isinstance(node, Mapping):
        return {k: _to_plain(v) if _is_node(v) else v for k, v in node.items()}
    if hasattr(node, "__dict__"):
        return {k: _to_plain(v) if _is_node(v) else v for k, v in vars(node).items()}
    raise TypeError(f"cannot convert {type(node).__name__} to a plain dict")


def _is_node(value) -> bool:
    if isinstance(value, (str, bytes, int, float, bool, list, tuple, np.ndarray, torch.Tensor)) or value is None:
        return False
    return isinstance(value, Mapping) or hasattr(value, "__dict__")


def _deep_merge(base: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _select(cfg, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if node is None:
            return default
        try:
            node = getattr(node, part)
        except (AttributeError, KeyError):
            return default
    return default if node is None else node


def _autocast(dtype: torch.dtype):
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    valid = mask.to(dtype=values.dtype).expand(values.shape)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


# ──────────────────────────────────────────────────────────────────────
#  Head-level losses reused by forward()
# ──────────────────────────────────────────────────────────────────────
def gather_action_token_embeddings(last_hidden: torch.Tensor, input_ids: torch.Tensor, token_id: int, chunk_len: int) -> torch.Tensor:
    """Pick the last ``chunk_len`` occurrences of ``token_id`` per row, in temporal order (QwenOFT)."""
    B, L, H = last_hidden.shape
    mask = input_ids == token_id
    counts = mask.sum(dim=1)
    if bool((counts < chunk_len).any()):
        bad = (counts < chunk_len).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(f"samples {bad} have fewer than {chunk_len} action query tokens (counts={counts.tolist()})")
    idx = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
    masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
    selected = masked_pos.topk(k=chunk_len, dim=-1).values.sort(dim=-1).values
    return last_hidden.gather(dim=1, index=selected.unsqueeze(-1).expand(-1, -1, H))


def flow_matching_loss(
    head: nn.Module,
    cond,
    target: torch.Tensor,
    *,
    layerwise: bool,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
    periodic_mask: Optional[torch.Tensor] = None,
    wrap_weight: float = 0.0,
    period=2.0,
) -> Dict[str, torch.Tensor]:
    """Masked flow-matching loss built from a StarVLA flow-matching head's sub-modules.

    Mirrors ``FlowmatchingActionHead.forward`` / ``LayerwiseFlowmatchingActionHead.forward`` (same
    Beta time sampling, ``x_t = (1-t) eps + t a``, velocity target ``a - eps``, ``future_tokens`` prefix,
    ``return_pre_output`` for the layer-wise DiT) and adds: masked mean over ``active_mask`` and an
    optional wrap-aware L1 on ``x1_hat`` restricted to ``active & periodic`` dims.
    """
    B, T, _ = target.shape
    device = target.device
    noise = torch.randn_like(target)
    t = head.sample_time(B, device=device, dtype=target.dtype)[:, None, None]
    noisy = (1.0 - t) * noise + t * target
    velocity = target - noise
    t_discrete = (t[:, 0, 0] * head.num_timestep_buckets).long()

    feats = head.action_encoder(noisy, t_discrete)
    if getattr(head.config, "add_pos_embed", False):
        pos = torch.arange(feats.shape[1], dtype=torch.long, device=device)
        feats = feats + head.position_embedding(pos).unsqueeze(0)
    future = head.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
    sa_embs = torch.cat((future, feats), dim=1)

    dit_kwargs = {"return_pre_output": True} if layerwise else {}
    out = head.model(
        hidden_states=sa_embs,
        encoder_hidden_states=cond,
        timestep=t_discrete,
        encoder_attention_mask=encoder_attention_mask,
        **dit_kwargs,
    )
    pred_velocity = head.action_decoder(out)[:, -T:].float()

    # module inputs stay in ``target.dtype`` (bf16 under DeepSpeed); the reductions run in fp32
    velocity_loss = _masked_mean((pred_velocity - velocity.float()) ** 2, active_mask)
    wrap_loss = torch.zeros((), device=device, dtype=torch.float32)
    if wrap_weight > 0 and periodic_mask is not None and bool(periodic_mask.any()):
        x1_hat = flow_matching_sample_estimate(noisy.float(), pred_velocity, t.float())
        wrap_region = periodic_mask if active_mask is None else (active_mask & periodic_mask)
        wrap_loss = masked_wrap_aware_l1(x1_hat, target.float(), wrap_region, periodic_mask, period)
    return {"loss": velocity_loss + wrap_weight * wrap_loss, "velocity_loss": velocity_loss, "wrap_loss": wrap_loss}


# ──────────────────────────────────────────────────────────────────────
#  Framework
# ──────────────────────────────────────────────────────────────────────
@FRAMEWORK_REGISTRY.register("QwenMultiHead")
class Qwen_MultiHead(baseframework):
    """Qwen-VL backbone + {OFT, GR00T, PI} heads trained with ``action_loss = sum_h w_h * L_h``."""

    def __init__(
        self,
        config=None,
        vlm: Optional[nn.Module] = None,
        heads: Optional[Mapping[str, nn.Module]] = None,
        project_layers: Optional[nn.ModuleList] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if _STARVLA_AVAILABLE:
            self.config = merge_framework_config(QwenMultiHeadDefaultConfig, config)
        else:
            if vlm is None or heads is None:
                raise ImportError(
                    "StarVLA is not importable: Qwen_MultiHead can only be constructed with injected "
                    "`vlm=` and `heads=` (copy this file into StarVLA for the default constructors)."
                )
            self.config = config

        fw = self.config.framework
        shared_am = _to_plain(fw.action_model)
        heads_cfg = _to_plain(fw.heads)
        wrap_cfg = _to_plain(_select(fw, "wrap_aware"))
        layout_cfg = _to_plain(_select(fw, "unified_layout"))

        self.action_horizon = int(shared_am["action_horizon"])
        self.chunk_len = self.action_horizon
        self.action_dim = int(shared_am["action_dim"])
        self.mask_oft_queries_for_fm_heads = bool(_select(fw, "mask_oft_queries_for_fm_heads", False))

        self.wrap_enabled = bool(wrap_cfg.get("enabled", False))
        period = wrap_cfg.get("period", 2.0)
        self.wrap_period = torch.as_tensor(np.asarray(period, dtype=np.float32)) if isinstance(period, (list, tuple)) else float(period)
        self.fm_sample_loss_weight = float(wrap_cfg.get("fm_sample_loss_weight", 1.0))

        self.layout: Optional[UnifiedActionLayout] = None
        if layout_cfg.get("enabled", False):
            self.layout = UnifiedActionLayout.from_config(layout_cfg)
            if self.layout.unified_dim != self.action_dim:
                raise ValueError(f"unified_layout.unified_dim={self.layout.unified_dim} != action_model.action_dim={self.action_dim}")

        # backbone
        self.qwen_vl_interface = vlm if vlm is not None else self._build_vlm()
        hidden_size, num_vl_layers = _vlm_dims(self.qwen_vl_interface)
        try:
            fw.qwenvl.vl_hidden_dim = hidden_size
            fw.qwenvl.num_vl_layers = num_vl_layers
        except Exception:
            pass

        # heads
        if heads is None:
            heads, project_layers = self._build_default_heads(shared_am, heads_cfg, hidden_size, num_vl_layers)
        unknown = [name for name in heads if name not in HEAD_NAMES]
        if unknown:
            raise ValueError(f"unknown head(s) {unknown}; supported: {HEAD_NAMES}")
        if not heads:
            raise ValueError("QwenMultiHead needs at least one enabled head")
        self.heads = nn.ModuleDict(dict(heads))
        for name in ("gr00t", "pi"):
            if name in self.heads:
                _check_fm_head(name, self.heads[name])

        # Head dropout support: ``None`` = every enabled head; otherwise only the named heads contribute
        # to ``action_loss`` for the current step (set per step by an external schedule, e.g.
        # ``starvla_lab.bench.HeadDropoutSchedule``). An empty or foreign selection falls back to all heads.
        self.active_heads: Optional[Sequence[str]] = None
        self.head_weights: Dict[str, float] = {
            name: float((heads_cfg.get(name) or {}).get("loss_weight", 1.0)) for name in self.heads
        }
        self.repeated_diffusion_steps: Dict[str, int] = {
            name: int(_deep_merge(shared_am, (heads_cfg.get(name) or {}).get("action_model")).get("repeated_diffusion_steps", 4))
            for name in self.heads
            if name != "oft"
        }

        if "pi" in self.heads:
            if project_layers is None:
                project_layers = nn.ModuleList(nn.Identity() for _ in range(len(self.heads["pi"].model.transformer_blocks)))
            self.project_layers = project_layers
        else:
            self.project_layers = nn.ModuleList()

        self.action_token = ACTION_QUERY_TOKEN
        self.action_token_id: Optional[int] = None
        if "oft" in self.heads:
            self.action_token_id = int(
                self.qwen_vl_interface.processor.tokenizer(self.action_token, add_special_tokens=False)["input_ids"][0]
            )

        wanted = str(_select(fw, "predict_head", "oft"))
        if wanted not in self.heads:
            fallback = next(iter(self.heads))
            warnings.warn(f"predict_head={wanted!r} is not an enabled head; predict_action defaults to {fallback!r}")
            wanted = fallback
        self.predict_head = wanted

    # ------------------------------------------------------------------ construction
    def _build_vlm(self) -> nn.Module:
        from starVLA.model.modules.vlm import get_vlm_model

        return get_vlm_model(config=self.config)

    def _build_default_heads(
        self, shared_am: Dict[str, Any], heads_cfg: Dict[str, Any], hidden_size: int, num_vl_layers: int
    ) -> Tuple[Dict[str, nn.Module], Optional[nn.ModuleList]]:
        from omegaconf import OmegaConf

        from starVLA.model.framework.share_tools import populate_layerwise_dit_cfg

        heads: Dict[str, nn.Module] = {}
        project_layers: Optional[nn.ModuleList] = None
        for name, head_cfg in heads_cfg.items():
            head_cfg = head_cfg or {}
            if not head_cfg.get("enabled", True):
                continue
            if name not in HEAD_NAMES:
                raise ValueError(f"unknown head {name!r} in framework.heads; supported: {HEAD_NAMES}")
            am = _deep_merge(shared_am, head_cfg.get("action_model"))
            cfg = OmegaConf.create({"framework": {"action_model": am}})
            if name == "oft":
                from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model as build_oft

                cfg.framework.action_model.action_hidden_dim = int(hidden_size)
                heads[name] = build_oft(config=cfg)
            elif name == "gr00t":
                from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model as build_gr00t

                cfg.framework.action_model.diffusion_model_cfg.cross_attention_dim = int(hidden_size)
                heads[name] = build_gr00t(config=cfg)
            else:
                from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model as build_pi

                dit_hidden = cfg.framework.action_model.diffusion_model_cfg.get("action_dit_hidden_dim", None) or hidden_size
                dit_hidden = int(dit_hidden)
                populate_layerwise_dit_cfg(cfg, dit_hidden_dim=dit_hidden, num_dit_layers=int(num_vl_layers))
                heads[name] = build_pi(config=cfg)
                num_dit_layers = len(heads[name].model.transformer_blocks)
                project_layers = nn.ModuleList(
                    (
                        nn.Identity()
                        if dit_hidden == int(hidden_size)
                        else nn.Sequential(nn.LayerNorm(int(hidden_size)), nn.Linear(int(hidden_size), dit_hidden))
                    )
                    for _ in range(num_dit_layers)
                )
        return heads, project_layers

    # ------------------------------------------------------------------ shared pieces
    def _prepare_instructions(self, examples: Sequence[dict]) -> List[str]:
        instructions = [example["lang"] for example in examples]
        if "state" in examples[0] and examples[0]["state"] is not None:
            states = []
            for example in examples:
                state = np.asarray(example["state"])
                states.append(state if state.ndim == 2 else state.reshape(1, -1))
            instructions = add_discretized_state_to_instruction(instructions, states)
        if "oft" in self.heads:
            suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{self.action_token * self.chunk_len}<action>."
            instructions = [instruction + suffix for instruction in instructions]
        return instructions

    def _encode(self, batch_images, instructions: List[str]):
        """Backbone forward. Returns (inputs, last_hidden, layer-wise PI embeddings or None)."""
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with _autocast(torch.bfloat16):
            outputs = self.qwen_vl_interface(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
            hidden_states = tuple(outputs.hidden_states)
            pi_embs = None
            if "pi" in self.heads:
                n = len(self.project_layers)
                if len(hidden_states) - 1 < n:
                    raise ValueError(f"backbone returned {len(hidden_states) - 1} layer outputs but the PI head expects {n}")
                pi_embs = [proj(h) for proj, h in zip(self.project_layers, hidden_states[-n:])]
        return inputs, hidden_states[-1], pi_embs

    def _encoder_mask(self, inputs) -> Optional[torch.Tensor]:
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            return None
        mask = attention_mask.to(dtype=torch.bool)
        if self.mask_oft_queries_for_fm_heads and self.action_token_id is not None:
            mask = mask & (inputs["input_ids"] != self.action_token_id)
        return mask

    def _collect_targets(self, examples: Sequence[dict]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        actions, masks, periodic = [], [], []
        for example in examples:
            action = np.asarray(example["action"])
            mask = example.get("action_mask")
            pmask = example.get("periodic_mask")
            if action.shape[-1] != self.action_dim:
                if self.layout is None:
                    raise ValueError(
                        f"action dim {action.shape[-1]} != action_model.action_dim {self.action_dim}; "
                        "enable framework.unified_layout or apply UnifiedActionTransform in the dataloader"
                    )
                action, active, per = self.layout.to_unified(action, example.get("robot_tag"))
                mask, pmask = active, per
            actions.append(action)
            masks.append(None if mask is None else np.broadcast_to(np.asarray(mask, dtype=bool), action.shape))
            periodic.append(None if pmask is None else np.broadcast_to(np.asarray(pmask, dtype=bool), action.shape))

        present = [m is not None for m in masks]
        if any(present) and not all(present):
            raise ValueError("action_mask must be present for every example in a batch or for none")
        target = np.stack(actions).astype(np.float32)[:, -self.action_horizon :, :]
        active_np = np.stack(masks)[:, -self.action_horizon :, :] if all(present) else None
        periodic_np = None
        if self.wrap_enabled and all(p is not None for p in periodic):
            periodic_np = np.stack(periodic)[:, -self.action_horizon :, :]
        return target, active_np, periodic_np

    def _period(self, device: torch.device):
        if isinstance(self.wrap_period, torch.Tensor):
            return self.wrap_period.to(device)
        return self.wrap_period

    # ------------------------------------------------------------------ training
    def _active_head_set(self) -> set:
        """Heads that contribute to this step's loss (see ``active_heads``)."""
        enabled = set(self.heads.keys())
        if not self.active_heads:
            return enabled
        chosen = enabled & set(self.active_heads)
        return chosen or enabled

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        batch_images = [example["image"] for example in examples]
        instructions = self._prepare_instructions(examples)
        target_np, active_np, periodic_np = self._collect_targets(examples)

        inputs, last_hidden, pi_embs = self._encode(batch_images, instructions)
        device = last_hidden.device
        encoder_mask = self._encoder_mask(inputs)

        with _autocast(torch.float32):
            # same dtype as the backbone output (bf16 under DeepSpeed) so the heads' Linear layers match
            target = torch.as_tensor(target_np, device=device, dtype=last_hidden.dtype)
            active = None if active_np is None else torch.as_tensor(active_np, device=device, dtype=torch.bool)
            periodic = None if periodic_np is None else torch.as_tensor(periodic_np, device=device, dtype=torch.bool)
            period = self._period(device)
            wrap_weight = self.fm_sample_loss_weight if self.wrap_enabled else 0.0

            losses: Dict[str, torch.Tensor] = {}
            active_set = self._active_head_set()
            if "oft" in active_set:
                queries = gather_action_token_embeddings(last_hidden, inputs["input_ids"], self.action_token_id, self.chunk_len)
                pred = self.heads["oft"].predict_action(queries).float()
                losses["oft"] = masked_wrap_aware_l1(pred, target.float(), active, periodic, period)

            for name in ("gr00t", "pi"):
                if name not in active_set:
                    continue
                r = self.repeated_diffusion_steps[name]
                cond = (
                    [h.repeat(r, 1, 1) for h in pi_embs] if name == "pi" else last_hidden.repeat(r, 1, 1)
                )
                out = flow_matching_loss(
                    self.heads[name],
                    cond,
                    target.repeat(r, 1, 1),
                    layerwise=(name == "pi"),
                    encoder_attention_mask=None if encoder_mask is None else encoder_mask.repeat(r, 1),
                    active_mask=None if active is None else active.repeat(r, 1, 1),
                    periodic_mask=None if periodic is None else periodic.repeat(r, 1, 1),
                    wrap_weight=wrap_weight,
                    period=period,
                )
                losses[name] = out["loss"]

            action_loss = sum(self.head_weights[name] * loss for name, loss in losses.items())

        result: Dict[str, torch.Tensor] = {"action_loss": action_loss}
        for name in HEAD_NAMES:
            result[LOSS_KEYS[name]] = losses.get(name, torch.zeros((), device=device, dtype=torch.float32))
        return result

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, head: Optional[str] = None, robot_tag: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if not isinstance(examples, list):
            examples = [examples]
        head = head or self.predict_head
        if head not in self.heads:
            raise ValueError(f"head {head!r} is not enabled; available heads: {list(self.heads)}")

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = self._prepare_instructions(examples)
        train_obs_image_size = _select(self.config, "datasets.vla_data.obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        inputs, last_hidden, pi_embs = self._encode(batch_images, instructions)
        encoder_mask = self._encoder_mask(inputs)
        with _autocast(torch.float32):
            if head == "oft":
                queries = gather_action_token_embeddings(last_hidden, inputs["input_ids"], self.action_token_id, self.chunk_len)
                pred = self.heads["oft"].predict_action(queries)
            elif head == "gr00t":
                pred = self.heads["gr00t"].predict_action(last_hidden, None, encoder_attention_mask=encoder_mask)
            else:
                pred = self.heads["pi"].predict_action(pi_embs, None, encoder_attention_mask=encoder_mask)

        normalized_actions = pred.detach().float().cpu().numpy()
        if robot_tag is not None:
            if self.layout is None:
                raise ValueError("robot_tag given but framework.unified_layout is disabled")
            normalized_actions = self.layout.from_unified(normalized_actions, robot_tag)
        return {"normalized_actions": normalized_actions, "head": head}


def _vlm_dims(vlm: nn.Module) -> Tuple[int, int]:
    hf_cfg = vlm.model.config
    text_cfg = getattr(hf_cfg, "text_config", None) or hf_cfg
    hidden = getattr(hf_cfg, "hidden_size", None) or getattr(text_cfg, "hidden_size")
    return int(hidden), int(text_cfg.num_hidden_layers)


def _check_fm_head(name: str, head: nn.Module) -> None:
    missing = [attr for attr in _FM_HEAD_ATTRS if not hasattr(head, attr)]
    if missing:
        raise TypeError(f"head {name!r} lacks {missing}; flow_matching_loss needs a StarVLA-style flow-matching head")
