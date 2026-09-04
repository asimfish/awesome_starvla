"""Freeze-rule parser for StarVLA ``trainer.freeze_modules`` (VLAct recipe (a)).

StarVLA's ``TrainerUtils.freeze_backbones`` and ``build_param_lr_groups`` only accept exact
dotted module paths. This module adds three more forms; rules are comma-separated (or a list):

    qwen_vl_interface.model.model.visual                      exact module path (unchanged)
    re:^qwen_vl_interface\\.model\\.model\\.visual\\.        regex, matched against parameter names
    qwen_vl_interface.model.model.language_model.layers[0:18] index range of an nn.ModuleList
    qwen_vl_interface...layers[0:18].mlp                      range + sub-path
    llm_layers_below:18                                       sugar for ``<llm_layers_path>[0:18]``

Default ``llm_layers_path`` matches Qwen2.5-VL / Qwen3-VL under StarVLA's ``qwen_vl_interface``
(transformers >= 4.52 layout: ``model.model.language_model.layers``); verify with ``print(model)``.

Two ways to use it inside StarVLA without editing its source:
    * ``expand_to_exact_paths(model, rules)`` turns exact/range/sugar rules into the plain
      comma-separated path list the unmodified trainer understands (regex is not expressible).
    * ``install_into_starvla(train_module)`` monkeypatches ``TrainerUtils.freeze_backbones`` and
      the ``build_param_lr_groups`` name used by the training script, so all forms and the
      ``trainer.freeze_llm_layers_below`` key work directly from YAML.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch.nn as nn

DEFAULT_LLM_LAYERS_PATH = "qwen_vl_interface.model.model.language_model.layers"
DEFAULT_VISUAL_PATH = "qwen_vl_interface.model.model.visual"
DEFAULT_EMBED_PATH = "qwen_vl_interface.model.model.language_model.embed_tokens"

RuleSpec = Union[str, Sequence[str], None]

_RANGE_RE = re.compile(r"^(?P<base>[^\[\]]+)\[(?P<lo>-?\d*):(?P<hi>-?\d*)\](?P<suffix>(?:\.[^\[\]]+)?)$")
_SUGAR_RE = re.compile(r"^llm_layers_below\s*[:=]\s*(?P<n>\d+)$")


@dataclass(frozen=True)
class FreezeRule:
    kind: str  # "exact" | "regex" | "range"
    raw: str
    path: str = ""
    pattern: Optional["re.Pattern[str]"] = None
    start: Optional[int] = None
    stop: Optional[int] = None
    suffix: str = ""


@dataclass
class FreezeReport:
    frozen: List[str]
    unmatched: List[str]

    @property
    def num_frozen(self) -> int:
        return len(self.frozen)


def split_rules(spec: RuleSpec) -> List[str]:
    """Normalise a comma-separated string / list / None into a list of non-empty tokens."""
    if spec is None or spec is False:
        return []
    if isinstance(spec, str):
        return [tok.strip() for tok in spec.split(",") if tok.strip()]
    if isinstance(spec, (list, tuple)):
        out: List[str] = []
        for item in spec:
            out.extend(split_rules(item))
        return out
    raise TypeError(f"freeze rules must be a string or a list of strings, got {type(spec).__name__}")


def freeze_llm_layers_below(n: int, llm_layers_path: str = DEFAULT_LLM_LAYERS_PATH) -> str:
    """Rule string freezing LLM decoder layers ``0 .. n-1``."""
    return f"{llm_layers_path}[0:{int(n)}]"


def parse_rule(token: str, llm_layers_path: str = DEFAULT_LLM_LAYERS_PATH) -> FreezeRule:
    token = token.strip()
    sugar = _SUGAR_RE.match(token)
    if sugar:
        token = freeze_llm_layers_below(int(sugar.group("n")), llm_layers_path)
    if token.startswith("re:"):
        return FreezeRule(kind="regex", raw=token, pattern=re.compile(token[3:]))
    m = _RANGE_RE.match(token)
    if m:
        lo, hi = m.group("lo"), m.group("hi")
        return FreezeRule(
            kind="range",
            raw=token,
            path=m.group("base").strip(),
            start=int(lo) if lo else None,
            stop=int(hi) if hi else None,
            suffix=m.group("suffix").lstrip("."),
        )
    if "[" in token or "]" in token:
        raise ValueError(f"malformed range rule {token!r}; expected `path[lo:hi]` or `path[lo:hi].sub.path`")
    return FreezeRule(kind="exact", raw=token, path=token)


def parse_rules(
    spec: RuleSpec,
    freeze_llm_layers_below_n: Optional[int] = None,
    llm_layers_path: str = DEFAULT_LLM_LAYERS_PATH,
) -> List[FreezeRule]:
    tokens = split_rules(spec)
    if freeze_llm_layers_below_n is not None and int(freeze_llm_layers_below_n) > 0:
        tokens.append(freeze_llm_layers_below(int(freeze_llm_layers_below_n), llm_layers_path))
    return [parse_rule(tok, llm_layers_path) for tok in tokens]


def get_submodule(model: nn.Module, path: str) -> nn.Module:
    """``getattr`` chain that also accepts numeric ModuleList indices (same as StarVLA)."""
    module = model
    if not path:
        return module
    for attr in path.split("."):
        try:
            module = getattr(module, attr)
        except AttributeError as exc:
            raise AttributeError(f"module path {path!r} does not exist (failed at {attr!r})") from exc
    if not isinstance(module, nn.Module):
        raise AttributeError(f"{path!r} resolves to {type(module).__name__}, not an nn.Module")
    return module


def _range_indices(container: nn.Module, rule: FreezeRule) -> List[int]:
    try:
        n = len(container)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{rule.path!r} is a {type(container).__name__} without a length; range rules need nn.ModuleList/Sequential"
        ) from exc
    return list(range(n)[slice(rule.start, rule.stop)])


def _range_paths(model: nn.Module, rule: FreezeRule) -> List[str]:
    container = get_submodule(model, rule.path)
    return [f"{rule.path}.{idx}" + (f".{rule.suffix}" if rule.suffix else "") for idx in _range_indices(container, rule)]


def resolve_rule(model: nn.Module, rule: FreezeRule) -> Dict[str, nn.Parameter]:
    """Return ``{param_name: param}`` matched by one rule (empty when nothing matches)."""
    if rule.kind == "regex":
        assert rule.pattern is not None
        return {name: p for name, p in model.named_parameters() if rule.pattern.search(name)}
    paths = [rule.path] if rule.kind == "exact" else _range_paths(model, rule)
    matched: Dict[str, nn.Parameter] = {}
    for path in paths:
        prefix = f"{path}." if path else ""
        matched.update({prefix + n: p for n, p in get_submodule(model, path).named_parameters()})
    return matched


def _resolve_all(model: nn.Module, rules: List[FreezeRule], strict: bool) -> Tuple[Dict[str, nn.Parameter], List[str]]:
    frozen: Dict[str, nn.Parameter] = {}
    unmatched: List[str] = []
    for rule in rules:
        try:
            found = resolve_rule(model, rule)
        except (AttributeError, TypeError) as exc:
            if strict:
                raise
            warnings.warn(f"freeze rule {rule.raw!r} skipped: {exc}", stacklevel=3)
            unmatched.append(rule.raw)
            continue
        if not found:
            if strict:
                raise ValueError(f"freeze rule {rule.raw!r} matched no parameters")
            warnings.warn(f"freeze rule {rule.raw!r} matched no parameters", stacklevel=3)
            unmatched.append(rule.raw)
        frozen.update(found)
    return frozen, unmatched


def resolve_frozen_params(
    model: nn.Module,
    spec: RuleSpec,
    freeze_llm_layers_below_n: Optional[int] = None,
    llm_layers_path: str = DEFAULT_LLM_LAYERS_PATH,
    strict: bool = False,
) -> Dict[str, nn.Parameter]:
    """Union of parameters selected by all rules. Missing paths warn (StarVLA behaviour) unless ``strict``."""
    frozen, _ = _resolve_all(model, parse_rules(spec, freeze_llm_layers_below_n, llm_layers_path), strict)
    return frozen


def resolve_frozen_param_ids(
    model: nn.Module,
    spec: RuleSpec,
    freeze_llm_layers_below_n: Optional[int] = None,
    llm_layers_path: str = DEFAULT_LLM_LAYERS_PATH,
    strict: bool = False,
) -> Set[int]:
    return {id(p) for p in resolve_frozen_params(model, spec, freeze_llm_layers_below_n, llm_layers_path, strict).values()}


def freeze_by_rules(
    model: nn.Module,
    spec: RuleSpec,
    freeze_llm_layers_below_n: Optional[int] = None,
    llm_layers_path: str = DEFAULT_LLM_LAYERS_PATH,
    strict: bool = False,
) -> FreezeReport:
    """Set ``requires_grad=False`` on every parameter selected by the rules."""
    frozen, unmatched = _resolve_all(model, parse_rules(spec, freeze_llm_layers_below_n, llm_layers_path), strict)
    for p in frozen.values():
        p.requires_grad = False
    return FreezeReport(frozen=sorted(frozen), unmatched=unmatched)


def expand_to_exact_paths(
    model: nn.Module,
    spec: RuleSpec,
    freeze_llm_layers_below_n: Optional[int] = None,
    llm_layers_path: str = DEFAULT_LLM_LAYERS_PATH,
) -> str:
    """Expand exact/range/sugar rules into the comma-separated exact-path string unmodified StarVLA accepts.

    Regex rules select parameters, not modules, and cannot be expressed as module paths; they raise.
    """
    paths: List[str] = []
    for rule in parse_rules(spec, freeze_llm_layers_below_n, llm_layers_path):
        if rule.kind == "regex":
            raise ValueError(f"regex rule {rule.raw!r} cannot be expanded to exact module paths; use install_into_starvla()")
        if rule.kind == "exact":
            get_submodule(model, rule.path)
            paths.append(rule.path)
        else:
            paths.extend(_range_paths(model, rule))
    return ",".join(dict.fromkeys(paths))


# --------------------------------------------------------------------------------------
# Drop-in replacements for StarVLA trainer utilities (same signatures).
# --------------------------------------------------------------------------------------


def _cfg_get(node, key: str, default=None):
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            value = getter(key, default)
            return default if value is None else value
        except Exception:
            pass
    return getattr(node, key, default)


def freeze_backbones(model: nn.Module, freeze_modules: RuleSpec = "") -> nn.Module:
    """Replacement for ``TrainerUtils.freeze_backbones`` understanding all rule forms."""
    if freeze_modules and not isinstance(freeze_modules, (str, list, tuple)):
        # StarVLA receives a bool when the CLI flag is passed without a value; keep its no-op behaviour.
        return model
    report = freeze_by_rules(model, freeze_modules)
    print(f"[vlact_ext] frozen {report.num_frozen} parameter tensors; rules={split_rules(freeze_modules)}; unmatched={report.unmatched}")
    return model


def build_param_lr_groups(model: nn.Module, cfg) -> List[dict]:
    """Replacement for ``trainer_tools.build_param_lr_groups`` that shares the rule parser.

    Reads ``cfg.trainer.freeze_modules`` (any rule form), ``cfg.trainer.freeze_llm_layers_below`` and
    ``cfg.trainer.llm_layers_path``. The sugar key is folded into ``cfg.trainer.freeze_modules`` so the
    later ``freeze_backbones(model, cfg.trainer.freeze_modules)`` call freezes the same parameters.
    """
    trainer_cfg = cfg.trainer
    lr_cfg = trainer_cfg.learning_rate
    base_lr = _cfg_get(lr_cfg, "base", 1e-4)

    freeze_modules = _cfg_get(trainer_cfg, "freeze_modules", "")
    if not isinstance(freeze_modules, (str, list, tuple)):
        freeze_modules = ""
    llm_layers_path = _cfg_get(trainer_cfg, "llm_layers_path", DEFAULT_LLM_LAYERS_PATH)
    below = _cfg_get(trainer_cfg, "freeze_llm_layers_below", None)
    if below is not None and int(below) > 0:
        tokens = split_rules(freeze_modules)
        sugar = freeze_llm_layers_below(int(below), llm_layers_path)
        if sugar not in tokens:
            tokens.append(sugar)
        freeze_modules = ",".join(tokens)
        try:
            trainer_cfg.freeze_modules = freeze_modules
        except Exception:
            pass

    frozen_ids = resolve_frozen_param_ids(model, freeze_modules, llm_layers_path=llm_layers_path)

    used: Set[int] = set()
    groups: List[dict] = []
    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        try:
            module = get_submodule(model, module_name)
        except AttributeError:
            warnings.warn(f"learning_rate group path {module_name!r} not found in model; skipped", stacklevel=2)
            continue
        params = [p for p in module.parameters() if id(p) not in frozen_ids and id(p) not in used]
        if params:
            groups.append({"params": params, "lr": lr, "name": module_name})
            used.update(id(p) for p in params)
    rest = [p for p in model.parameters() if id(p) not in used and id(p) not in frozen_ids]
    if rest:
        groups.append({"params": rest, "lr": base_lr, "name": "base"})
    return groups


def install_into_starvla(*train_modules) -> None:
    """Monkeypatch StarVLA so YAML freeze rules may use every form of this module.

    Patches ``TrainerUtils.freeze_backbones`` and ``trainer_tools.build_param_lr_groups``. Training
    scripts import ``build_param_lr_groups`` by name, so pass the script module too, e.g.::

        import starVLA.training.train_starvla as train_mod
        install_into_starvla(train_mod)
    """
    from starVLA.training.trainer_utils import trainer_tools

    trainer_tools.TrainerUtils.freeze_backbones = staticmethod(freeze_backbones)
    trainer_tools.build_param_lr_groups = build_param_lr_groups
    for mod in train_modules:
        if hasattr(mod, "build_param_lr_groups"):
            mod.build_param_lr_groups = build_param_lr_groups
