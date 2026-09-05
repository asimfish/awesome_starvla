"""Ad-hoc LeRobot mixtures for probe batches without editing StarVLA's mixture registry.

StarVLA resolves ``datasets.vla_data.data_mix`` through the plain dict
``starVLA.dataloader.gr00t_lerobot.mixtures.DATASET_NAMED_MIXTURES`` (``name -> [(dataset_dir, weight,
robot_type), ...]``). :func:`register_mixture` accepts either an existing mixture name or an inline spec and
returns a name that the registry resolves::

    libero_goal                                                         # existing mixture, returned unchanged
    libero_goal_no_noops_1.0.0_lerobot:libero_franka,libero_spatial_no_noops_1.0.0_lerobot:libero_franka
    libero_goal_no_noops_1.0.0_lerobot:libero_franka:2.0                # optional third field = sampling weight
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, MutableMapping, Tuple

__all__ = ["parse_mixture_spec", "register_mixture"]

MixtureSpec = List[Tuple[str, float, str]]


def parse_mixture_spec(spec: str) -> MixtureSpec:
    """Parse ``dir:robot[:weight],...`` into StarVLA's ``[(dataset_dir, weight, robot_type)]`` list."""
    out: MixtureSpec = []
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
            raise ValueError(f"mixture item {item!r} must be 'dataset_dir:robot_type[:weight]'")
        weight = float(parts[2]) if len(parts) == 3 else 1.0
        if weight <= 0:
            raise ValueError(f"mixture item {item!r}: weight must be positive")
        out.append((parts[0], weight, parts[1]))
    if not out:
        raise ValueError("empty mixture spec")
    return out


def register_mixture(spec: str, registry: MutableMapping[str, MixtureSpec], prefix: str = "lab_probe") -> str:
    """Return a registry key for ``spec``: the name itself when it is already registered, otherwise a new
    ``<prefix>_<hash>`` entry holding the parsed inline spec (idempotent for the same spec)."""
    spec = spec.strip()
    if spec in registry:
        return spec
    parsed = parse_mixture_spec(spec)
    name = f"{prefix}_{hashlib.sha1(spec.encode('utf-8')).hexdigest()[:8]}"
    registry[name] = parsed
    return name
