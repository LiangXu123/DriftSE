"""
Composable JSON config loader with __base__ inheritance.

Usage (drop-in replacement for json.load):
    from config_loader import load_config
    cfg = load_config("config/experiments/SE/SE_EARS_distillhubert_tfgrid_causal_incond.json")

Merge semantics:
  - __base__ is a list of paths (relative to the config file being loaded).
  - Bases are merged left-to-right; later entries override earlier ones.
  - The experiment file itself overrides everything.
  - Dict values are merged recursively; all other types are overwritten.
"""

import json
from pathlib import Path


def _deep_merge(base: dict, override: dict) -> None:
    """Merge override into base in-place. Dicts are merged recursively."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _load_and_resolve(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)

    bases = cfg.pop("__base__", [])
    if isinstance(bases, str):
        bases = [bases]

    merged = {}
    for base_rel in bases:
        base_path = (config_path.parent / base_rel).resolve()
        base_cfg = _load_and_resolve(base_path)  # recursive — supports nested __base__
        _deep_merge(merged, base_cfg)

    _deep_merge(merged, cfg)  # experiment overrides win
    return merged


def load_config(config_path: str) -> dict:
    """Load a (possibly composed) JSON config, resolving all __base__ references."""
    return _load_and_resolve(Path(config_path).resolve())
