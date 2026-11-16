"""Configuration loading: configs/default.yaml, optionally overridden by a smaller YAML file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the default config, then deep-merge `path` on top of it (only changed keys needed)."""
    config = _read(DEFAULT_CONFIG)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        config = deep_update(config, _read(Path(path)))
    return config


def deep_update(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(path: str | Path) -> Path:
    """Relative paths in configs are relative to the project root, not to the current directory."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
