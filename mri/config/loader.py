"""Config loading utilities for nested YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import re

import yaml


DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _coerce_numeric(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _coerce_numeric(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric(v) for v in value]
    if isinstance(value, str) and _NUMERIC_RE.match(value.strip()):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = _coerce_numeric(value)
    return base


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open() as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg: Dict[str, Any] = {}
    if DEFAULTS_PATH.exists():
        with DEFAULTS_PATH.open() as f:
            defaults = yaml.safe_load(f) or {}
        cfg = _deep_update(cfg, defaults)

    cfg = _deep_update(cfg, user_cfg)
    return cfg
