"""Transform registry."""

from __future__ import annotations

from typing import Callable, Dict

TRANSFORMS: Dict[str, Callable] = {}


def register_transform(name: str):
    def decorator(fn: Callable):
        TRANSFORMS[name] = fn
        return fn
    return decorator


def get_transform(name: str):
    if name in (None, "none"):
        return None
    if name not in TRANSFORMS:
        raise KeyError(f"Unknown transform: {name}")
    return TRANSFORMS[name]
