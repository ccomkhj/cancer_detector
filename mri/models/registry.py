"""Model registry for segmentation and classification models."""

from __future__ import annotations

from typing import Callable, Dict

SEGMENTATION_MODELS: Dict[str, Callable] = {}
CLASSIFICATION_MODELS: Dict[str, Callable] = {}


def register_segmentation_model(name: str):
    def decorator(fn: Callable):
        SEGMENTATION_MODELS[name] = fn
        return fn
    return decorator


def register_classification_model(name: str):
    def decorator(fn: Callable):
        CLASSIFICATION_MODELS[name] = fn
        return fn
    return decorator


def create_segmentation_model(name: str, **kwargs):
    if name not in SEGMENTATION_MODELS:
        raise KeyError(f"Unknown segmentation model: {name}")
    return SEGMENTATION_MODELS[name](**kwargs)


def create_classification_model(name: str, **kwargs):
    if name not in CLASSIFICATION_MODELS:
        raise KeyError(f"Unknown classification model: {name}")
    return CLASSIFICATION_MODELS[name](**kwargs)
