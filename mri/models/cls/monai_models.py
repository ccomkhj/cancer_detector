"""MONAI classification model builders (3D)."""

from __future__ import annotations

from typing import Any

from mri.models.registry import register_classification_model

try:
    from monai.networks.nets import SwinTransformer, ViT, DenseNet, EfficientNetBN
    from monai.networks.nets import resnet as monai_resnet
except Exception:  # pragma: no cover - optional dependency
    SwinTransformer = ViT = DenseNet = EfficientNetBN = None
    monai_resnet = None


@register_classification_model("swin")
def build_swin(**kwargs: Any):
    if SwinTransformer is None:
        raise ImportError("MONAI not installed")
    return SwinTransformer(**kwargs)


@register_classification_model("vit")
def build_vit(**kwargs: Any):
    if ViT is None:
        raise ImportError("MONAI not installed")
    return ViT(**kwargs)


@register_classification_model("resnet101")
def build_resnet101(**kwargs: Any):
    if monai_resnet is None:
        raise ImportError("MONAI not installed")
    if hasattr(monai_resnet, "resnet101"):
        return monai_resnet.resnet101(**kwargs)
    raise AttributeError("MONAI resnet101 not available")


@register_classification_model("resnext101")
def build_resnext101(**kwargs: Any):
    if monai_resnet is None:
        raise ImportError("MONAI not installed")
    fn = getattr(monai_resnet, "resnext101_32x8d", None) or getattr(monai_resnet, "resnext101", None)
    if fn is None:
        raise AttributeError("MONAI resnext101 not available")
    return fn(**kwargs)


@register_classification_model("densenet121")
def build_densenet121(**kwargs: Any):
    if DenseNet is None:
        raise ImportError("MONAI not installed")
    return DenseNet(**kwargs)


@register_classification_model("efficientnetb7")
def build_efficientnetb7(**kwargs: Any):
    if EfficientNetBN is None:
        raise ImportError("MONAI not installed")
    kwargs.setdefault("model_name", "efficientnet-b7")
    return EfficientNetBN(**kwargs)
