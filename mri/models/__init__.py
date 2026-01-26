"""Model registry and builders."""

from .registry import (
    create_classification_model,
    create_segmentation_model,
    register_classification_model,
    register_segmentation_model,
)

# Import builders to populate registries
from .seg import monai_models as _seg_models  # noqa: F401
from .cls import monai_models as _cls_models  # noqa: F401

__all__ = [
    "create_classification_model",
    "create_segmentation_model",
    "register_classification_model",
    "register_segmentation_model",
]
