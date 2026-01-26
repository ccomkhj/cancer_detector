"""Dataset implementations."""

from .segmentation import SegmentationDataset
from .classification import ClassificationDataset

__all__ = ["SegmentationDataset", "ClassificationDataset"]
