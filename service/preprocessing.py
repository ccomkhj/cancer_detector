#!/usr/bin/env python3
"""
Shared preprocessing utilities for multimodal MRI segmentation.

This module centralizes the preprocessing settings used by both
training (`service/train.py`) and inference (`service/inference.py`)
so both paths build inputs in the same way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from tools.dataset.dataset_multimodal import MultiModalDataset

DEFAULT_MULTIMODAL_METADATA = "data/aligned_v2/metadata.json"


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value_norm = value.strip().lower()
        if value_norm in {"1", "true", "yes", "y", "on"}:
            return True
        if value_norm in {"0", "false", "no", "n", "off"}:
            return False
    return default


@dataclass(frozen=True)
class MultiModalPreprocessingConfig:
    metadata_path: str = DEFAULT_MULTIMODAL_METADATA
    stack_depth: int = 5
    normalize: bool = True
    require_complete: bool = False
    require_positive: bool = False


def build_multimodal_preprocessing_config(
    config: Mapping[str, Any],
    *,
    metadata_override: Optional[str] = None,
) -> MultiModalPreprocessingConfig:
    """Build typed preprocessing config from args/checkpoint/config mapping."""
    metadata_path = metadata_override or config.get("metadata") or DEFAULT_MULTIMODAL_METADATA

    stack_depth_raw = config.get("stack_depth", 5)
    stack_depth = int(stack_depth_raw) if stack_depth_raw is not None else 5

    normalize = _as_bool(config.get("normalize"), default=True)
    require_complete = _as_bool(config.get("require_complete"), default=False)
    require_positive = _as_bool(config.get("require_positive"), default=False)

    return MultiModalPreprocessingConfig(
        metadata_path=str(metadata_path),
        stack_depth=stack_depth,
        normalize=normalize,
        require_complete=require_complete,
        require_positive=require_positive,
    )


def create_multimodal_dataset(
    preprocessing: MultiModalPreprocessingConfig,
) -> MultiModalDataset:
    """Create `MultiModalDataset` with shared preprocessing settings."""
    return MultiModalDataset(
        metadata_path=preprocessing.metadata_path,
        stack_depth=preprocessing.stack_depth,
        normalize=preprocessing.normalize,
        require_complete=preprocessing.require_complete,
        require_positive=preprocessing.require_positive,
    )


def infer_class_from_manifest_path(manifest_path: Path) -> Optional[int]:
    """Extract class id from manifest path (e.g., class4/manifest.csv -> 4)."""
    match = re.search(r"class(\d+)", str(manifest_path))
    if not match:
        return None
    return int(match.group(1))


def _parse_case_number(case_token: str) -> Optional[int]:
    token = str(case_token).strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    match = re.search(r"case[_-]?(\d+)", token, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _case_number_from_sample_case_id(sample_case_id: str) -> Optional[int]:
    match = re.search(r"case_(\d+)", str(sample_case_id))
    if not match:
        return None
    return int(match.group(1))


def select_multimodal_sample_indices(
    samples: Sequence[Dict[str, Any]],
    *,
    manifest_path: Path,
    case_ids: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[int]] = None,
) -> list[int]:
    """
    Select multimodal sample indices using the same class/case filtering
    logic used by inference.
    """
    manifest_class = infer_class_from_manifest_path(manifest_path)
    class_filter = set(classes or [])
    if manifest_class is not None:
        class_filter.add(manifest_class)

    raw_case_filters = {str(c) for c in (case_ids or [])}
    num_case_filters = {
        n for n in (_parse_case_number(c) for c in (case_ids or [])) if n is not None
    }

    selected_indices: list[int] = []
    for i, sample in enumerate(samples):
        sample_class = sample.get("class")
        if class_filter and sample_class not in class_filter:
            continue

        if raw_case_filters or num_case_filters:
            sample_case = str(sample.get("case_id", ""))
            sample_case_num = _case_number_from_sample_case_id(sample_case)
            if sample_case not in raw_case_filters and (
                sample_case_num is None or sample_case_num not in num_case_filters
            ):
                continue

        selected_indices.append(i)

    return selected_indices
