"""Unit tests for the shared per-lesion findings model (shortlist #2)."""

from __future__ import annotations

import numpy as np
import pytest

from mri.inference.findings import (
    Finding,
    build_findings,
    suspicion_band,
    build_impression,
)


def _volume(shape: tuple[int, int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def test_build_findings_geometry_with_spacing() -> None:
    shape = (5, 8, 8)
    lesion = _volume(shape)
    lesion[1:3, 2:4, 1:5] = 1  # z 1-2, rows 2-3, cols 1-4
    gland = np.ones(shape, dtype=np.uint8)
    prob = np.zeros(shape, dtype=np.float32)
    prob[lesion.astype(bool)] = 0.8

    findings = build_findings(
        lesion, gland, prob,
        voxel_spacing_mm=(3.0, 1.0, 1.0),  # (z, y, x)
    )

    assert len(findings) == 1
    f = findings[0]
    assert f.slice_range == (1, 2)
    assert f.n_slices == 2
    assert f.voxels == 2 * 2 * 4
    assert f.volume_mm3 == 48.0            # 16 voxels * (3*1*1)
    assert f.max_diameter_mm == 4.0        # 4 cols * 1mm > 2 rows * 1mm
    assert f.max_probability == pytest.approx(0.8)
    assert f.suspicion == "High"


def test_build_findings_omits_mm_when_spacing_missing() -> None:
    shape = (3, 8, 8)
    lesion = _volume(shape)
    lesion[1, 2:4, 2:4] = 1
    gland = np.ones(shape, dtype=np.uint8)
    prob = np.full(shape, 0.5, dtype=np.float32)

    findings = build_findings(lesion, gland, prob, voxel_spacing_mm=None)

    assert len(findings) == 1
    f = findings[0]
    assert f.max_diameter_mm is None
    assert f.volume_mm3 is None
    assert f.voxels == 4  # raw voxel count still available


def test_zone_posterior_is_peripheral_anterior_is_transition() -> None:
    shape = (3, 8, 8)
    gland = np.zeros(shape, dtype=np.uint8)
    gland[1, :, :] = 1  # full-slice gland, centroid row = 3.5
    prob = np.full(shape, 0.9, dtype=np.float32)

    posterior = _volume(shape)
    posterior[1, 5:7, 3:5] = 1  # rows 5-6, behind the gland centroid
    pz = build_findings(posterior, gland, prob, voxel_spacing_mm=None)
    assert pz[0].zone == "PZ"

    anterior = _volume(shape)
    anterior[1, 1:3, 3:5] = 1  # rows 1-2, in front of the gland centroid
    tz = build_findings(anterior, gland, prob, voxel_spacing_mm=None)
    assert tz[0].zone == "TZ"


def test_zone_other_when_no_gland() -> None:
    shape = (3, 8, 8)
    lesion = _volume(shape)
    lesion[1, 3:5, 3:5] = 1
    gland = np.zeros(shape, dtype=np.uint8)
    prob = np.full(shape, 0.9, dtype=np.float32)

    findings = build_findings(lesion, gland, prob, voxel_spacing_mm=None)

    assert findings[0].zone == "other"


def test_suspicion_band_thresholds() -> None:
    assert suspicion_band(0.2) == "Low"
    assert suspicion_band(0.5) == "Intermediate"
    assert suspicion_band(0.8) == "High"
    # Boundary values fall into the upper band (>=).
    assert suspicion_band(0.34) == "Intermediate"
    assert suspicion_band(0.67) == "High"


def test_empty_lesion_mask_yields_no_findings() -> None:
    shape = (3, 8, 8)
    lesion = _volume(shape)
    gland = np.ones(shape, dtype=np.uint8)
    prob = np.zeros(shape, dtype=np.float32)

    assert build_findings(lesion, gland, prob, voxel_spacing_mm=None) == []


def test_multiple_components_ordered_most_suspicious_first() -> None:
    shape = (3, 10, 10)
    lesion = _volume(shape)
    lesion[1, 1:3, 1:3] = 1   # component A
    lesion[1, 7:9, 7:9] = 1   # component B (disjoint)
    gland = np.ones(shape, dtype=np.uint8)
    prob = np.zeros(shape, dtype=np.float32)
    prob[1, 1:3, 1:3] = 0.4   # A less suspicious
    prob[1, 7:9, 7:9] = 0.9   # B more suspicious

    findings = build_findings(lesion, gland, prob, voxel_spacing_mm=None)

    assert len(findings) == 2
    assert findings[0].max_probability == pytest.approx(0.9)   # most suspicious first
    assert findings[1].max_probability == pytest.approx(0.4)
    # lesion_id is a stable per-case identifier (distinct per component).
    assert {f.lesion_id for f in findings} == {1, 2}


def test_build_impression_empty_is_no_suspicious_focus() -> None:
    assert build_impression([]) == "No suspicious focus identified."


def test_build_impression_summarises_findings() -> None:
    findings = [
        Finding(lesion_id=1, zone="PZ", slice_range=(3, 5), n_slices=3,
                voxels=40, max_probability=0.9, suspicion="High",
                centroid=(4.0, 6.0, 5.0), max_diameter_mm=14.0, volume_mm3=120.0),
        Finding(lesion_id=2, zone="TZ", slice_range=(7, 7), n_slices=1,
                voxels=8, max_probability=0.4, suspicion="Intermediate",
                centroid=(7.0, 2.0, 3.0), max_diameter_mm=6.0, volume_mm3=20.0),
    ]

    impression = build_impression(findings)

    assert "2 suspicious foci" in impression
    assert "14" in impression        # largest diameter mm
    assert "PZ" in impression         # zone of the largest
    assert "High" in impression       # highest suspicion band
