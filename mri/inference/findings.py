"""Shared per-lesion findings model for structured reporting and DICOM-SR.

Deterministic, pure NumPy/scipy. ``build_findings`` turns a postprocessed
lesion mask (plus the gland mask, lesion probability, and voxel spacing) into
an ordered list of :class:`Finding` records. The same records feed the
PI-RADS-style report today (shortlist #2) and DICOM-SR export later
(shortlist #1), so the report and the SR can never disagree on a lesion's
size or suspicion.

Lesion components are labelled with the same routine the evaluation path uses
(:func:`mri.diagnostics.detection.label_lesion_components`), so a "lesion"
means the same thing here and in the metrics.

The ``zone`` and ``suspicion`` fields are coarse, documented approximations —
decision support, **not** a certified PI-RADS category or an anatomical zonal
segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mri.diagnostics.detection import label_lesion_components


# Default suspicion bands over the model's max lesion probability. These are a
# display convenience, not a regulatory claim.
DEFAULT_SUSPICION_THRESHOLDS = (0.34, 0.67)


@dataclass(frozen=True)
class Finding:
    """One suspicious focus (one connected lesion component).

    ``max_diameter_mm`` and ``volume_mm3`` are ``None`` when voxel spacing is
    unavailable; ``voxels`` (the raw voxel count) is always populated.
    """
    lesion_id: int                        # stable per-case component identifier
    zone: str                             # "PZ" | "TZ" | "other" (coarse heuristic)
    slice_range: tuple[int, int]          # (first_z, last_z) the lesion spans
    n_slices: int
    voxels: int
    max_probability: float
    suspicion: str                        # "Low" | "Intermediate" | "High"
    centroid: tuple[float, float, float]  # (z, y, x)
    max_diameter_mm: float | None
    volume_mm3: float | None


def suspicion_band(
    max_probability: float,
    thresholds: Sequence[float] = DEFAULT_SUSPICION_THRESHOLDS,
) -> str:
    """Map a max lesion probability to a coarse Low/Intermediate/High band.

    Boundaries fall into the upper band: ``p >= thresholds[1]`` is High,
    ``p >= thresholds[0]`` is Intermediate, otherwise Low.
    """
    low, high = thresholds
    if max_probability >= high:
        return "High"
    if max_probability >= low:
        return "Intermediate"
    return "Low"


def _zone_from_centroid(
    centroid_row: float, gland_mask: np.ndarray,
) -> str:
    """Coarse peripheral-vs-transition zone from the lesion centroid row.

    Peripheral zone (PZ) is posterior, transition zone (TZ) is central-
    anterior. Using the standard axial display convention (posterior at the
    bottom, i.e. larger row index), a lesion centroid posterior to the gland
    centroid is called PZ, otherwise TZ. When no gland is present the zone is
    "other". Documented as an approximation, not an anatomical segmentation.
    """
    if not gland_mask.any():
        return "other"
    gland_rows = np.argwhere(gland_mask.astype(bool))[:, 1]
    gland_centroid_row = float(gland_rows.mean())
    return "PZ" if centroid_row >= gland_centroid_row else "TZ"


def _max_in_plane_diameter_mm(
    component: np.ndarray, spacing_y: float, spacing_x: float,
) -> float:
    """Largest in-plane bounding-box extent in mm across the slices spanned."""
    max_mm = 0.0
    slice_has = component.any(axis=(1, 2))
    for z in np.flatnonzero(slice_has):
        rows = np.flatnonzero(component[z].any(axis=1))
        cols = np.flatnonzero(component[z].any(axis=0))
        row_extent_mm = (int(rows[-1]) - int(rows[0]) + 1) * spacing_y
        col_extent_mm = (int(cols[-1]) - int(cols[0]) + 1) * spacing_x
        max_mm = max(max_mm, row_extent_mm, col_extent_mm)
    return float(max_mm)


def build_findings(
    lesion_mask: np.ndarray,
    gland_mask: np.ndarray,
    lesion_prob: np.ndarray,
    *,
    voxel_spacing_mm: tuple[float, float, float] | None,
    connectivity_rank: int = 1,
    suspicion_thresholds: Sequence[float] = DEFAULT_SUSPICION_THRESHOLDS,
) -> list[Finding]:
    """Build ordered per-lesion findings from a postprocessed lesion mask.

    Args:
      lesion_mask: (Z, H, W) postprocessed lesion voxels.
      gland_mask:  (Z, H, W) postprocessed gland voxels (for the zone heuristic).
      lesion_prob: (Z, H, W) lesion probability volume.
      voxel_spacing_mm: (z, y, x) spacing in mm, or ``None``. When ``None``,
          ``max_diameter_mm`` and ``volume_mm3`` are omitted (left ``None``).
      connectivity_rank: component connectivity, as in
          :func:`label_lesion_components`.
      suspicion_thresholds: (low, high) probability cutoffs for the band.

    Returns:
      Findings ordered most-suspicious first (descending max probability, then
      ascending ``lesion_id``). Empty when the lesion mask has no components.
    """
    assert lesion_mask.shape == gland_mask.shape == lesion_prob.shape, (
        f"shape mismatch: lesion {lesion_mask.shape}, gland {gland_mask.shape}, "
        f"prob {lesion_prob.shape}"
    )

    labels, n = label_lesion_components(
        lesion_mask, connectivity_rank=connectivity_rank,
    )
    if n == 0:
        return []

    if voxel_spacing_mm is not None:
        spacing_z, spacing_y, spacing_x = voxel_spacing_mm
        voxel_volume_mm3 = spacing_z * spacing_y * spacing_x
    else:
        spacing_y = spacing_x = voxel_volume_mm3 = None  # type: ignore[assignment]

    findings: list[Finding] = []
    for k in range(1, n + 1):
        component = labels == k
        coords = np.argwhere(component)
        zs = coords[:, 0]
        centroid = (
            float(coords[:, 0].mean()),
            float(coords[:, 1].mean()),
            float(coords[:, 2].mean()),
        )
        max_prob = float(lesion_prob[component].max())
        voxels = int(component.sum())

        if voxel_spacing_mm is not None:
            max_diameter_mm = _max_in_plane_diameter_mm(
                component, spacing_y, spacing_x,
            )
            volume_mm3 = float(voxels * voxel_volume_mm3)
        else:
            max_diameter_mm = None
            volume_mm3 = None

        findings.append(Finding(
            lesion_id=k,
            zone=_zone_from_centroid(centroid[1], gland_mask),
            slice_range=(int(zs.min()), int(zs.max())),
            n_slices=int(np.unique(zs).size),
            voxels=voxels,
            max_probability=max_prob,
            suspicion=suspicion_band(max_prob, suspicion_thresholds),
            centroid=centroid,
            max_diameter_mm=max_diameter_mm,
            volume_mm3=volume_mm3,
        ))

    findings.sort(key=lambda f: (-f.max_probability, f.lesion_id))
    return findings


def build_impression(findings: Sequence[Finding]) -> str:
    """Deterministic one-line study impression derived from the findings list.

    Negative cases yield an explicit "no suspicious focus" impression.
    """
    if not findings:
        return "No suspicious focus identified."

    n = len(findings)
    focus_word = "focus" if n == 1 else "foci"

    # The "largest" focus drives the headline: by diameter when spacing is
    # available, otherwise by voxel count.
    largest = max(
        findings,
        key=lambda f: (f.max_diameter_mm if f.max_diameter_mm is not None else f.voxels),
    )
    highest_suspicion = _highest_band(findings)

    if largest.max_diameter_mm is not None:
        size_clause = f"largest {largest.max_diameter_mm:.0f} mm"
    else:
        size_clause = f"largest {largest.voxels} voxels"

    return (
        f"{n} suspicious {focus_word}, {size_clause}, {largest.zone} "
        f"({highest_suspicion} suspicion)."
    )


_BAND_ORDER = {"Low": 0, "Intermediate": 1, "High": 2}


def _highest_band(findings: Sequence[Finding]) -> str:
    return max((f.suspicion for f in findings), key=lambda b: _BAND_ORDER[b])
