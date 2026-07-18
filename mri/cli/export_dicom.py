"""CLI to export a finished case as DICOM-SEG + DICOM-SR (shortlist #1).

Orchestration only: the DICOM objects are built by ``dicom_mapper`` (multi-
segment SEG + measurement SR); this maps cancer_detector's findings onto the
SR measurement contract, supplies the predicted masks and the retained source
T2 datasets, and writes ``<case>_seg.dcm`` / ``<case>_sr.dcm``.

Usage::

    python -m mri.cli.export_dicom <run_dir> --case CASE \\
        --source-series <t2_source_dir> [--seg] [--sr] [--output-dir DIR]

The source series is the ordered T2 DICOMs retained by dicom_mapper's
``process-vendor --retain-source-t2`` (its ``t2_source/`` folder). SEG frames
and SR references land on the source study registered to the correct slices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from mri.cli.diagnose import resolve_run_dir
from mri.inference.findings import Finding, build_findings

from dicom_mapper.core.highdicom_creation import (
    SegmentDefinition, create_multisegment_segmentation,
)
from dicom_mapper.core.sr_creation import (
    LesionMeasurement, create_measurement_report,
)
from dicom_mapper.io.dicom import load_dicom_series


# Coded meanings for the two segments (structural; highdicom does not validate
# these against a dictionary).
# Producer identity and the same research-use disclaimer the report carries
# (mirrors mri.cli.report._DISCLAIMER); both DICOM objects carry it so their
# status is unambiguous inside PACS.
_PRODUCER = "cancer_detector"
_RESEARCH_DISCLAIMER = (
    "Research use only — not a diagnostic device. Zone and suspicion band are "
    "coarse model-derived approximations, not a certified PI-RADS assessment."
)

_GLAND_SEGMENT = SegmentDefinition(
    label_value=1, segment_label="Prostate gland",
    category_code=("123037004", "SCT", "Anatomical Structure"),
    type_code=("41216001", "SCT", "Prostate"),
)
_LESION_SEGMENT = SegmentDefinition(
    label_value=2, segment_label="Suspicious lesion",
    category_code=("49755003", "SCT", "Morphologically Abnormal Structure"),
    type_code=("108369006", "SCT", "Neoplasm"),
)


def findings_to_measurements(findings: Sequence[Finding]) -> list[LesionMeasurement]:
    """Map shared Finding records onto dicom_mapper's SR measurement contract.

    Pure. ``max_diameter_mm`` becomes the SR largest-diameter; a finding
    without millimetre size (spacing unavailable) is exported with 0.0 mm.
    The referenced slice is the finding's centroid z; the in-plane point is
    the centroid (column, row).
    """
    measurements: list[LesionMeasurement] = []
    for f in findings:
        measurements.append(LesionMeasurement(
            tracking_id=f"lesion-{f.lesion_id}",
            largest_diameter_mm=(
                f.max_diameter_mm if f.max_diameter_mm is not None else 0.0
            ),
            suspicion=f.suspicion,
            slice_index=int(round(f.centroid[0])),
            centroid_xy=(f.centroid[2], f.centroid[1]),  # (x=col, y=row)
        ))
    return measurements


def build_label_mask(gland_mask: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
    """Combine gland (1) and lesion (2) into one label map; lesion wins overlap."""
    label = np.zeros(gland_mask.shape, dtype=np.uint8)
    label[gland_mask.astype(bool)] = 1
    label[lesion_mask.astype(bool)] = 2
    return label


def align_mask_to_source(label_mask: np.ndarray, n: int, rows: int, cols: int) -> np.ndarray:
    """Nearest-neighbour resample a (Z,H,W) label map onto the source grid.

    A no-op when the mask already matches the source series in count and
    in-plane size (the common case, since the aligned volume was built on the
    native T2 grid).
    """
    z, h, w = label_mask.shape
    if (z, h, w) == (n, rows, cols):
        return label_mask
    zi = np.minimum((np.arange(n) * z // max(n, 1)), z - 1)
    ri = np.minimum((np.arange(rows) * h // max(rows, 1)), h - 1)
    ci = np.minimum((np.arange(cols) * w // max(cols, 1)), w - 1)
    return label_mask[np.ix_(zi, ri, ci)]


def _read_voxel_spacing(meta_path: Path):
    if not meta_path.exists():
        return None
    try:
        spacing = json.loads(meta_path.read_text()).get("voxel_spacing_mm")
    except (OSError, json.JSONDecodeError):
        return None
    if spacing and len(spacing) == 3:
        return (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    return None


def export_case(
    *, predictions_dir: Path, postprocessed_dir: Path, source_series: Path,
    case_id: str, output_dir: Path, write_seg: bool, write_sr: bool,
) -> list[Path]:
    """Build and write the requested DICOM objects for one case."""
    gland = np.load(postprocessed_dir / case_id / "gland_mask.npz")["mask"]
    lesion = np.load(postprocessed_dir / case_id / "lesion_mask.npz")["mask"]
    lesion_prob = np.load(predictions_dir / case_id / "prob.npz")["lesion"]
    spacing = _read_voxel_spacing(predictions_dir / case_id / "meta.json")

    source_images = load_dicom_series(source_series)
    if not source_images:
        raise SystemExit(f"[export_dicom] no source DICOMs under {source_series}.")
    rows = int(source_images[0].Rows)
    cols = int(source_images[0].Columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if write_seg:
        label = align_mask_to_source(
            build_label_mask(gland, lesion), len(source_images), rows, cols,
        )
        seg = create_multisegment_segmentation(
            source_images, label,
            segments=[_GLAND_SEGMENT, _LESION_SEGMENT],
            series_description="AI prostate segmentation",
            manufacturer=_PRODUCER,
            research_disclaimer=_RESEARCH_DISCLAIMER,
        )
        seg_path = output_dir / f"{case_id}_seg.dcm"
        seg.save_as(seg_path)
        written.append(seg_path)

    if write_sr:
        findings = build_findings(
            lesion, gland, lesion_prob, voxel_spacing_mm=spacing,
        )
        measurements = findings_to_measurements(findings)
        sr = create_measurement_report(
            source_images, measurements,
            manufacturer=_PRODUCER, device_name=_PRODUCER,
            research_disclaimer=_RESEARCH_DISCLAIMER,
        )
        sr_path = output_dir / f"{case_id}_sr.dcm"
        sr.save_as(sr_path)
        written.append(sr_path)

    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a finished case as DICOM-SEG + DICOM-SR (orchestration).",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--case", required=True, help="Case id to export.")
    parser.add_argument(
        "--source-series", type=Path, required=True,
        help="Directory of the retained source T2 DICOM series (dicom_mapper t2_source/).",
    )
    parser.add_argument("--seg", action="store_true", help="Write the DICOM-SEG object.")
    parser.add_argument("--sr", action="store_true", help="Write the DICOM-SR object.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write .dcm files (default: <run_dir>/diagnostic/dicom/).",
    )
    args = parser.parse_args(argv)

    paths = resolve_run_dir(args.run_dir)
    diag_root = paths.run_dir / "diagnostic"
    output_dir = args.output_dir or (diag_root / "dicom")

    # Neither flag => write both objects.
    neither = not (args.seg or args.sr)
    write_seg = args.seg or neither
    write_sr = args.sr or neither

    written = export_case(
        predictions_dir=diag_root / "predictions",
        postprocessed_dir=diag_root / "postprocessed",
        source_series=args.source_series,
        case_id=args.case,
        output_dir=output_dir,
        write_seg=write_seg,
        write_sr=write_sr,
    )
    print(f"[export_dicom] wrote {len(written)} object(s): "
          + ", ".join(p.name for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
