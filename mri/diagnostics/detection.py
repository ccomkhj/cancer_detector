"""Per-3D-lesion detection scoring for postprocessed segmentation predictions.

Pure NumPy + scipy.ndimage. The CLI in mri/cli/evaluate.py is responsible
for I/O.
"""

from __future__ import annotations

import csv as _csv
import json as _json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import ndimage

from mri.diagnostics.postprocess import apply_postprocess


def label_lesion_components(
    gt_lesion: np.ndarray,
    *,
    connectivity_rank: int = 1,
) -> tuple[np.ndarray, int]:
    """3D connected-component labeling of a binary GT lesion volume.

    Args:
      gt_lesion: (Z, H, W) uint8 array; non-zero voxels are foreground.
      connectivity_rank: passed straight to
          ``scipy.ndimage.generate_binary_structure(3, rank)``. Use
          ``1`` for 6-connectivity (default) or ``3`` for 26-connectivity.

    Returns:
      ``(labels, n_components)`` where ``labels`` has the same shape as
      ``gt_lesion`` with components numbered 1..n (0 = background).
    """
    structure = ndimage.generate_binary_structure(3, connectivity_rank)
    labels, n = ndimage.label(gt_lesion.astype(bool), structure=structure)
    return labels.astype(np.int32), int(n)


@dataclass(frozen=True)
class LesionIoUResult:
    """Outcome of evaluating one 3D GT component against a prediction volume."""
    slices: tuple[int, ...]      # z indices the component spans, ascending
    max_slice_iou: float
    argmax_slice: int            # z index achieving max_slice_iou; lowest-z on ties


def compute_lesion_iou(
    component_mask: np.ndarray,
    pred_lesion_mask: np.ndarray,
) -> LesionIoUResult:
    """Per-slice IoU of one GT lesion component vs the full predicted lesion.

    For each slice z that the component spans, compute IoU between the
    component's slice mask and the *entire* predicted-lesion slice (no
    cropping). Returns the max IoU across those slices and the lowest z
    that achieves it.

    Args:
      component_mask: (Z, H, W) bool array with one connected GT component
          set to True; False elsewhere.
      pred_lesion_mask: (Z, H, W) bool/uint8 array of postprocessed
          predicted lesion voxels.

    Returns:
      ``LesionIoUResult`` with ``slices`` empty if the component is empty.
    """
    component = component_mask.astype(bool)
    pred = pred_lesion_mask.astype(bool)
    assert component.shape == pred.shape, (
        f"shape mismatch: component {component.shape} vs pred {pred.shape}"
    )

    slice_has_component = component.any(axis=(1, 2))
    slices = tuple(int(z) for z in np.flatnonzero(slice_has_component))

    if not slices:
        return LesionIoUResult(slices=(), max_slice_iou=0.0, argmax_slice=0)

    max_iou = -1.0
    argmax_z = slices[0]
    for z in slices:
        gt_z = component[z]
        pr_z = pred[z]
        inter = int(np.logical_and(gt_z, pr_z).sum())
        union = int(np.logical_or(gt_z, pr_z).sum())
        iou = (inter / union) if union > 0 else 0.0
        if iou > max_iou:
            max_iou = iou
            argmax_z = z

    return LesionIoUResult(
        slices=slices,
        max_slice_iou=float(max_iou),
        argmax_slice=int(argmax_z),
    )


@dataclass(frozen=True)
class LesionRow:
    """One row of metrics_by_lesion.csv (per 3D GT component)."""
    case_id: str
    class_label: int
    lesion_id: int
    lesion_voxels: int
    slices: str           # ";"-joined z indices the component spans
    n_slices: int
    max_slice_iou: float
    argmax_slice: int
    detected: bool


@dataclass(frozen=True)
class CaseRow:
    """One row of metrics_by_case.csv. Mixed positive/negative case schema.

    For positive cases: ``max_pred_area_frac`` and ``negative_correct`` are None.
    For negative cases: ``lesion_recall`` is None.
    Writers translate None to empty CSV cells.
    """
    case_id: str
    class_label: int
    case_kind: str        # "positive" | "negative"
    n_gt_lesions: int
    n_detected_lesions: int
    lesion_recall: float | None
    max_pred_area_frac: float | None
    negative_correct: bool | None


def evaluate_case(
    *,
    case_id: str,
    class_label: int,
    gt_lesion: np.ndarray,
    pred_lesion: np.ndarray,
    correctness_iou: float,
    negative_area_frac: float,
    connectivity_rank: int,
) -> tuple[CaseRow, list[LesionRow]]:
    """Score one case under the per-3D-lesion + negative-area rule.

    Returns:
      ``(case_row, lesion_rows)``. ``lesion_rows`` is empty for negative cases.
    """
    assert gt_lesion.shape == pred_lesion.shape, (
        f"shape mismatch: gt {gt_lesion.shape} vs pred {pred_lesion.shape}"
    )

    labels, n_components = label_lesion_components(
        gt_lesion, connectivity_rank=connectivity_rank,
    )

    if n_components == 0:
        Z, H, W = pred_lesion.shape
        per_slice_voxels = pred_lesion.astype(bool).reshape(Z, -1).sum(axis=1)
        per_slice_frac = per_slice_voxels / float(H * W)
        max_frac = float(per_slice_frac.max()) if Z > 0 else 0.0
        return (
            CaseRow(
                case_id=case_id,
                class_label=class_label,
                case_kind="negative",
                n_gt_lesions=0,
                n_detected_lesions=0,
                lesion_recall=None,
                max_pred_area_frac=max_frac,
                negative_correct=(max_frac <= negative_area_frac),
            ),
            [],
        )

    pred_bool = pred_lesion.astype(bool)
    lesion_rows: list[LesionRow] = []
    detected = 0
    for k in range(1, n_components + 1):
        component = (labels == k)
        ious = compute_lesion_iou(component, pred_bool)
        is_detected = ious.max_slice_iou > correctness_iou
        if is_detected:
            detected += 1
        lesion_rows.append(
            LesionRow(
                case_id=case_id,
                class_label=class_label,
                lesion_id=k,
                lesion_voxels=int(component.sum()),
                slices=";".join(str(z) for z in ious.slices),
                n_slices=len(ious.slices),
                max_slice_iou=ious.max_slice_iou,
                argmax_slice=ious.argmax_slice,
                detected=is_detected,
            )
        )

    return (
        CaseRow(
            case_id=case_id,
            class_label=class_label,
            case_kind="positive",
            n_gt_lesions=n_components,
            n_detected_lesions=detected,
            lesion_recall=detected / n_components,
            max_pred_area_frac=None,
            negative_correct=None,
        ),
        lesion_rows,
    )


def _empty_for_none(value: Any) -> Any:
    """Translate None to empty string for CSV cells. Other types pass through."""
    return "" if value is None else value


def write_lesion_csv(rows: Iterable[LesionRow], path: Path) -> None:
    """Write metrics_by_lesion.csv. Empty list => header-only file."""
    rows = list(rows)
    fieldnames = [
        "case_id", "class_label", "lesion_id", "lesion_voxels",
        "slices", "n_slices", "max_slice_iou", "argmax_slice", "detected",
    ]
    with Path(path).open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_case_csv(rows: Iterable[CaseRow], path: Path) -> None:
    """Write metrics_by_case.csv. None values become empty cells."""
    rows = list(rows)
    fieldnames = [
        "case_id", "class_label", "case_kind",
        "n_gt_lesions", "n_detected_lesions", "lesion_recall",
        "max_pred_area_frac", "negative_correct",
    ]
    with Path(path).open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            d = asdict(row)
            writer.writerow({k: _empty_for_none(v) for k, v in d.items()})


def build_summary(
    *,
    case_rows: Iterable[CaseRow],
    lesion_rows: Iterable[LesionRow],
    params: Mapping[str, Any],
    cases_skipped: Iterable[str],
) -> dict[str, Any]:
    """Aggregate cohort metrics into the summary.json shape."""
    case_rows = list(case_rows)
    lesion_rows = list(lesion_rows)

    pos_cases = [c for c in case_rows if c.case_kind == "positive"]
    neg_cases = [c for c in case_rows if c.case_kind == "negative"]

    n_gt_lesions = sum(c.n_gt_lesions for c in pos_cases)
    n_detected = sum(c.n_detected_lesions for c in pos_cases)
    lesion_recall = (n_detected / n_gt_lesions) if n_gt_lesions > 0 else 0.0

    n_neg_correct = sum(1 for c in neg_cases if c.negative_correct)
    neg_accuracy = (n_neg_correct / len(neg_cases)) if neg_cases else 0.0

    return {
        "params": dict(params),
        "positives": {
            "n_cases": len(pos_cases),
            "n_gt_lesions": n_gt_lesions,
            "n_detected_lesions": n_detected,
            "lesion_recall": lesion_recall,
        },
        "negatives": {
            "n_cases": len(neg_cases),
            "n_correct": n_neg_correct,
            "negative_accuracy": neg_accuracy,
        },
        "cases_skipped": list(cases_skipped),
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    Path(path).write_text(_json.dumps(summary, indent=2))


# --------------------------------------------------------------------------- #
# FROC curve + operating-point selection (shortlist #3)
#
# The fixed-threshold metrics above only measure recall of ground-truth
# lesions. FROC additionally counts *false-positive predicted lesions*: a
# predicted connected component that meets no ground-truth lesion above the
# correctness IoU. This mirrors the detection rule with the roles of
# prediction and ground truth swapped, and is the one new detection concept
# these functions introduce. All pure NumPy/scipy; the CLI does the I/O.
# --------------------------------------------------------------------------- #


def _component_max_ious(
    labels: np.ndarray, n_components: int, other: np.ndarray,
) -> list[float]:
    """Max per-slice IoU of each labelled component against ``other``.

    The single traversal shared by detection and false-positive counting, so
    the correctness-IoU rule is applied identically on both sides.
    """
    other_bool = other.astype(bool)
    return [
        compute_lesion_iou((labels == k), other_bool).max_slice_iou
        for k in range(1, n_components + 1)
    ]


def count_false_positive_components(
    pred_lesion: np.ndarray,
    gt_lesion: np.ndarray,
    *,
    correctness_iou: float,
    connectivity_rank: int,
) -> int:
    """Count predicted lesion components that meet no ground-truth lesion.

    A predicted connected component is a false positive when its maximum
    per-slice IoU against the *entire* GT lesion volume does not exceed
    ``correctness_iou`` -- the detection rule of :func:`compute_lesion_iou`
    with prediction and ground truth swapped.

    Args:
      pred_lesion: (Z, H, W) predicted (already postprocessed) lesion voxels.
      gt_lesion:   (Z, H, W) ground-truth lesion voxels.
      correctness_iou: minimum max-slice IoU for a predicted component to
          count as meeting a GT lesion (strictly greater than).
      connectivity_rank: component connectivity, as in
          :func:`label_lesion_components`.
    """
    assert pred_lesion.shape == gt_lesion.shape, (
        f"shape mismatch: pred {pred_lesion.shape} vs gt {gt_lesion.shape}"
    )
    labels, n = label_lesion_components(
        pred_lesion, connectivity_rank=connectivity_rank,
    )
    ious = _component_max_ious(labels, n, gt_lesion)
    return sum(1 for iou in ious if iou <= correctness_iou)


@dataclass(frozen=True)
class CaseDetectionCounts:
    """Per-case detection counts at a single lesion threshold."""
    n_gt_lesions: int
    n_detected: int
    n_false_positives: int


def count_case_detections(
    *,
    gt_lesion: np.ndarray,
    pred_lesion: np.ndarray,
    correctness_iou: float,
    connectivity_rank: int,
) -> CaseDetectionCounts:
    """Count detected GT lesions and false-positive predicted components.

    A GT lesion component is *detected* when its max per-slice IoU against the
    full predicted lesion exceeds ``correctness_iou`` (the same rule
    :func:`evaluate_case` uses). Detection and false positives are scored with
    the same connectivity and correctness IoU so a "lesion" means the same
    thing on both sides.
    """
    assert gt_lesion.shape == pred_lesion.shape, (
        f"shape mismatch: gt {gt_lesion.shape} vs pred {pred_lesion.shape}"
    )
    labels, n_gt = label_lesion_components(
        gt_lesion, connectivity_rank=connectivity_rank,
    )
    gt_ious = _component_max_ious(labels, n_gt, pred_lesion)
    detected = sum(1 for iou in gt_ious if iou > correctness_iou)
    n_fp = count_false_positive_components(
        pred_lesion, gt_lesion,
        correctness_iou=correctness_iou, connectivity_rank=connectivity_rank,
    )
    return CaseDetectionCounts(
        n_gt_lesions=n_gt, n_detected=detected, n_false_positives=n_fp,
    )


@dataclass(frozen=True)
class FrocCaseInput:
    """One case's arrays for the FROC sweep.

    ``lesion_prob`` / ``gland_prob`` are the probability volumes from
    ``diagnostic/predictions/<case>/prob.npz``; ``gt_lesion`` is the
    ground-truth lesion volume from ``gt.npz``.
    """
    case_id: str
    gt_lesion: np.ndarray
    lesion_prob: np.ndarray
    gland_prob: np.ndarray


@dataclass(frozen=True)
class FrocPoint:
    """One point on the cohort FROC curve (one lesion threshold)."""
    threshold: float
    sensitivity: float
    fp_per_case: float
    n_gt_lesions: int
    n_detected: int
    n_false_positives: int
    n_cases: int


def compute_froc_curve(
    cases: Iterable[FrocCaseInput],
    *,
    thresholds: Iterable[float],
    correctness_iou: float,
    connectivity_rank: int,
    gland_threshold: float,
) -> list[FrocPoint]:
    """Sweep the lesion threshold and aggregate cohort FROC points.

    For each threshold the lesion probability is binarised and
    gland-constrained with :func:`apply_postprocess` (so the curve reflects the
    deployed postprocessing), then scored with :func:`count_case_detections`.
    Cohort aggregation per threshold:

      - ``sensitivity  = total detected GT lesions / total GT lesions``
      - ``fp_per_case  = total false positives / number of cases``
    """
    cases = list(cases)
    n_cases = len(cases)
    points: list[FrocPoint] = []
    for threshold in thresholds:
        total_gt = 0
        total_detected = 0
        total_fp = 0
        for case in cases:
            pred_lesion, _gland_mask, _present = apply_postprocess(
                case.lesion_prob, case.gland_prob,
                lesion_threshold=threshold, gland_threshold=gland_threshold,
            )
            counts = count_case_detections(
                gt_lesion=case.gt_lesion, pred_lesion=pred_lesion,
                correctness_iou=correctness_iou,
                connectivity_rank=connectivity_rank,
            )
            total_gt += counts.n_gt_lesions
            total_detected += counts.n_detected
            total_fp += counts.n_false_positives
        points.append(FrocPoint(
            threshold=float(threshold),
            sensitivity=(total_detected / total_gt) if total_gt > 0 else 0.0,
            fp_per_case=(total_fp / n_cases) if n_cases > 0 else 0.0,
            n_gt_lesions=total_gt,
            n_detected=total_detected,
            n_false_positives=total_fp,
            n_cases=n_cases,
        ))
    return points


@dataclass(frozen=True)
class OperatingPoint:
    """The lesion threshold selected from a FROC curve."""
    threshold: float
    sensitivity: float
    fp_per_case: float
    target_fp_per_case: float
    target_met: bool


def select_operating_point(
    points: Iterable[FrocPoint],
    *,
    target_fp_per_case: float,
) -> OperatingPoint:
    """Pick the threshold maximising sensitivity within an FP/case budget.

    Among points at or below ``target_fp_per_case``, choose the one with the
    highest sensitivity; ties are broken toward the fewest false positives,
    then the highest threshold (least over-detection for the same
    sensitivity). If no point meets the budget, fall back to the point with
    the fewest false positives and mark ``target_met=False``.
    """
    points = list(points)
    if not points:
        raise ValueError("cannot select an operating point from an empty FROC curve")

    within = [p for p in points if p.fp_per_case <= target_fp_per_case]
    if within:
        best = max(within, key=lambda p: (p.sensitivity, -p.fp_per_case, p.threshold))
        target_met = True
    else:
        best = min(points, key=lambda p: (p.fp_per_case, -p.sensitivity, -p.threshold))
        target_met = False

    return OperatingPoint(
        threshold=best.threshold,
        sensitivity=best.sensitivity,
        fp_per_case=best.fp_per_case,
        target_fp_per_case=float(target_fp_per_case),
        target_met=target_met,
    )


def sensitivity_at_fp_rates(
    points: Iterable[FrocPoint],
    *,
    fp_rates: Iterable[float] = (0.5, 1.0, 2.0, 4.0),
) -> list[tuple[float, float]]:
    """Max sensitivity achievable at or below each standard FP/case rate."""
    points = list(points)
    out: list[tuple[float, float]] = []
    for rate in fp_rates:
        within = [p.sensitivity for p in points if p.fp_per_case <= rate]
        out.append((float(rate), max(within) if within else 0.0))
    return out


def write_froc_csv(points: Iterable[FrocPoint], path: Path) -> None:
    """Write froc.csv. Empty list => header-only file."""
    fieldnames = [
        "threshold", "sensitivity", "fp_per_case",
        "n_gt_lesions", "n_detected", "n_false_positives", "n_cases",
    ]
    with Path(path).open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def build_operating_point_summary(
    *,
    operating_point: OperatingPoint,
    sensitivity_at_rates: Iterable[tuple[float, float]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the operating_point.json payload (stable schema)."""
    return {
        "operating_point": {
            "threshold": operating_point.threshold,
            "sensitivity": operating_point.sensitivity,
            "fp_per_case": operating_point.fp_per_case,
            "target_fp_per_case": operating_point.target_fp_per_case,
            "target_met": operating_point.target_met,
        },
        "sensitivity_at_fp_rates": [
            {"fp_per_case": rate, "sensitivity": sensitivity}
            for rate, sensitivity in sensitivity_at_rates
        ],
        "params": dict(params),
    }
