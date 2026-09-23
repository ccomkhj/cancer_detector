"""Unit tests for per-3D-lesion detection scoring."""

from __future__ import annotations

import numpy as np

from mri.diagnostics.detection import label_lesion_components


def test_single_lesion_across_three_slices_is_one_component() -> None:
    gt = np.zeros((5, 6, 6), dtype=np.uint8)
    gt[1, 2, 2] = 1
    gt[2, 2, 2] = 1
    gt[3, 2, 2] = 1

    labels, n = label_lesion_components(gt, connectivity_rank=1)

    assert n == 1
    assert labels.shape == gt.shape
    assert labels.dtype.kind == "i"
    assert (labels[gt == 1] == 1).all()
    assert (labels[gt == 0] == 0).all()


def test_two_disjoint_lesions_are_two_components() -> None:
    gt = np.zeros((5, 6, 6), dtype=np.uint8)
    gt[1, 1, 1] = 1
    gt[1, 4, 4] = 1  # spatially disjoint on the same slice

    labels, n = label_lesion_components(gt, connectivity_rank=1)

    assert n == 2
    assert sorted(np.unique(labels[gt == 1]).tolist()) == [1, 2]


def test_diagonal_only_split_under_6_connectivity_joined_under_26() -> None:
    gt = np.zeros((3, 4, 4), dtype=np.uint8)
    gt[1, 1, 1] = 1
    gt[1, 2, 2] = 1  # diagonal in-plane

    _, n6 = label_lesion_components(gt, connectivity_rank=1)
    _, n26 = label_lesion_components(gt, connectivity_rank=3)

    assert n6 == 2
    assert n26 == 1


def test_empty_gt_yields_zero_components() -> None:
    gt = np.zeros((3, 4, 4), dtype=np.uint8)

    labels, n = label_lesion_components(gt, connectivity_rank=1)

    assert n == 0
    assert (labels == 0).all()


from mri.diagnostics.detection import compute_lesion_iou


def test_lesion_iou_max_across_slices_with_argmax() -> None:
    # Component spans z=1..3. Pred overlaps best at z=2.
    component = np.zeros((5, 4, 4), dtype=bool)
    component[1, 1, 1] = True
    component[2, 1, 1] = True
    component[2, 1, 2] = True
    component[3, 1, 1] = True

    pred = np.zeros((5, 4, 4), dtype=bool)
    pred[1, 1, 1] = True             # iou = 1/1 = 1.0  (single voxel exact)
    pred[2, 1, 1] = True             # iou = 1/2 on z=2 (component has 2 voxels)
    pred[3, 0, 0] = True             # iou = 0 on z=3

    result = compute_lesion_iou(component, pred)

    assert result.slices == (1, 2, 3)
    # z=1: 1/1, z=2: 1/2, z=3: 0/(1+1)=0
    assert result.max_slice_iou == 1.0
    assert result.argmax_slice == 1


def test_lesion_iou_argmax_breaks_ties_with_lowest_z() -> None:
    component = np.zeros((4, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    component[2, 1, 1] = True
    pred = np.zeros((4, 3, 3), dtype=bool)
    pred[1, 1, 1] = True
    pred[2, 1, 1] = True

    result = compute_lesion_iou(component, pred)

    assert result.max_slice_iou == 1.0
    assert result.argmax_slice == 1


def test_lesion_iou_all_zero_pred_is_zero() -> None:
    component = np.zeros((3, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    pred = np.zeros((3, 3, 3), dtype=bool)

    result = compute_lesion_iou(component, pred)

    assert result.max_slice_iou == 0.0
    assert result.argmax_slice == 1


def test_lesion_iou_partial_overlap_value() -> None:
    # Component on z=0 = 4 voxels. Pred on z=0 = 2 voxels overlapping. iou = 2/4 = 0.5
    component = np.zeros((1, 4, 4), dtype=bool)
    component[0, 1:3, 1:3] = True  # 4 voxels
    pred = np.zeros((1, 4, 4), dtype=bool)
    pred[0, 1, 1] = True
    pred[0, 1, 2] = True

    result = compute_lesion_iou(component, pred)

    assert result.max_slice_iou == 0.5
    assert result.argmax_slice == 0


from mri.diagnostics.detection import (
    LesionRow, CaseRow, evaluate_case,
)


def test_evaluate_case_positive_two_lesions_one_detected() -> None:
    gt = np.zeros((4, 6, 6), dtype=np.uint8)
    gt[1, 1, 1] = 1
    gt[2, 4, 4] = 1

    pred = np.zeros((4, 6, 6), dtype=np.uint8)
    pred[1, 1, 1] = 1

    case_row, lesion_rows = evaluate_case(
        case_id="c1", class_label=2,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.case_kind == "positive"
    assert case_row.n_gt_lesions == 2
    assert case_row.n_detected_lesions == 1
    assert case_row.lesion_recall == 0.5
    assert case_row.max_pred_area_frac is None
    assert case_row.negative_correct is None

    assert len(lesion_rows) == 2
    detected_ids = {row.lesion_id for row in lesion_rows if row.detected}
    assert len(detected_ids) == 1


def test_evaluate_case_negative_below_threshold_is_correct() -> None:
    gt = np.zeros((3, 10, 10), dtype=np.uint8)
    pred = np.zeros((3, 10, 10), dtype=np.uint8)
    pred[0, 0, 0] = 1

    case_row, lesion_rows = evaluate_case(
        case_id="c2", class_label=0,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.case_kind == "negative"
    assert case_row.n_gt_lesions == 0
    assert case_row.n_detected_lesions == 0
    assert case_row.lesion_recall is None
    assert case_row.max_pred_area_frac == 0.01
    assert case_row.negative_correct is True
    assert lesion_rows == []


def test_evaluate_case_negative_above_threshold_is_false() -> None:
    gt = np.zeros((3, 10, 10), dtype=np.uint8)
    pred = np.zeros((3, 10, 10), dtype=np.uint8)
    pred[0, 0, 0:3] = 1

    case_row, _ = evaluate_case(
        case_id="c3", class_label=0,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.negative_correct is False
    assert case_row.max_pred_area_frac == 0.03


def test_evaluate_case_negative_at_threshold_is_correct() -> None:
    gt = np.zeros((1, 10, 10), dtype=np.uint8)
    pred = np.zeros((1, 10, 10), dtype=np.uint8)
    pred[0, 0, 0:2] = 1

    case_row, _ = evaluate_case(
        case_id="c4", class_label=0,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.negative_correct is True


def test_evaluate_case_positive_iou_at_threshold_is_not_detected() -> None:
    gt = np.zeros((1, 10, 10), dtype=np.uint8)
    gt[0, 0, 0:10] = 1
    pred = np.zeros((1, 10, 10), dtype=np.uint8)
    pred[0, 0, 0] = 1  # iou = 1/10 = 0.1 exactly => detected = False (strict >)

    case_row, lesion_rows = evaluate_case(
        case_id="c5", class_label=2,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.case_kind == "positive"
    assert lesion_rows[0].max_slice_iou == 0.1
    assert lesion_rows[0].detected is False
    assert case_row.n_detected_lesions == 0


import csv
import json
from pathlib import Path

from mri.diagnostics.detection import (
    write_lesion_csv, write_case_csv, build_summary, write_summary_json,
)


def _make_pos_rows() -> tuple[CaseRow, list[LesionRow]]:
    case = CaseRow(
        case_id="c1", class_label=2, case_kind="positive",
        n_gt_lesions=2, n_detected_lesions=1, lesion_recall=0.5,
        max_pred_area_frac=None, negative_correct=None,
    )
    rows = [
        LesionRow(case_id="c1", class_label=2, lesion_id=1, lesion_voxels=4,
                  slices="1;2", n_slices=2, max_slice_iou=0.42, argmax_slice=2,
                  detected=True),
        LesionRow(case_id="c1", class_label=2, lesion_id=2, lesion_voxels=3,
                  slices="3", n_slices=1, max_slice_iou=0.05, argmax_slice=3,
                  detected=False),
    ]
    return case, rows


def _make_neg_row() -> CaseRow:
    return CaseRow(
        case_id="c2", class_label=0, case_kind="negative",
        n_gt_lesions=0, n_detected_lesions=0, lesion_recall=None,
        max_pred_area_frac=0.015, negative_correct=True,
    )


def test_lesion_csv_columns_and_values(tmp_path: Path) -> None:
    _, rows = _make_pos_rows()
    out = tmp_path / "metrics_by_lesion.csv"

    write_lesion_csv(rows, out)

    with out.open() as f:
        reader = csv.DictReader(f)
        records = list(reader)
    assert reader.fieldnames == [
        "case_id", "class_label", "lesion_id", "lesion_voxels",
        "slices", "n_slices", "max_slice_iou", "argmax_slice", "detected",
    ]
    assert records[0]["lesion_id"] == "1"
    assert records[0]["detected"] == "True"
    assert records[1]["detected"] == "False"


def test_case_csv_writes_empty_string_for_none(tmp_path: Path) -> None:
    case_pos, _ = _make_pos_rows()
    case_neg = _make_neg_row()
    out = tmp_path / "metrics_by_case.csv"

    write_case_csv([case_pos, case_neg], out)

    text = out.read_text()
    assert "None" not in text
    assert "nan" not in text.lower()

    with out.open() as f:
        reader = csv.DictReader(f)
        records = list(reader)
    assert records[0]["max_pred_area_frac"] == ""
    assert records[0]["negative_correct"] == ""
    assert records[1]["lesion_recall"] == ""
    assert records[0]["lesion_recall"] == "0.5"
    assert records[1]["max_pred_area_frac"] == "0.015"
    assert records[1]["negative_correct"] == "True"


def test_build_summary_aggregates_positive_and_negative(tmp_path: Path) -> None:
    case_pos, rows_pos = _make_pos_rows()
    case_neg = _make_neg_row()

    summary = build_summary(
        case_rows=[case_pos, case_neg],
        lesion_rows=rows_pos,
        params={
            "correctness_iou": 0.1, "negative_area_frac": 0.02,
            "connectivity": 6, "lesion_threshold": 0.5, "gland_threshold": 0.5,
        },
        cases_skipped=[],
    )

    assert summary["positives"]["n_cases"] == 1
    assert summary["positives"]["n_gt_lesions"] == 2
    assert summary["positives"]["n_detected_lesions"] == 1
    assert summary["positives"]["lesion_recall"] == 0.5
    assert summary["negatives"]["n_cases"] == 1
    assert summary["negatives"]["n_correct"] == 1
    assert summary["negatives"]["negative_accuracy"] == 1.0
    assert summary["params"]["correctness_iou"] == 0.1
    assert summary["cases_skipped"] == []


def test_write_summary_json_round_trip(tmp_path: Path) -> None:
    summary = {"params": {"correctness_iou": 0.1}, "positives": {"n_cases": 0}}
    out = tmp_path / "summary.json"

    write_summary_json(summary, out)

    loaded = json.loads(out.read_text())
    assert loaded == summary


# --------------------------------------------------------------------------- #
# FROC curve + operating-point selection (shortlist #3)
# --------------------------------------------------------------------------- #

from mri.diagnostics.detection import (
    count_false_positive_components,
    count_case_detections,
    CaseDetectionCounts,
    FrocCaseInput,
    FrocPoint,
    compute_froc_curve,
    select_operating_point,
    sensitivity_at_fp_rates,
    write_froc_csv,
    build_operating_point_summary,
)


def _block(z: int, r0: int, c0: int, *, shape: tuple[int, int, int]) -> np.ndarray:
    """A (Z,H,W) uint8 volume with a 2x2 block set on slice ``z``."""
    vol = np.zeros(shape, dtype=np.uint8)
    vol[z, r0:r0 + 2, c0:c0 + 2] = 1
    return vol


def test_false_positive_component_overlapping_gt_is_not_counted() -> None:
    shape = (3, 8, 8)
    gt = _block(0, 0, 0, shape=shape)
    pred = _block(0, 0, 0, shape=shape)  # exact overlap

    fp = count_false_positive_components(
        pred, gt, correctness_iou=0.1, connectivity_rank=1,
    )

    assert fp == 0


def test_predicted_component_meeting_no_gt_is_a_false_positive() -> None:
    shape = (3, 8, 8)
    gt = _block(0, 0, 0, shape=shape)
    pred = _block(0, 5, 5, shape=shape)  # disjoint from GT

    fp = count_false_positive_components(
        pred, gt, correctness_iou=0.1, connectivity_rank=1,
    )

    assert fp == 1


def test_false_positive_count_mixes_hits_and_misses() -> None:
    shape = (3, 8, 8)
    gt = _block(0, 0, 0, shape=shape)
    pred = _block(0, 0, 0, shape=shape) | _block(2, 5, 5, shape=shape)

    fp = count_false_positive_components(
        pred, gt, correctness_iou=0.1, connectivity_rank=1,
    )

    assert fp == 1  # the slice-2 component meets no GT


def test_empty_prediction_has_no_false_positives() -> None:
    shape = (3, 8, 8)
    gt = _block(0, 0, 0, shape=shape)
    pred = np.zeros(shape, dtype=np.uint8)

    assert count_false_positive_components(
        pred, gt, correctness_iou=0.1, connectivity_rank=1,
    ) == 0


def test_count_case_detections_reports_gt_detected_and_fp() -> None:
    shape = (3, 8, 8)
    gt = _block(0, 0, 0, shape=shape) | _block(2, 0, 0, shape=shape)  # 2 GT lesions
    pred = _block(0, 0, 0, shape=shape) | _block(1, 5, 0, shape=shape)  # 1 hit + 1 FP

    counts = count_case_detections(
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, connectivity_rank=1,
    )

    assert counts == CaseDetectionCounts(
        n_gt_lesions=2, n_detected=1, n_false_positives=1,
    )


def _froc_cohort() -> list[FrocCaseInput]:
    """One case with 2 GT lesions and probability-graded FP/detection regions.

    Layout (Z=3, H=8, W=8), gland present everywhere:
      - GT1 + prob 0.90 at slice 0, rows/cols 0:2  -> detected once t <= 0.90
      - FP  B  prob 0.70 at slice 0, rows/cols 5:7  -> FP once t <= 0.70
      - FP  C  prob 0.55 at slice 1, rows 5:7 c 0:2 -> FP once t <= 0.55
      - GT2 + prob 0.45 at slice 2, rows/cols 0:2  -> detected once t <= 0.45
    """
    shape = (3, 8, 8)
    lesion_prob = np.zeros(shape, dtype=np.float32)
    lesion_prob[0, 0:2, 0:2] = 0.90
    lesion_prob[0, 5:7, 5:7] = 0.70
    lesion_prob[1, 5:7, 0:2] = 0.55
    lesion_prob[2, 0:2, 0:2] = 0.45
    gland_prob = np.ones(shape, dtype=np.float32)
    gt = _block(0, 0, 0, shape=shape) | _block(2, 0, 0, shape=shape)
    return [FrocCaseInput(
        case_id="c1", gt_lesion=gt, lesion_prob=lesion_prob, gland_prob=gland_prob,
    )]


def test_compute_froc_curve_values_and_monotonicity() -> None:
    cases = _froc_cohort()
    thresholds = [0.8, 0.6, 0.5, 0.4]

    points = compute_froc_curve(
        cases, thresholds=thresholds,
        correctness_iou=0.1, connectivity_rank=1, gland_threshold=0.5,
    )

    by_t = {p.threshold: p for p in points}
    assert (by_t[0.8].sensitivity, by_t[0.8].fp_per_case) == (0.5, 0.0)
    assert (by_t[0.6].sensitivity, by_t[0.6].fp_per_case) == (0.5, 1.0)
    assert (by_t[0.5].sensitivity, by_t[0.5].fp_per_case) == (0.5, 2.0)
    assert (by_t[0.4].sensitivity, by_t[0.4].fp_per_case) == (1.0, 2.0)

    # Monotonic: lowering the threshold never lowers sensitivity or FP/case.
    ordered = [by_t[t] for t in thresholds]  # thresholds descending
    for hi, lo in zip(ordered, ordered[1:]):
        assert lo.sensitivity >= hi.sensitivity
        assert lo.fp_per_case >= hi.fp_per_case


def test_compute_froc_curve_respects_gland_constraint() -> None:
    cases = _froc_cohort()
    # Zero out the gland everywhere -> postprocess zeroes every lesion mask.
    cases = [FrocCaseInput(
        case_id=c.case_id, gt_lesion=c.gt_lesion,
        lesion_prob=c.lesion_prob, gland_prob=np.zeros_like(c.gland_prob),
    ) for c in cases]

    points = compute_froc_curve(
        cases, thresholds=[0.4], correctness_iou=0.1,
        connectivity_rank=1, gland_threshold=0.5,
    )

    assert points[0].sensitivity == 0.0
    assert points[0].fp_per_case == 0.0


def _froc_points() -> list[FrocPoint]:
    return [
        FrocPoint(threshold=0.8, sensitivity=0.5, fp_per_case=0.0,
                  n_gt_lesions=2, n_detected=1, n_false_positives=0, n_cases=1),
        FrocPoint(threshold=0.6, sensitivity=0.5, fp_per_case=1.0,
                  n_gt_lesions=2, n_detected=1, n_false_positives=1, n_cases=1),
        FrocPoint(threshold=0.5, sensitivity=0.5, fp_per_case=2.0,
                  n_gt_lesions=2, n_detected=1, n_false_positives=2, n_cases=1),
        FrocPoint(threshold=0.4, sensitivity=1.0, fp_per_case=2.0,
                  n_gt_lesions=2, n_detected=2, n_false_positives=2, n_cases=1),
    ]


def test_operating_point_picks_max_sensitivity_within_budget() -> None:
    op = select_operating_point(_froc_points(), target_fp_per_case=2.0)

    assert op.threshold == 0.4
    assert op.sensitivity == 1.0
    assert op.fp_per_case == 2.0
    assert op.target_met is True


def test_operating_point_ties_prefer_fewest_false_positives() -> None:
    # Budget 1.0: thresholds 0.8 and 0.6 both reach sensitivity 0.5; 0.8 has 0 FP.
    op = select_operating_point(_froc_points(), target_fp_per_case=1.0)

    assert op.threshold == 0.8
    assert op.fp_per_case == 0.0
    assert op.target_met is True


def test_operating_point_marks_target_not_met() -> None:
    op = select_operating_point(_froc_points(), target_fp_per_case=-1.0)

    assert op.target_met is False
    assert op.threshold == 0.8  # fewest FPs is the best we can do


def test_sensitivity_at_standard_fp_rates() -> None:
    rates = sensitivity_at_fp_rates(_froc_points(), fp_rates=(0.5, 1.0, 2.0, 4.0))

    assert rates == [(0.5, 0.5), (1.0, 0.5), (2.0, 1.0), (4.0, 1.0)]


def test_write_froc_csv_columns_and_values(tmp_path: Path) -> None:
    out = tmp_path / "froc.csv"

    write_froc_csv(_froc_points(), out)

    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == [
        "threshold", "sensitivity", "fp_per_case",
        "n_gt_lesions", "n_detected", "n_false_positives", "n_cases",
    ]
    assert rows[0]["threshold"] == "0.8"
    assert rows[3]["sensitivity"] == "1.0"


def test_build_operating_point_summary_shape() -> None:
    op = select_operating_point(_froc_points(), target_fp_per_case=2.0)
    rates = sensitivity_at_fp_rates(_froc_points())

    summary = build_operating_point_summary(
        operating_point=op, sensitivity_at_rates=rates,
        params={"correctness_iou": 0.1, "connectivity": 6},
    )

    assert summary["operating_point"]["threshold"] == 0.4
    assert summary["operating_point"]["target_met"] is True
    assert summary["sensitivity_at_fp_rates"][0] == {
        "fp_per_case": 0.5, "sensitivity": 0.5,
    }
    assert summary["params"]["connectivity"] == 6
