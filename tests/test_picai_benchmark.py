"""Unit tests for mri.diagnostics.benchmark (picai_eval bridge + bootstrap CIs)."""

from __future__ import annotations

import numpy as np

from mri.diagnostics.benchmark import (
    BootstrapCI,
    PicaiCaseInput,
    bootstrap_case_metric_cis,
    lesion_detection_map,
    percentile_ci,
    score_with_picai_eval,
)
from mri.diagnostics.detection import CaseRow


# --------------------------------------------------------------------------- #
# lesion_detection_map
# --------------------------------------------------------------------------- #

def test_detection_map_fills_each_component_with_its_max_prob() -> None:
    mask = np.zeros((2, 6, 6), dtype=np.uint8)
    mask[0, 0:2, 0:2] = 1          # component A
    mask[1, 4:6, 4:6] = 1          # component B (disjoint in 3D, 6-conn)
    prob = np.zeros((2, 6, 6), dtype=np.float32)
    prob[0, 0, 0] = 0.9
    prob[0, 1, 1] = 0.4
    prob[1, 4:6, 4:6] = 0.6

    det = lesion_detection_map(mask, prob, connectivity_rank=1)

    assert det.dtype == np.float32
    a = det[0, 0:2, 0:2]
    b = det[1, 4:6, 4:6]
    assert set(np.unique(a)) == {np.float32(0.9)}     # max prob, uniform fill
    assert set(np.unique(b)) == {np.float32(0.6)}
    # non-overlapping: everything outside the two components is 0
    assert det.sum() == np.float32(0.9) * 4 + np.float32(0.6) * 4


def test_detection_map_empty_mask_is_all_zero() -> None:
    mask = np.zeros((1, 4, 4), dtype=np.uint8)
    prob = np.full((1, 4, 4), 0.7, dtype=np.float32)
    det = lesion_detection_map(mask, prob)
    assert not det.any()


# --------------------------------------------------------------------------- #
# percentile_ci
# --------------------------------------------------------------------------- #

def test_percentile_ci_drops_non_finite_replicates() -> None:
    reps = [0.5, 0.6, float("nan"), 0.7, float("inf")]
    ci = percentile_ci(reps, ci_level=0.95)
    assert ci.n_valid == 3
    assert 0.5 <= ci.lower <= ci.upper <= 0.7


def test_percentile_ci_all_nan_yields_none_bounds() -> None:
    ci = percentile_ci([float("nan")] * 5)
    assert ci == BootstrapCI(lower=None, upper=None, n_valid=0)


# --------------------------------------------------------------------------- #
# bootstrap_case_metric_cis
# --------------------------------------------------------------------------- #

def _pos_row(case_id: str, n_gt: int, n_det: int) -> CaseRow:
    return CaseRow(case_id=case_id, class_label=2, case_kind="positive",
                   n_gt_lesions=n_gt, n_detected_lesions=n_det,
                   lesion_recall=n_det / n_gt, max_pred_area_frac=None,
                   negative_correct=None)


def _neg_row(case_id: str, correct: bool) -> CaseRow:
    return CaseRow(case_id=case_id, class_label=0, case_kind="negative",
                   n_gt_lesions=0, n_detected_lesions=0, lesion_recall=None,
                   max_pred_area_frac=0.0, negative_correct=correct)


def test_bootstrap_case_metric_cis_perfect_cohort_is_degenerate_at_1() -> None:
    rows = [_pos_row(f"p{i}", 2, 2) for i in range(3)] + \
           [_neg_row(f"n{i}", True) for i in range(3)]
    cis = bootstrap_case_metric_cis(rows, n_boot=100, seed=0)
    assert cis["lesion_recall_ci"]["lower"] == 1.0
    assert cis["lesion_recall_ci"]["upper"] == 1.0
    assert cis["negative_accuracy_ci"]["lower"] == 1.0


def test_bootstrap_case_metric_cis_is_seed_deterministic() -> None:
    rows = [_pos_row("p0", 2, 1), _pos_row("p1", 1, 1), _neg_row("n0", False)]
    a = bootstrap_case_metric_cis(rows, n_boot=50, seed=7)
    b = bootstrap_case_metric_cis(rows, n_boot=50, seed=7)
    assert a == b
    # mixed detection => the recall CI must actually spread
    assert a["lesion_recall_ci"]["lower"] < a["lesion_recall_ci"]["upper"]


# --------------------------------------------------------------------------- #
# score_with_picai_eval (official package, tiny synthetic cohort)
# --------------------------------------------------------------------------- #

def _picai_cohort() -> list[PicaiCaseInput]:
    Z, H, W = 2, 8, 8
    cases = []
    for i in range(2):  # positives: GT lesion hit with confidence 0.9
        gt = np.zeros((Z, H, W), dtype=np.uint8)
        gt[0, 1:3, 1:3] = 1
        det = np.zeros((Z, H, W), dtype=np.float32)
        det[0, 1:3, 1:3] = 0.9
        cases.append(PicaiCaseInput(f"pos_{i}", gt, det))
    for i in range(2):  # negatives: empty detection map
        gt = np.zeros((Z, H, W), dtype=np.uint8)
        det = np.zeros((Z, H, W), dtype=np.float32)
        cases.append(PicaiCaseInput(f"neg_{i}", gt, det))
    return cases


def test_score_with_picai_eval_perfect_separation() -> None:
    block = score_with_picai_eval(_picai_cohort(), n_boot=50, seed=0)
    assert block["AP"] == 1.0
    assert block["auroc"] == 1.0
    assert block["ranking_score"] == 1.0
    assert block["num_cases"] == 4
    assert block["num_lesions"] == 2
    for key in ("AP_ci", "auroc_ci", "ranking_score_ci"):
        ci = block[key]
        assert ci["n_valid"] > 0
        assert ci["lower"] == 1.0 and ci["upper"] == 1.0


def test_score_with_picai_eval_no_bootstrap_when_disabled() -> None:
    block = score_with_picai_eval(_picai_cohort(), n_boot=0)
    assert "AP_ci" not in block
    assert block["params"]["n_boot"] == 0
