"""End-to-end CLI tests for `python -m mri.cli.evaluate`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mri.cli import evaluate as evaluate_cli


def _seed_predictions(run_dir: Path, case_id: str, *, gt_lesion, gt_gland) -> None:
    pdir = run_dir / "diagnostic" / "predictions" / case_id
    pdir.mkdir(parents=True, exist_ok=True)
    Z, H, W = gt_lesion.shape
    np.savez_compressed(pdir / "prob.npz",
                         gland=np.zeros((Z, H, W), dtype=np.float32),
                         lesion=np.zeros((Z, H, W), dtype=np.float32))
    np.savez_compressed(pdir / "gt.npz",
                         gland=gt_gland.astype(np.uint8),
                         lesion=gt_lesion.astype(np.uint8))
    (pdir / "meta.json").write_text(json.dumps({
        "case_id": case_id, "class_label": 2 if gt_lesion.any() else 0,
        "spatial_shape": [H, W], "num_slices": Z,
        "predicted_slices": list(range(Z)),
        "lesion_threshold": 0.5, "gland_threshold": 0.5,
    }))


def _seed_postprocessed(run_dir: Path, case_id: str, *, lesion_mask, gland_mask) -> None:
    pdir = run_dir / "diagnostic" / "postprocessed" / case_id
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / "lesion_mask.npz", mask=lesion_mask.astype(np.uint8))
    np.savez_compressed(pdir / "gland_mask.npz", mask=gland_mask.astype(np.uint8))
    (pdir / "meta.json").write_text(json.dumps({
        "case_id": case_id,
        "lesion_threshold": 0.5, "gland_threshold": 0.5,
        "gland_present": bool(gland_mask.any()),
        "lesion_voxels_raw": int(lesion_mask.sum()),
        "lesion_voxels_post": int(lesion_mask.sum()),
        "gland_voxels": int(gland_mask.sum()),
    }))


def _seed_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_best.pt").write_bytes(b"")
    (run_dir / "resolved_config.yaml").write_text(
        "metrics:\n  segmentation_threshold: 0.5\n"
    )
    return run_dir


def test_evaluate_cli_writes_lesion_case_csv_and_summary(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 3, 10, 10

    # Positive case: 2 lesions, 1 detected.
    gt_pos = np.zeros((Z, H, W), dtype=np.uint8)
    gt_pos[1, 1, 1] = 1
    gt_pos[2, 8, 8] = 1
    pred_pos = np.zeros((Z, H, W), dtype=np.uint8)
    pred_pos[1, 1, 1] = 1
    _seed_predictions(run_dir, "case_pos",
                      gt_lesion=gt_pos, gt_gland=np.zeros_like(gt_pos))
    _seed_postprocessed(run_dir, "case_pos",
                         lesion_mask=pred_pos, gland_mask=np.zeros_like(gt_pos))

    # Negative case: 1% predicted area, below 2% ⇒ correct.
    gt_neg = np.zeros((1, 10, 10), dtype=np.uint8)
    pred_neg = np.zeros_like(gt_neg)
    pred_neg[0, 0, 0] = 1
    _seed_predictions(run_dir, "case_neg",
                      gt_lesion=gt_neg, gt_gland=np.zeros_like(gt_neg))
    _seed_postprocessed(run_dir, "case_neg",
                         lesion_mask=pred_neg, gland_mask=np.zeros_like(gt_neg))

    rc = evaluate_cli.main([str(run_dir), "--visualize-only", "none"])

    assert rc == 0
    eval_dir = run_dir / "diagnostic" / "evaluation"
    assert (eval_dir / "metrics_by_lesion.csv").exists()
    assert (eval_dir / "metrics_by_case.csv").exists()
    assert (eval_dir / "summary.json").exists()
    assert not (eval_dir / "visuals").exists()

    with (eval_dir / "metrics_by_lesion.csv").open() as f:
        lesion_rows = list(csv.DictReader(f))
    assert len(lesion_rows) == 2
    assert {row["detected"] for row in lesion_rows} == {"True", "False"}

    with (eval_dir / "metrics_by_case.csv").open() as f:
        case_rows = list(csv.DictReader(f))
    assert {row["case_kind"] for row in case_rows} == {"positive", "negative"}

    summary = json.loads((eval_dir / "summary.json").read_text())
    assert summary["positives"]["n_cases"] == 1
    assert summary["positives"]["n_detected_lesions"] == 1
    assert summary["positives"]["n_gt_lesions"] == 2
    assert summary["negatives"]["n_correct"] == 1


def test_evaluate_cli_correctness_iou_flag_changes_detection(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 1, 10, 10
    gt = np.zeros((Z, H, W), dtype=np.uint8)
    gt[0, 0, 0:5] = 1
    pred = np.zeros((Z, H, W), dtype=np.uint8)
    pred[0, 0, 0:1] = 1  # iou = 1/5 = 0.2
    _seed_predictions(run_dir, "case_a",
                      gt_lesion=gt, gt_gland=np.zeros_like(gt))
    _seed_postprocessed(run_dir, "case_a",
                         lesion_mask=pred, gland_mask=np.zeros_like(gt))

    assert evaluate_cli.main([str(run_dir), "--visualize-only", "none"]) == 0
    summary = json.loads(
        (run_dir / "diagnostic" / "evaluation" / "summary.json").read_text()
    )
    assert summary["positives"]["n_detected_lesions"] == 1

    assert evaluate_cli.main([
        str(run_dir), "--correctness-iou", "0.5", "--visualize-only", "none",
    ]) == 0
    summary = json.loads(
        (run_dir / "diagnostic" / "evaluation" / "summary.json").read_text()
    )
    assert summary["positives"]["n_detected_lesions"] == 0


def test_evaluate_cli_errors_when_postprocessed_missing(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    (run_dir / "diagnostic" / "predictions").mkdir(parents=True)

    with pytest.raises(SystemExit, match="postprocess"):
        evaluate_cli.main([str(run_dir), "--visualize-only", "none"])


def test_evaluate_cli_visualize_all_writes_html_per_case(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 4, 8, 8
    gt = np.zeros((Z, H, W), dtype=np.uint8); gt[1, 2, 2] = 1
    pred = np.zeros((Z, H, W), dtype=np.uint8); pred[1, 2, 2] = 1
    _seed_predictions(run_dir, "case_a",
                      gt_lesion=gt, gt_gland=np.zeros_like(gt))
    _seed_postprocessed(run_dir, "case_a",
                         lesion_mask=pred, gland_mask=np.zeros_like(gt))
    gt_neg = np.zeros((1, 8, 8), dtype=np.uint8)
    pred_neg = np.zeros_like(gt_neg)
    _seed_predictions(run_dir, "case_b",
                      gt_lesion=gt_neg, gt_gland=np.zeros_like(gt_neg))
    _seed_postprocessed(run_dir, "case_b",
                         lesion_mask=pred_neg, gland_mask=np.zeros_like(gt_neg))

    rc = evaluate_cli.main([str(run_dir), "--visualize-only", "all"])

    assert rc == 0
    visuals = run_dir / "diagnostic" / "evaluation" / "visuals"
    assert (visuals / "case_a.html").exists()
    assert (visuals / "case_b.html").exists()
    assert (visuals / "index.html").exists()


def test_evaluate_cli_visualize_failed_only_renders_failures(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 1, 10, 10
    gt_pass = np.zeros((Z, H, W), dtype=np.uint8); gt_pass[0, 0, 0] = 1
    pred_pass = np.zeros_like(gt_pass); pred_pass[0, 0, 0] = 1
    _seed_predictions(run_dir, "case_pass",
                      gt_lesion=gt_pass, gt_gland=np.zeros_like(gt_pass))
    _seed_postprocessed(run_dir, "case_pass",
                         lesion_mask=pred_pass,
                         gland_mask=np.zeros_like(gt_pass))
    gt_fail = np.zeros((Z, H, W), dtype=np.uint8); gt_fail[0, 5, 5] = 1
    pred_fail = np.zeros_like(gt_fail)
    _seed_predictions(run_dir, "case_fail",
                      gt_lesion=gt_fail, gt_gland=np.zeros_like(gt_fail))
    _seed_postprocessed(run_dir, "case_fail",
                         lesion_mask=pred_fail,
                         gland_mask=np.zeros_like(gt_fail))

    rc = evaluate_cli.main([str(run_dir), "--visualize-only", "failed"])

    assert rc == 0
    visuals = run_dir / "diagnostic" / "evaluation" / "visuals"
    assert not (visuals / "case_pass.html").exists()
    assert (visuals / "case_fail.html").exists()
    assert (visuals / "index.html").exists()


# --------------------------------------------------------------------------- #
# FROC sweep + operating point (shortlist #3)
# --------------------------------------------------------------------------- #

def _seed_prob(run_dir: Path, case_id: str, *, lesion_prob, gland_prob) -> None:
    """Overwrite predictions/<case>/prob.npz with graded probabilities."""
    pdir = run_dir / "diagnostic" / "predictions" / case_id
    np.savez_compressed(pdir / "prob.npz",
                        gland=gland_prob.astype(np.float32),
                        lesion=lesion_prob.astype(np.float32))


def _seed_froc_case(run_dir: Path) -> None:
    Z, H, W = 3, 8, 8
    gt = np.zeros((Z, H, W), dtype=np.uint8)
    gt[0, 0:2, 0:2] = 1  # one GT lesion on slice 0
    lesion_prob = np.zeros((Z, H, W), dtype=np.float32)
    lesion_prob[0, 0:2, 0:2] = 0.9   # detected once t <= 0.9
    lesion_prob[0, 5:7, 5:7] = 0.7   # false positive once t <= 0.7
    gland_prob = np.ones((Z, H, W), dtype=np.float32)
    pred = np.zeros((Z, H, W), dtype=np.uint8)
    pred[0, 0:2, 0:2] = 1
    _seed_predictions(run_dir, "case_a", gt_lesion=gt, gt_gland=np.zeros_like(gt))
    _seed_prob(run_dir, "case_a", lesion_prob=lesion_prob, gland_prob=gland_prob)
    _seed_postprocessed(run_dir, "case_a",
                        lesion_mask=pred, gland_mask=np.ones_like(gt))


def test_evaluate_cli_froc_writes_curve_operating_point_and_plot(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    _seed_froc_case(run_dir)

    rc = evaluate_cli.main([
        str(run_dir), "--visualize-only", "none",
        "--froc", "--froc-thresholds", "0.8", "0.6",
        "--froc-target-fp", "1.0",
    ])

    assert rc == 0
    eval_dir = run_dir / "diagnostic" / "evaluation"

    # Fixed-threshold outputs are still produced.
    assert (eval_dir / "metrics_by_case.csv").exists()
    assert (eval_dir / "summary.json").exists()

    # FROC artifacts.
    assert (eval_dir / "froc.csv").exists()
    assert (eval_dir / "operating_point.json").exists()
    assert (eval_dir / "froc.png").exists()

    with (eval_dir / "froc.csv").open() as f:
        froc_rows = list(csv.DictReader(f))
    assert list(froc_rows[0].keys()) == [
        "threshold", "sensitivity", "fp_per_case",
        "n_gt_lesions", "n_detected", "n_false_positives", "n_cases",
    ]
    by_t = {r["threshold"]: r for r in froc_rows}
    assert by_t["0.8"]["sensitivity"] == "1.0"
    assert by_t["0.8"]["fp_per_case"] == "0.0"
    assert by_t["0.6"]["fp_per_case"] == "1.0"

    op = json.loads((eval_dir / "operating_point.json").read_text())
    # Budget 1.0: both thresholds reach sensitivity 1.0; fewest-FP wins -> 0.8.
    assert op["operating_point"]["threshold"] == 0.8
    assert op["operating_point"]["target_met"] is True
    assert {d["fp_per_case"] for d in op["sensitivity_at_fp_rates"]} == {
        0.5, 1.0, 2.0, 4.0,
    }


def test_evaluate_cli_froc_off_by_default(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    _seed_froc_case(run_dir)

    assert evaluate_cli.main([str(run_dir), "--visualize-only", "none"]) == 0

    eval_dir = run_dir / "diagnostic" / "evaluation"
    assert (eval_dir / "metrics_by_case.csv").exists()
    assert not (eval_dir / "froc.csv").exists()
    assert not (eval_dir / "operating_point.json").exists()
    assert not (eval_dir / "froc.png").exists()


def test_evaluate_cli_froc_threshold_range(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    _seed_froc_case(run_dir)

    rc = evaluate_cli.main([
        str(run_dir), "--visualize-only", "none",
        "--froc", "--froc-threshold-range", "0.4", "0.8", "0.2",
    ])

    assert rc == 0
    with (run_dir / "diagnostic" / "evaluation" / "froc.csv").open() as f:
        thresholds = [r["threshold"] for r in csv.DictReader(f)]
    # start=0.4, stop=0.8 inclusive, step=0.2 -> 0.4, 0.6, 0.8
    assert thresholds == ["0.4", "0.6", "0.8"]


# --------------------------------------------------------------------------- #
# Official picai_eval scoring + bootstrap CIs (map ticket #12)
# --------------------------------------------------------------------------- #

def _seed_picai_cohort(run_dir: Path) -> None:
    """3 positives detected at 0.9 + 3 clean negatives, graded probs."""
    Z, H, W = 3, 12, 12
    for i in range(3):
        gt = np.zeros((Z, H, W), dtype=np.uint8)
        gt[1, 2:5, 2:5] = 1
        prob = np.zeros((Z, H, W), dtype=np.float32)
        prob[1, 2:5, 2:5] = 0.9
        mask = (prob >= 0.5).astype(np.uint8)
        _seed_predictions(run_dir, f"pos_{i}", gt_lesion=gt, gt_gland=np.zeros_like(gt))
        _seed_prob(run_dir, f"pos_{i}", lesion_prob=prob,
                   gland_prob=np.ones_like(prob))
        _seed_postprocessed(run_dir, f"pos_{i}", lesion_mask=mask,
                            gland_mask=np.ones_like(mask))
    for i in range(3):
        gt = np.zeros((Z, H, W), dtype=np.uint8)
        prob = np.zeros((Z, H, W), dtype=np.float32)
        mask = np.zeros((Z, H, W), dtype=np.uint8)
        _seed_predictions(run_dir, f"neg_{i}", gt_lesion=gt, gt_gland=np.zeros_like(gt))
        _seed_prob(run_dir, f"neg_{i}", lesion_prob=prob,
                   gland_prob=np.ones_like(prob))
        _seed_postprocessed(run_dir, f"neg_{i}", lesion_mask=mask,
                            gland_mask=np.ones_like(mask))


def test_evaluate_cli_picai_smoke_ap_auroc_and_cis_in_summary(tmp_path: Path) -> None:
    """CPU smoke test for the full official-scoring path (ticket #12)."""
    run_dir = _seed_run_dir(tmp_path)
    _seed_picai_cohort(run_dir)

    rc = evaluate_cli.main([
        str(run_dir), "--visualize-only", "none",
        "--bootstrap-iters", "50", "--bootstrap-seed", "0",
    ])

    assert rc == 0
    summary = json.loads(
        (run_dir / "diagnostic" / "evaluation" / "summary.json").read_text()
    )

    picai = summary["picai_eval"]
    assert picai["AP"] == 1.0
    assert picai["auroc"] == 1.0
    assert picai["ranking_score"] == 1.0
    assert picai["num_cases"] == 6
    assert picai["num_lesions"] == 3
    for key in ("AP_ci", "auroc_ci", "ranking_score_ci"):
        assert set(picai[key]) == {"lower", "upper", "n_valid"}
        assert picai[key]["n_valid"] > 0

    # C3 core: the existing cohort metrics carry CIs too.
    assert summary["bootstrap"] == {"n_boot": 50, "seed": 0, "ci_level": 0.95}
    assert summary["positives"]["lesion_recall_ci"]["lower"] == 1.0
    assert set(summary["negatives"]["negative_accuracy_ci"]) == {
        "lower", "upper", "n_valid",
    }


def test_evaluate_cli_no_picai_eval_flag_omits_block(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    _seed_picai_cohort(run_dir)

    rc = evaluate_cli.main([
        str(run_dir), "--visualize-only", "none",
        "--no-picai-eval", "--bootstrap-iters", "0",
    ])

    assert rc == 0
    summary = json.loads(
        (run_dir / "diagnostic" / "evaluation" / "summary.json").read_text()
    )
    assert "picai_eval" not in summary
    assert "bootstrap" not in summary
    assert "lesion_recall_ci" not in summary["positives"]


def test_evaluate_cli_save_detection_maps_writes_per_case_npz(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    _seed_picai_cohort(run_dir)

    rc = evaluate_cli.main([
        str(run_dir), "--visualize-only", "none",
        "--bootstrap-iters", "0", "--save-detection-maps",
    ])

    assert rc == 0
    maps_dir = run_dir / "diagnostic" / "evaluation" / "detection_maps"
    saved = sorted(p.name for p in maps_dir.iterdir())
    assert saved == [
        "neg_0.npz", "neg_1.npz", "neg_2.npz",
        "pos_0.npz", "pos_1.npz", "pos_2.npz",
    ]
    det = np.load(maps_dir / "pos_0.npz")["detection_map"]
    # one component filled with its max confidence, zero elsewhere
    assert set(np.unique(det)) == {np.float32(0.0), np.float32(0.9)}


def test_evaluate_cli_picai_skips_with_warning_when_prob_missing(
    tmp_path: Path,
) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 1, 8, 8
    gt = np.zeros((Z, H, W), dtype=np.uint8); gt[0, 1, 1] = 1
    pred = np.zeros_like(gt); pred[0, 1, 1] = 1
    _seed_predictions(run_dir, "case_a", gt_lesion=gt, gt_gland=np.zeros_like(gt))
    _seed_postprocessed(run_dir, "case_a", lesion_mask=pred,
                        gland_mask=np.zeros_like(gt))
    (run_dir / "diagnostic" / "predictions" / "case_a" / "prob.npz").unlink()

    with pytest.warns(UserWarning, match="picai_eval"):
        rc = evaluate_cli.main([str(run_dir), "--visualize-only", "none",
                                "--bootstrap-iters", "10"])

    assert rc == 0
    summary = json.loads(
        (run_dir / "diagnostic" / "evaluation" / "summary.json").read_text()
    )
    assert "picai_eval" not in summary
    # bootstrap CIs on the fixed-threshold metrics still land
    assert summary["positives"]["lesion_recall_ci"]["n_valid"] > 0
