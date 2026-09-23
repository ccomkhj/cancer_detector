"""Official PI-CAI (`picai_eval`) scoring + bootstrap confidence intervals.

Bridges the repo's postprocessed predictions to DIAGNijmegen's `picai_eval`
package: builds the detection maps it expects (non-overlapping 3D connected
components, one confidence value each), runs the official lesion-AP /
case-AUROC / ranking-score computation, and attaches case-level bootstrap
CIs to every metric. The metric values themselves always come from
`picai_eval`; only the resampling loop is local, so degenerate resamples
(e.g. a single-class draw on a small cohort) are dropped instead of
poisoning the percentiles with NaN.

Pure NumPy + picai_eval. The CLI in mri/cli/evaluate.py does the I/O.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mri.diagnostics.detection import CaseRow, label_lesion_components


def lesion_detection_map(
    pred_lesion_mask: np.ndarray,
    lesion_prob: np.ndarray,
    *,
    connectivity_rank: int = 1,
) -> np.ndarray:
    """Convert a postprocessed lesion mask into a picai_eval detection map.

    Each 3D connected component of ``pred_lesion_mask`` is filled with a
    single confidence value: the maximum ``lesion_prob`` inside the
    component (consistent with the case score being the map's maximum).
    Components are non-overlapping by construction.

    Args:
      pred_lesion_mask: (Z, H, W) binary postprocessed lesion voxels.
      lesion_prob: (Z, H, W) float lesion probability volume.
      connectivity_rank: component connectivity, as in
          :func:`mri.diagnostics.detection.label_lesion_components`.

    Returns:
      (Z, H, W) float32 detection map; 0.0 outside all components.
    """
    assert pred_lesion_mask.shape == lesion_prob.shape, (
        f"shape mismatch: mask {pred_lesion_mask.shape} vs prob {lesion_prob.shape}"
    )
    labels, n = label_lesion_components(
        pred_lesion_mask, connectivity_rank=connectivity_rank,
    )
    detection_map = np.zeros(pred_lesion_mask.shape, dtype=np.float32)
    for k in range(1, n + 1):
        component = labels == k
        detection_map[component] = float(lesion_prob[component].max())
    return detection_map


@dataclass(frozen=True)
class BootstrapCI:
    """A percentile bootstrap confidence interval for one metric.

    ``lower``/``upper`` are None when no resample produced a finite value
    (e.g. AUROC on a cohort with a single class). ``n_valid`` counts the
    finite replicates the percentiles were taken over.
    """
    lower: float | None
    upper: float | None
    n_valid: int


def percentile_ci(
    replicates: Sequence[float],
    *,
    ci_level: float = 0.95,
) -> BootstrapCI:
    """Percentile CI over bootstrap replicates, dropping non-finite values."""
    values = np.asarray(replicates, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return BootstrapCI(lower=None, upper=None, n_valid=0)
    alpha = (1.0 - ci_level) / 2.0
    lower, upper = np.percentile(values, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return BootstrapCI(lower=float(lower), upper=float(upper), n_valid=int(values.size))


def _ci_as_json(ci: BootstrapCI) -> dict[str, Any]:
    return {"lower": ci.lower, "upper": ci.upper, "n_valid": ci.n_valid}


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


@dataclass(frozen=True)
class PicaiCaseInput:
    """One case's arrays for official picai_eval scoring."""
    case_id: str
    gt_lesion: np.ndarray        # (Z, H, W) binary ground-truth lesion voxels
    detection_map: np.ndarray    # (Z, H, W) float32, from lesion_detection_map()


def score_with_picai_eval(
    cases: Iterable[PicaiCaseInput],
    *,
    min_overlap: float = 0.10,
    n_boot: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Run official picai_eval scoring with bootstrap CIs on every metric.

    Metrics: lesion AP, case AUROC (case score = max of the detection map,
    the picai_eval default), and the PI-CAI ranking score ((AP + AUROC) / 2,
    ``Metrics.score``). CIs are case-level percentile bootstraps; each
    replicate's metric is computed by picai_eval's own ``calc_AP`` /
    ``calc_auroc`` on the resampled subject list, and the ranking replicate
    is the mean of the paired AP/AUROC replicates.

    Returns the JSON-ready metrics block. Set ``n_boot=0`` to skip CIs.
    """
    from picai_eval import evaluate as picai_evaluate

    cases = list(cases)
    if not cases:
        raise ValueError("cannot run picai_eval scoring on an empty case list")

    metrics = picai_evaluate(
        y_det=[c.detection_map for c in cases],
        y_true=[c.gt_lesion.astype(np.uint8) for c in cases],
        subject_list=[c.case_id for c in cases],
        min_overlap=min_overlap,
    )

    block: dict[str, Any] = {
        "AP": _finite_or_none(metrics.AP),
        "auroc": _finite_or_none(metrics.auroc),
        "ranking_score": _finite_or_none(metrics.score),
        "num_cases": int(metrics.num_cases),
        "num_lesions": int(metrics.num_lesions),
        "params": {
            "min_overlap": min_overlap,
            "case_confidence": "max",
            "picai_eval_version": metrics.version,
            "n_boot": n_boot,
            "bootstrap_seed": seed,
            "ci_level": ci_level,
        },
    }

    if n_boot > 0:
        rng = np.random.default_rng(seed)
        subjects = list(metrics.subject_list)
        ap_reps: list[float] = []
        auroc_reps: list[float] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(n_boot):
                sample = rng.choice(subjects, size=len(subjects), replace=True)
                # A degenerate resample (e.g. no lesions or a single case
                # class) makes the metric undefined: sklearn either returns
                # NaN or raises. Both count as an invalid replicate.
                try:
                    ap_reps.append(float(metrics.calc_AP(subject_list=sample)))
                except ValueError:
                    ap_reps.append(float("nan"))
                try:
                    auroc_reps.append(
                        float(metrics.calc_auroc(subject_list=sample)),
                    )
                except ValueError:
                    auroc_reps.append(float("nan"))
        ranking_reps = [
            (ap + auroc) / 2.0
            if np.isfinite(ap) and np.isfinite(auroc) else float("nan")
            for ap, auroc in zip(ap_reps, auroc_reps)
        ]
        block["AP_ci"] = _ci_as_json(percentile_ci(ap_reps, ci_level=ci_level))
        block["auroc_ci"] = _ci_as_json(percentile_ci(auroc_reps, ci_level=ci_level))
        block["ranking_score_ci"] = _ci_as_json(
            percentile_ci(ranking_reps, ci_level=ci_level),
        )

    return block


def bootstrap_case_metric_cis(
    case_rows: Iterable[CaseRow],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Case-level bootstrap CIs for the fixed-threshold cohort metrics.

    Resamples the full case list with replacement; each replicate recomputes
    lesion recall over its positive cases and negative accuracy over its
    negative cases (a replicate with none of that kind contributes no value
    to that metric's percentiles).

    Returns ``{"lesion_recall_ci": ..., "negative_accuracy_ci": ...}``
    JSON-ready blocks.
    """
    case_rows = list(case_rows)
    n_gt = np.array([c.n_gt_lesions for c in case_rows], dtype=np.int64)
    n_det = np.array([c.n_detected_lesions for c in case_rows], dtype=np.int64)
    is_neg = np.array([c.case_kind == "negative" for c in case_rows], dtype=bool)
    neg_ok = np.array(
        [bool(c.negative_correct) for c in case_rows], dtype=bool,
    )

    rng = np.random.default_rng(seed)
    recall_reps: list[float] = []
    neg_acc_reps: list[float] = []
    n_cases = len(case_rows)
    for _ in range(n_boot):
        idx = rng.integers(0, n_cases, size=n_cases) if n_cases else np.array([], int)
        gt_total = int(n_gt[idx].sum())
        recall_reps.append(
            (int(n_det[idx].sum()) / gt_total) if gt_total > 0 else float("nan")
        )
        neg_idx = idx[is_neg[idx]]
        neg_acc_reps.append(
            (int(neg_ok[neg_idx].sum()) / neg_idx.size)
            if neg_idx.size > 0 else float("nan")
        )

    return {
        "lesion_recall_ci": _ci_as_json(percentile_ci(recall_reps, ci_level=ci_level)),
        "negative_accuracy_ci": _ci_as_json(
            percentile_ci(neg_acc_reps, ci_level=ci_level),
        ),
    }


def build_bootstrap_params(
    *, n_boot: int, seed: int, ci_level: float = 0.95,
) -> Mapping[str, Any]:
    """The bootstrap parameter block recorded next to any CI outputs."""
    return {"n_boot": n_boot, "seed": seed, "ci_level": ci_level}
