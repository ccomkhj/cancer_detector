"""CLI entry point for the PI-RADS-style structured findings report.

Usage::

    python -m mri.cli.report <run_dir> [--case CASE] \\
        [--suspicion-thresholds LOW HIGH]

Reads ``<run_dir>/diagnostic/postprocessed/<case>/{lesion_mask,gland_mask}.npz``
and ``<run_dir>/diagnostic/predictions/<case>/{prob.npz, meta.json}`` and writes
``<run_dir>/diagnostic/report/<case>/{report.html, report.pdf, findings.json}``.

The report is decision support labelled research-use-only, not a diagnostic
device: the zone and suspicion band are coarse model-derived approximations,
not a certified PI-RADS assessment.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mri.cli.diagnose import resolve_run_dir
from mri.inference.findings import (
    DEFAULT_SUSPICION_THRESHOLDS, Finding, build_findings, build_impression,
)


_DISCLAIMER = (
    "Research use only — not a diagnostic device. Zone and suspicion band are "
    "coarse model-derived approximations, not a certified PI-RADS assessment."
)
_SPACING_NOTE = (
    "Voxel spacing unavailable — millimetre size and volume are omitted "
    "rather than estimated."
)


def _banding_note(thresholds: tuple[float, float]) -> str:
    """State the actual suspicion bands so the report never contradicts them."""
    low, high = thresholds
    return (
        "Suspicion bands are derived from the maximum lesion probability: "
        f"Low < {low:g} ≤ Intermediate < {high:g} ≤ High."
    )


@dataclass(frozen=True)
class CaseReport:
    """Everything one case's report renders from — HTML, PDF, and JSON share it."""
    case_id: str
    findings: list[Finding]
    impression: str
    spacing_available: bool
    suspicion_thresholds: tuple[float, float]


def _load_case_arrays(predictions_dir: Path, postprocessed_dir: Path, case_id: str):
    prob = np.load(predictions_dir / case_id / "prob.npz")
    lesion = np.load(postprocessed_dir / case_id / "lesion_mask.npz")["mask"]
    gland = np.load(postprocessed_dir / case_id / "gland_mask.npz")["mask"]
    return lesion, gland, prob["lesion"]


def _read_voxel_spacing(predictions_dir: Path, postprocessed_dir: Path, case_id: str):
    """Read (z, y, x) voxel spacing in mm from case metadata, or ``None``.

    ``voxel_spacing_mm`` is the canonical key across the pipeline. It must be
    persisted into the case ``meta.json`` upstream, at the dicom_mapper
    resampling step that owns spacing (the PNG-based training data does not
    otherwise carry it). Until it is, this returns ``None`` and the report
    omits millimetre size/volume with an explicit note rather than guessing.
    """
    for meta_path in (
        predictions_dir / case_id / "meta.json",
        postprocessed_dir / case_id / "meta.json",
    ):
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        spacing = meta.get("voxel_spacing_mm")
        if spacing and len(spacing) == 3:
            return (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    return None


def _fmt_mm(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "—"


def _finding_row_values(f: Finding) -> tuple[str, str, str, str, str, str, str]:
    """The ordered per-lesion cell strings shared by the HTML and PDF renderers.

    One place defines the column order and formatting so the two renderers can
    never drift.
    """
    return (
        str(f.lesion_id),
        f.zone,
        _fmt_mm(f.max_diameter_mm),
        _fmt_mm(f.volume_mm3),
        f"{f.slice_range[0]}–{f.slice_range[1]} ({f.n_slices})",
        f"{f.max_probability:.3f}",
        f.suspicion,
    )


_COLUMNS = ("Lesion", "Zone", "Max Ø (mm)", "Volume (mm³)",
            "Slices (n)", "Max prob", "Suspicion")


def _build_payload(report: CaseReport) -> dict:
    return {
        "case_id": report.case_id,
        "impression": report.impression,
        "spacing_available": report.spacing_available,
        "findings": [asdict(f) for f in report.findings],
        "disclaimer": _DISCLAIMER,
    }


def _render_html(report: CaseReport) -> str:
    if report.findings:
        rows = "".join(
            "<tr>" + "".join(f"<td>{html_lib.escape(v)}</td>"
                             for v in _finding_row_values(f)) + "</tr>"
            for f in report.findings
        )
        head = "".join(f"<th>{html_lib.escape(c)}</th>" for c in _COLUMNS)
        table = f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
    else:
        table = "<p class='empty'>No suspicious focus identified.</p>"

    spacing_html = (
        "" if report.spacing_available
        else f"<div class='note'>{html_lib.escape(_SPACING_NOTE)}</div>"
    )

    style = (
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "margin:24px;color:#222;background:#f7f7f8;}"
        "h1{font-size:20px;margin:0 0 12px;}"
        ".disclaimer{background:#fff4e5;border:1px solid #f9a825;color:#6b3b00;"
        "border-radius:8px;padding:10px 14px;font-size:13px;max-width:900px;}"
        ".impression{background:#fff;border:1px solid #e0e0e0;border-radius:8px;"
        "padding:12px 16px;margin:12px 0;max-width:900px;font-size:15px;}"
        ".note{color:#6b3b00;font-size:13px;margin:8px 0;}"
        "table{border-collapse:collapse;background:#fff;margin-top:8px;font-size:13px;}"
        "th,td{border:1px solid #e0e0e0;padding:6px 10px;text-align:left;}"
        "th{background:#f0f0f2;font-variant:all-small-caps;letter-spacing:.5px;}"
        ".empty{background:#e8f5e9;border:1px solid #66bb6a;color:#1b5e20;"
        "border-radius:8px;padding:12px 16px;max-width:900px;}"
        ".banding{color:#555;font-size:12px;margin-top:10px;}"
    )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Findings report — {html_lib.escape(report.case_id)}</title>"
        f"<style>{style}</style></head><body>"
        f"<h1>Structured findings report — {html_lib.escape(report.case_id)}</h1>"
        f"<div class='disclaimer'>{html_lib.escape(_DISCLAIMER)}</div>"
        f"<div class='impression'><b>Impression:</b> "
        f"{html_lib.escape(report.impression)}</div>"
        f"{spacing_html}"
        f"{table}"
        f"<div class='banding'>"
        f"{html_lib.escape(_banding_note(report.suspicion_thresholds))}</div>"
        "</body></html>"
    )


def _render_pdf(report: CaseReport, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    lines = [
        f"Structured findings report — {report.case_id}",
        "",
        _DISCLAIMER,
        "",
        f"Impression: {report.impression}",
        "",
    ]
    if not report.spacing_available:
        lines += [_SPACING_NOTE, ""]
    if report.findings:
        lines.append("  ".join(f"{c:<12}" for c in _COLUMNS))
        for f in report.findings:
            lines.append("  ".join(f"{v:<12}" for v in _finding_row_values(f)))
    else:
        lines.append("No suspicious focus identified.")
    lines += ["", _banding_note(report.suspicion_thresholds)]

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.06, 0.94, "\n".join(lines), va="top", ha="left",
             fontsize=9, family="monospace")
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def _write_case_report(
    *, predictions_dir: Path, postprocessed_dir: Path, report_dir: Path,
    case_id: str, suspicion_thresholds: tuple[float, float],
) -> None:
    lesion, gland, lesion_prob = _load_case_arrays(
        predictions_dir, postprocessed_dir, case_id,
    )
    spacing = _read_voxel_spacing(predictions_dir, postprocessed_dir, case_id)
    findings = build_findings(
        lesion, gland, lesion_prob,
        voxel_spacing_mm=spacing, suspicion_thresholds=suspicion_thresholds,
    )
    report = CaseReport(
        case_id=case_id,
        findings=findings,
        impression=build_impression(findings),
        spacing_available=spacing is not None,
        suspicion_thresholds=suspicion_thresholds,
    )

    out = report_dir / case_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "findings.json").write_text(json.dumps(_build_payload(report), indent=2))
    (out / "report.html").write_text(_render_html(report), encoding="utf-8")
    _render_pdf(report, out / "report.pdf")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PI-RADS-style structured findings report from postprocessed masks.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--case", default=None,
        help="Report a single case id (default: every postprocessed case).",
    )
    parser.add_argument(
        "--suspicion-thresholds", type=float, nargs=2,
        default=DEFAULT_SUSPICION_THRESHOLDS,
        metavar=("LOW", "HIGH"),
        help="Probability cutoffs for Low/Intermediate/High suspicion bands.",
    )
    args = parser.parse_args(argv)

    paths = resolve_run_dir(args.run_dir)
    diag_root = paths.run_dir / "diagnostic"
    predictions_dir = diag_root / "predictions"
    postprocessed_dir = diag_root / "postprocessed"
    report_dir = diag_root / "report"

    if not postprocessed_dir.exists() or not any(postprocessed_dir.iterdir()):
        raise SystemExit(
            f"[report] no postprocessed predictions at {postprocessed_dir}. "
            "Run `python -m mri.cli.postprocess <run_dir>` first."
        )

    if args.case is not None:
        case_ids = [args.case]
    else:
        case_ids = sorted(p.name for p in postprocessed_dir.iterdir() if p.is_dir())

    thresholds = (
        float(args.suspicion_thresholds[0]), float(args.suspicion_thresholds[1]),
    )
    for case_id in case_ids:
        _write_case_report(
            predictions_dir=predictions_dir,
            postprocessed_dir=postprocessed_dir,
            report_dir=report_dir,
            case_id=case_id,
            suspicion_thresholds=thresholds,
        )

    print(f"[report] wrote report/ for {len(case_ids)} case(s) to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
