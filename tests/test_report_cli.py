"""End-to-end CLI tests for `python -m mri.cli.report` (shortlist #2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mri.cli import report as report_cli


def _seed_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_best.pt").write_bytes(b"")
    (run_dir / "resolved_config.yaml").write_text(
        "metrics:\n  segmentation_threshold: 0.5\n"
    )
    return run_dir


def _seed_case(run_dir: Path, case_id: str, *, spacing=None) -> None:
    Z, H, W = 3, 8, 8
    lesion = np.zeros((Z, H, W), dtype=np.uint8)
    lesion[1, 5:7, 3:5] = 1  # posterior -> PZ
    gland = np.ones((Z, H, W), dtype=np.uint8)
    lesion_prob = np.zeros((Z, H, W), dtype=np.float32)
    lesion_prob[1, 5:7, 3:5] = 0.9
    gland_prob = np.ones((Z, H, W), dtype=np.float32)

    pred = run_dir / "diagnostic" / "predictions" / case_id
    pred.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred / "prob.npz", gland=gland_prob, lesion=lesion_prob)
    np.savez_compressed(pred / "gt.npz",
                        gland=np.zeros((Z, H, W), np.uint8),
                        lesion=np.zeros((Z, H, W), np.uint8))
    meta = {"case_id": case_id, "class_label": 2,
            "spatial_shape": [H, W], "num_slices": Z}
    if spacing is not None:
        meta["voxel_spacing_mm"] = list(spacing)
    (pred / "meta.json").write_text(json.dumps(meta))

    post = run_dir / "diagnostic" / "postprocessed" / case_id
    post.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(post / "lesion_mask.npz", mask=lesion)
    np.savez_compressed(post / "gland_mask.npz", mask=gland)
    (post / "meta.json").write_text(json.dumps({
        "case_id": case_id, "lesion_threshold": 0.5, "gland_threshold": 0.5,
    }))


def test_report_cli_writes_html_pdf_and_findings_json(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    _seed_case(run_dir, "case_a", spacing=(3.0, 1.0, 1.0))

    rc = report_cli.main([str(run_dir)])

    assert rc == 0
    rdir = run_dir / "diagnostic" / "report" / "case_a"
    assert (rdir / "report.html").exists()
    assert (rdir / "report.pdf").exists()
    assert (rdir / "findings.json").exists()

    html = (rdir / "report.html").read_text()
    assert "suspicious focus" in html or "suspicious foci" in html  # impression
    assert "PZ" in html          # zone in the table
    assert "High" in html        # suspicion band
    assert "not a diagnostic device" in html.lower()  # disclaimer

    payload = json.loads((rdir / "findings.json").read_text())
    assert payload["case_id"] == "case_a"
    assert len(payload["findings"]) == 1
    assert payload["findings"][0]["suspicion"] == "High"
    assert payload["findings"][0]["max_diameter_mm"] == 2.0  # 2 rows/cols * 1mm

    pdf_head = (rdir / "report.pdf").read_bytes()[:5]
    assert pdf_head == b"%PDF-"


def test_report_cli_spacing_unavailable_omits_mm_with_note(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    _seed_case(run_dir, "case_a", spacing=None)  # no voxel_spacing_mm in meta

    assert report_cli.main([str(run_dir)]) == 0

    rdir = run_dir / "diagnostic" / "report" / "case_a"
    payload = json.loads((rdir / "findings.json").read_text())
    assert payload["spacing_available"] is False
    assert payload["findings"][0]["max_diameter_mm"] is None
    assert payload["findings"][0]["volume_mm3"] is None
    assert "spacing unavailable" in (rdir / "report.html").read_text().lower()


def test_report_cli_negative_case_reads_no_suspicious_focus(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 3, 8, 8
    # A case with an empty lesion mask.
    pred = run_dir / "diagnostic" / "predictions" / "case_neg"
    pred.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred / "prob.npz",
                        gland=np.ones((Z, H, W), np.float32),
                        lesion=np.zeros((Z, H, W), np.float32))
    (pred / "meta.json").write_text(json.dumps({"case_id": "case_neg"}))
    post = run_dir / "diagnostic" / "postprocessed" / "case_neg"
    post.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(post / "lesion_mask.npz", mask=np.zeros((Z, H, W), np.uint8))
    np.savez_compressed(post / "gland_mask.npz", mask=np.ones((Z, H, W), np.uint8))
    (post / "meta.json").write_text(json.dumps({"case_id": "case_neg"}))

    assert report_cli.main([str(run_dir), "--case", "case_neg"]) == 0

    rdir = run_dir / "diagnostic" / "report" / "case_neg"
    payload = json.loads((rdir / "findings.json").read_text())
    assert payload["findings"] == []
    assert payload["impression"] == "No suspicious focus identified."
