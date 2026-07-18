"""Tests for the DICOM-SEG/SR export orchestration CLI (shortlist #1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from mri.cli import export_dicom
from mri.inference.findings import Finding


# --------------------------------------------------------------------------- #
# Pure mapping: Finding -> dicom_mapper LesionMeasurement
# --------------------------------------------------------------------------- #

def test_findings_to_measurements_maps_size_suspicion_and_slice() -> None:
    findings = [
        Finding(lesion_id=2, zone="PZ", slice_range=(3, 5), n_slices=3,
                voxels=40, max_probability=0.9, suspicion="High",
                centroid=(4.0, 6.0, 5.0), max_diameter_mm=14.0, volume_mm3=120.0),
    ]

    measurements = export_dicom.findings_to_measurements(findings)

    assert len(measurements) == 1
    m = measurements[0]
    assert m.tracking_id == "lesion-2"
    assert m.largest_diameter_mm == 14.0
    assert m.suspicion == "High"
    assert m.slice_index == 4            # round(centroid z)
    assert m.centroid_xy == (5.0, 6.0)   # (x=col, y=row)


def test_findings_to_measurements_defaults_mm_when_missing() -> None:
    findings = [
        Finding(lesion_id=1, zone="other", slice_range=(1, 1), n_slices=1,
                voxels=4, max_probability=0.4, suspicion="Intermediate",
                centroid=(1.0, 2.0, 3.0), max_diameter_mm=None, volume_mm3=None),
    ]

    m = export_dicom.findings_to_measurements(findings)[0]
    assert m.largest_diameter_mm == 0.0


def test_align_mask_to_source_matches_source_grid_count_and_dims() -> None:
    # Mask on a coarser grid than the source series.
    label = np.zeros((2, 4, 4), dtype=np.uint8)
    label[0, 0:2, 0:2] = 1
    label[1, 2:4, 2:4] = 2

    aligned = export_dicom.align_mask_to_source(label, n=3, rows=8, cols=8)

    assert aligned.shape == (3, 8, 8)  # matches the source count + in-plane dims
    # Same-shape input is returned untouched (the common native-grid case).
    same = export_dicom.align_mask_to_source(label, n=2, rows=4, cols=4)
    assert np.array_equal(same, label)


def test_build_label_mask_lesion_wins_overlap() -> None:
    gland = np.zeros((2, 4, 4), dtype=np.uint8)
    gland[0, 0:3, 0:3] = 1
    lesion = np.zeros((2, 4, 4), dtype=np.uint8)
    lesion[0, 1:2, 1:2] = 1  # inside the gland

    label = export_dicom.build_label_mask(gland, lesion)

    assert label[0, 0, 0] == 1   # gland only
    assert label[0, 1, 1] == 2   # lesion wins the overlap


# --------------------------------------------------------------------------- #
# CLI integration
# --------------------------------------------------------------------------- #

def _write_source_series(out_dir: Path, n=3, rows=8, cols=8, sz=3.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    for i in range(n):
        fm = FileMetaDataset()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.MediaStorageSOPClassUID = MRImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        ds = Dataset()
        ds.file_meta = fm
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series
        ds.FrameOfReferenceUID = frame
        ds.PatientID = "P1"; ds.PatientName = "Test^Case"
        ds.PatientBirthDate = ""; ds.PatientSex = ""
        ds.StudyDate = "20240101"; ds.StudyTime = "000000"
        ds.AccessionNumber = ""; ds.StudyID = ""; ds.SeriesNumber = 1
        ds.ReferringPhysicianName = ""
        ds.ContentDate = "20240101"; ds.ContentTime = "000000"
        ds.Modality = "MR"; ds.Rows = rows; ds.Columns = cols
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * sz]
        ds.PixelSpacing = [1.0, 1.0]; ds.SliceThickness = sz
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = np.zeros((rows, cols), dtype=np.uint16).tobytes()
        ds.InstanceNumber = i + 1
        ds.save_as(out_dir / f"{i:04d}.dcm", enforce_file_format=True)


def _seed_case(run_dir: Path, case_id: str) -> None:
    Z, H, W = 3, 8, 8
    gland = np.zeros((Z, H, W), dtype=np.uint8); gland[:, 2:6, 2:6] = 1
    lesion = np.zeros((Z, H, W), dtype=np.uint8); lesion[1, 3:5, 3:5] = 1
    lesion_prob = np.zeros((Z, H, W), dtype=np.float32); lesion_prob[1, 3:5, 3:5] = 0.9

    pred = run_dir / "diagnostic" / "predictions" / case_id
    pred.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred / "prob.npz",
                        gland=np.ones((Z, H, W), np.float32), lesion=lesion_prob)
    (pred / "meta.json").write_text(json.dumps(
        {"case_id": case_id, "voxel_spacing_mm": [3.0, 1.0, 1.0]}))

    post = run_dir / "diagnostic" / "postprocessed" / case_id
    post.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(post / "lesion_mask.npz", mask=lesion)
    np.savez_compressed(post / "gland_mask.npz", mask=gland)


def _seed_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_best.pt").write_bytes(b"")
    (run_dir / "resolved_config.yaml").write_text(
        "metrics:\n  segmentation_threshold: 0.5\n"
    )
    return run_dir


def test_export_writes_readable_seg_and_sr(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    _seed_case(run_dir, "case_a")
    src_dir = tmp_path / "t2_source"
    _write_source_series(src_dir, n=3, rows=8, cols=8)

    rc = export_dicom.main([
        str(run_dir), "--case", "case_a",
        "--source-series", str(src_dir), "--seg", "--sr",
    ])

    assert rc == 0
    out = run_dir / "diagnostic" / "dicom"
    seg_path = out / "case_a_seg.dcm"
    sr_path = out / "case_a_sr.dcm"
    assert seg_path.exists() and sr_path.exists()

    seg = pydicom.dcmread(seg_path)
    assert seg.Modality == "SEG"
    assert len(seg.SegmentSequence) == 2  # gland + lesion
    assert "not a diagnostic device" in seg.ImageComments.lower()

    sr = pydicom.dcmread(sr_path)
    assert sr.Modality == "SR"
    assert "not a diagnostic device" in sr.ImageComments.lower()


def test_export_seg_only_when_seg_flag(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    _seed_case(run_dir, "case_a")
    src_dir = tmp_path / "t2_source"
    _write_source_series(src_dir, n=3, rows=8, cols=8)

    assert export_dicom.main([
        str(run_dir), "--case", "case_a",
        "--source-series", str(src_dir), "--seg",
    ]) == 0

    out = run_dir / "diagnostic" / "dicom"
    assert (out / "case_a_seg.dcm").exists()
    assert not (out / "case_a_sr.dcm").exists()
