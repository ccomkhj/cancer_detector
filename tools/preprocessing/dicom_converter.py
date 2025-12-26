#!/usr/bin/env python3
"""
DICOM Converter for NBIA-style DICOM trees.

Traverse an NBIA-style DICOM tree, convert each series to per-slice images,
and (optionally) per-slice masks from DICOM SEG or existing labelmaps.

Output structure:
  out_dir/
    case_{case_id}/
      {SeriesInstanceUID}/
        meta.json
        images/
          0000.png
          0001.png
          ...
        labels/                 (if single-class) OR labels_{CLASS}/
          0000.png
        volume.nii.gz           (optional)
        labels.nii.gz           (optional, or labels_{CLASS}.nii.gz)
  manifest.csv

Usage:
    # Single class
    python dicom_converter.py --class 1

    # All classes (defaults to T2 + ep2d_adc + ep2d_calc sequences)
    python dicom_converter.py --all

    # Or import and use programmatically
    from dicom_converter import DicomConverter, ConverterConfig
"""

import os
import sys
import csv
import math
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Sequence

import numpy as np
from PIL import Image
import pydicom
from pydicom.uid import SegmentationStorage
import SimpleITK as sitk
from tqdm import tqdm

# Optional readers for segmentation objects
try:
    import highdicom as hd

    _HAS_HIGHDICOM = True
except Exception:
    _HAS_HIGHDICOM = False


@dataclass
class ConverterConfig:
    root_dir: Path
    out_dir: Path
    image_format: str = "png"  # "png" | "jpg"
    resample_spacing: Optional[Tuple[float, float, float]] = None
    save_nifti: bool = False
    window: Optional[Tuple[float, float]] = None  # (lo, hi)
    auto_window: bool = True
    percentile_window: Tuple[float, float] = (1.0, 99.0)  # used if auto_window
    write_per_class_masks: bool = True
    anonymize_filenames: bool = True  # avoid PHI in names


class DicomConverter:
    """
    Traverse an NBIA-style DICOM tree, convert each series to per-slice images,
    and (optionally) per-slice masks from DICOM SEG or existing labelmaps.
    """

    def __init__(self, config: ConverterConfig) -> None:
        self.cfg = config
        self.cfg.root_dir = Path(self.cfg.root_dir)
        self.cfg.out_dir = Path(self.cfg.out_dir)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            stream=sys.stdout,
        )
        self.manifest_rows: List[Dict[str, str]] = []

    # -------------------------
    # Public API
    # -------------------------

    def convert_all(self) -> None:
        series_dirs = self._discover_series_dirs(self.cfg.root_dir)
        logging.info(f"Found {len(series_dirs)} DICOM series directories.")

        # Use tqdm for progress tracking
        for sd in tqdm(series_dirs, desc="Converting series", unit="series"):
            try:
                self._convert_series_dir(sd)
            except Exception as e:
                logging.exception(f"Failed series at {sd}: {e}")

        self._write_manifest()

    # -------------------------
    # Core steps
    # -------------------------

    def _convert_series_dir(self, series_dir: Path) -> None:
        # 1) Read volume from this series
        image_sitk, meta = self._read_volume_from_series(series_dir)
        if image_sitk is None:
            logging.warning(f"Skipping (no volume) -> {series_dir}")
            return

        # 2) Optional resampling to target spacing
        if self.cfg.resample_spacing is not None:
            image_sitk = self._resample_image(
                image_sitk, self.cfg.resample_spacing, sitk.sitkLinear
            )

        # 3) Intensity transform -> uint8 0..255
        vol_u8 = self._to_uint8(image_sitk)

        # 4) Find segmentation, if any
        label_info = self._load_any_labels(series_dir, image_sitk, meta)

        # 5) Prepare output tree
        case_id = self._case_id_from_path(series_dir)
        series_uid = meta.get("SeriesInstanceUID", "unknown_series")
        safe_case = self._safe_id(case_id) if self.cfg.anonymize_filenames else case_id

        out_base = self.cfg.out_dir / f"case_{safe_case}" / series_uid
        (out_base / "images").mkdir(parents=True, exist_ok=True)

        # 6) Save slices
        vol_np = sitk.GetArrayFromImage(vol_u8)  # (Z, Y, X)
        depth = vol_np.shape[0]

        # optional: save NIfTI
        if self.cfg.save_nifti:
            self._write_nifti(image_sitk, out_base / "volume.nii.gz")

        # Precompute labels (None or dict[class_name or 'mask'] -> sitk.Image)
        labels_to_save: Dict[str, Optional[sitk.Image]] = {}
        if label_info is not None:
            for name, lab_img in label_info.items():
                # Resample labels to image grid if needed
                lab_img = self._match_geometry(lab_img, image_sitk)
                # Ensure binary uint8
                lab_img = sitk.Cast(
                    sitk.BinaryThreshold(lab_img, 0.5, 1e6, 255, 0), sitk.sitkUInt8
                )
                labels_to_save[name] = lab_img
                if self.cfg.save_nifti:
                    self._write_nifti(lab_img, out_base / f"labels_{name}.nii.gz")
        else:
            labels_to_save = {}

        # meta.json
        meta_out = {
            **meta,
            "output_images": str((out_base / "images").resolve()),
            "num_slices": depth,
        }
        (out_base / "meta.json").write_text(json.dumps(meta_out, indent=2))

        # 7) Write per-slice images (and masks) + collect manifest rows
        for z in range(depth):
            img_path = out_base / "images" / f"{z:04d}.{self.cfg.image_format}"
            Image.fromarray(vol_np[z]).save(img_path)

            # default single-class mask folder if there's exactly one label
            if len(labels_to_save) == 1:
                only_name = next(iter(labels_to_save))
                lab_folder = out_base / "labels"
                lab_folder.mkdir(exist_ok=True)
                mask_path = lab_folder / f"{z:04d}.png"
                arr = sitk.GetArrayFromImage(labels_to_save[only_name])[z]
                Image.fromarray(arr).save(mask_path)
                mask_paths = str(mask_path)
            elif len(labels_to_save) > 1:
                paths = []
                for name, lab_img in labels_to_save.items():
                    lab_folder = out_base / f"labels_{self._safe_id(name)}"
                    lab_folder.mkdir(exist_ok=True)
                    mask_path = lab_folder / f"{z:04d}.png"
                    arr = sitk.GetArrayFromImage(lab_img)[z]
                    Image.fromarray(arr).save(mask_path)
                    paths.append(str(mask_path))
                mask_paths = "|".join(paths)
            else:
                mask_paths = ""

            self.manifest_rows.append(
                {
                    "case_id": safe_case,
                    "series_uid": series_uid,
                    "slice_idx": str(z),
                    "image_path": str(img_path),
                    "mask_path": mask_paths,
                    "num_labels": str(len(labels_to_save)),
                    "spacing_x": str(image_sitk.GetSpacing()[0]),
                    "spacing_y": str(image_sitk.GetSpacing()[1]),
                    "spacing_z": str(image_sitk.GetSpacing()[2]),
                    "modality": meta.get("Modality", ""),
                    "study_date": meta.get("StudyDate", ""),
                    "manufacturer": meta.get("Manufacturer", ""),
                }
            )

        logging.info(
            f"Done series: case={safe_case} series={series_uid} slices={depth} labels={len(labels_to_save)}"
        )

    # -------------------------
    # Discovery & reading
    # -------------------------

    def _discover_series_dirs(self, root: Path) -> List[Path]:
        # A series dir is any dir containing multiple .dcm files (>= 3)
        series_dirs = []
        for dirpath, dirnames, filenames in os.walk(root):
            dcm_cnt = sum(1 for f in filenames if f.lower().endswith(".dcm"))
            if dcm_cnt >= 3:
                series_dirs.append(Path(dirpath))
        return series_dirs

    def _read_volume_from_series(self, series_dir: Path):
        reader = sitk.ImageSeriesReader()
        # Let SimpleITK find the series files in this directory
        series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
        if not series_ids:
            return None, {}

        # Pick the first series in this folder (NBIA typically has one)
        file_names = reader.GetGDCMSeriesFileNames(str(series_dir), series_ids[0])
        reader.SetFileNames(file_names)
        image = reader.Execute()  # sitk.Image

        # Grab representative DICOM header for metadata
        ds = pydicom.dcmread(file_names[0], stop_before_pixels=True)
        meta = {
            "PatientID": getattr(ds, "PatientID", ""),
            "StudyInstanceUID": getattr(ds, "StudyInstanceUID", ""),
            "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", ""),
            "StudyDate": getattr(ds, "StudyDate", ""),
            "Modality": getattr(ds, "Modality", ""),
            "Manufacturer": getattr(ds, "Manufacturer", ""),
        }
        return image, meta

    # -------------------------
    # Labels
    # -------------------------

    def _load_any_labels(
        self, series_dir: Path, image_sitk: sitk.Image, meta: Dict
    ) -> Optional[Dict[str, sitk.Image]]:
        """Try, in order:
        1) DICOM SEG somewhere in the same case subtree (if highdicom available)
        2) NIfTI/NRRD labelmaps in the same folder
        Returns dict: {label_name: sitk.Image (binary)} or None.
        """
        # 1) DICOM SEG
        if _HAS_HIGHDICOM:
            seg_paths = self._find_dicom_seg_candidates(series_dir)
            if seg_paths:
                try:
                    return self._read_dicom_seg(seg_paths, image_sitk, meta)
                except Exception as e:
                    logging.warning(
                        f"SEG present but failed to decode at {seg_paths[0]}: {e}"
                    )

        # 2) NIfTI/NRRD labelmaps with known patterns (same dir)
        lm = self._find_labelmaps(series_dir)
        if lm:
            out = {}
            for path in lm:
                name = path.stem.replace(".nii", "")
                lab = sitk.ReadImage(str(path))
                out[name] = lab
            return out

        # None
        return None

    def _find_dicom_seg_candidates(self, series_dir: Path) -> List[Path]:
        # Look in current and up to 3 parents for .dcm that are Segmentation Storage
        candidates = []
        search_roots = [series_dir, series_dir.parent, series_dir.parent.parent]
        for root in search_roots:
            if root is None:
                continue
            for f in root.rglob("*.dcm"):
                try:
                    ds = pydicom.dcmread(
                        str(f), stop_before_pixels=True, specific_tags=["SOPClassUID"]
                    )
                    if getattr(ds, "SOPClassUID", None) == SegmentationStorage:
                        candidates.append(f)
                except Exception:
                    pass
        return candidates

    def _read_dicom_seg(
        self, seg_paths: Sequence[Path], ref_image: sitk.Image, meta: Dict
    ) -> Dict[str, sitk.Image]:
        """Decode first matching DICOM SEG whose ReferencedSeries matches this series."""
        # Take first that references this SeriesInstanceUID
        ref_series_uid = meta.get("SeriesInstanceUID")
        for seg_path in seg_paths:
            ds = pydicom.dcmread(str(seg_path))
            try:
                seg = hd.seg.Segmentation.from_dataset(ds)  # highdicom object
            except Exception as e:
                logging.debug(f"Skipping non-seg or unreadable seg {seg_path}: {e}")
                continue

            # Check reference
            ref_uids = {
                str(x.ReferencedSOPInstanceUID)
                for f in seg.functional_groups
                for x in getattr(f, "DerivationImageSequence", [])
            }
            # If we cannot extract, accept and hope the geometry matches; otherwise check FrameOfReference
            if ref_series_uid and ref_series_uid not in ref_uids:
                # fallback: accept anyway; many SEG reference instances not series
                pass

            # Build label volumes per segment number
            # seg.pixel_array shape: (num_frames, rows, cols)
            arr = seg.pixel_array.astype(np.uint8)
            num_frames, rows, cols = arr.shape

            # Try to infer depth from ref image:
            depth = ref_image.GetSize()[2]

            # Frame -> slice index mapping (assume contiguous slices)
            # NOTE: For robust mapping, parse PerFrameFunctionalGroups. Here we do naive stack.
            if num_frames % depth != 0:
                logging.warning(
                    f"SEG frames ({num_frames}) not divisible by volume depth ({depth}). "
                    f"Will best-effort stack (may be misaligned)."
                )
            frames_per_slice = num_frames // depth if depth > 0 else num_frames

            # Build per-segment masks
            out: Dict[str, sitk.Image] = {}
            # Iterate segments in the dataset
            for seg_item in seg.segment_descriptions:
                seg_number = int(seg_item.SegmentNumber)
                seg_name = seg_item.SegmentLabel or f"segment_{seg_number}"

                # Build a (Z,Y,X) mask for this segment
                mask_z = []
                for z in range(depth):
                    start = z * frames_per_slice
                    end = (z + 1) * frames_per_slice
                    # any frame > 0 means this voxel is inside
                    slice_mask = (arr[start:end] == seg_number).any(axis=0).astype(
                        np.uint8
                    ) * 1
                    mask_z.append(slice_mask)

                mask_vol = np.stack(mask_z, axis=0)  # (Z,Y,X)

                # Convert to sitk.Image using same geometry as ref_image
                mask_img = sitk.GetImageFromArray(mask_vol)
                mask_img.CopyInformation(ref_image)  # origin/spacing/direction

                out[seg_name] = mask_img

            if out:
                return out

        raise RuntimeError("No usable DICOM SEG matched the series or decoding failed.")

    def _find_labelmaps(self, series_dir: Path) -> List[Path]:
        # Look for labelmaps user might have exported via Slicer
        patterns = ["*.nii", "*.nii.gz", "*.nrrd"]
        found = []
        for pat in patterns:
            found.extend(series_dir.glob(pat))
        return found

    # -------------------------
    # Geometry helpers
    # -------------------------

    def _match_geometry(self, img: sitk.Image, ref: sitk.Image) -> sitk.Image:
        """Return img resampled onto ref geometry if size/spacing/origin/direction differ."""
        same_size = img.GetSize() == ref.GetSize()
        same_sp = np.allclose(img.GetSpacing(), ref.GetSpacing(), rtol=0, atol=1e-6)
        same_dir = np.allclose(
            list(img.GetDirection()), list(ref.GetDirection()), rtol=0, atol=1e-6
        )
        same_org = np.allclose(img.GetOrigin(), ref.GetOrigin(), rtol=0, atol=1e-4)

        if same_size and same_sp and same_dir and same_org:
            return img

        resampled = sitk.Resample(
            img,
            ref,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,  # labels
            0,
            img.GetPixelID(),
        )
        return resampled

    def _resample_image(
        self, img: sitk.Image, spacing_xyz: Tuple[float, float, float], interp
    ) -> sitk.Image:
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        new_spacing = spacing_xyz

        new_size = [
            int(round(osz * (ospc / nspc)))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]

        return sitk.Resample(
            img,
            new_size,
            sitk.Transform(),
            interp,
            img.GetOrigin(),
            new_spacing,
            img.GetDirection(),
            0,
            img.GetPixelID(),
        )

    # -------------------------
    # Intensity & saving
    # -------------------------

    def _to_uint8(self, img: sitk.Image) -> sitk.Image:
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)

        # Windowing
        if self.cfg.window is not None:
            lo, hi = self.cfg.window
        else:
            lo, hi = (
                np.percentile(arr[np.isfinite(arr)], list(self.cfg.percentile_window))
                if self.cfg.auto_window
                else (arr.min(), arr.max())
            )
            if math.isclose(hi, lo):
                lo, hi = lo, lo + 1.0

        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)

        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)
        return out

    def _write_nifti(self, img: sitk.Image, path: Path) -> None:
        sitk.WriteImage(img, str(path))

    # -------------------------
    # Manifest & naming
    # -------------------------

    def _write_manifest(self) -> None:
        if not self.manifest_rows:
            logging.warning("No slices exported; manifest will be empty.")

        manifest_path = self.cfg.out_dir / "manifest.csv"
        fieldnames = [
            "case_id",
            "series_uid",
            "slice_idx",
            "image_path",
            "mask_path",
            "num_labels",
            "spacing_x",
            "spacing_y",
            "spacing_z",
            "modality",
            "study_date",
            "manufacturer",
        ]
        with manifest_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in self.manifest_rows:
                w.writerow(row)

        logging.info(f"Wrote manifest: {manifest_path}")

    def _case_id_from_path(self, series_dir: Path) -> str:
        # Use the nearest NBIA case folder (e.g., Prostate-MRI-US-Biopsy-0144)
        parts = list(series_dir.parts)
        for p in reversed(parts):
            if "Prostate-MRI-US-Biopsy-" in p:
                return p.split("Prostate-MRI-US-Biopsy-")[-1]
        # fallback to patient id if present in DICOM
        return Path(parts[-3]).name if len(parts) >= 3 else "unknown"

    def _safe_id(self, s: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


# -------------------------
# Batch processing functions
# -------------------------

DEFAULT_SEQUENCE_DIRS = {
    "t2": (Path("data/nbia"), Path("data/processed")),
    "ep2d_adc": (Path("data/nbia_ep2d_adc"), Path("data/processed_ep2d_adc")),
    "ep2d_calc": (Path("data/nbia_ep2d_calc"), Path("data/processed_ep2d_calc")),
}


def process_class(
    class_num: int,
    base_dir: Path = Path("data/nbia"),
    output_base: Path = Path("data/processed"),
) -> Dict[str, int]:
    """
    Process all DICOM series for a specific class.

    Args:
        class_num: Class number (1-4)
        base_dir: Base directory containing NBIA downloads
        output_base: Base output directory

    Returns:
        Dict with statistics (num_series, num_slices, num_with_labels)
    """
    class_dir = base_dir / f"class{class_num}"
    output_dir = output_base / f"class{class_num}"

    if not class_dir.exists():
        logging.warning(f"Class directory not found: {class_dir}")
        return {"num_series": 0, "num_slices": 0, "num_with_labels": 0}

    logging.info(f"\n{'='*70}")
    logging.info(f"Processing Class {class_num}")
    logging.info(f"{'='*70}")
    logging.info(f"Input:  {class_dir}")
    logging.info(f"Output: {output_dir}")

    cfg = ConverterConfig(
        root_dir=class_dir,
        out_dir=output_dir,
        image_format="png",
        resample_spacing=None,
        save_nifti=False,
        window=None,
        auto_window=True,
    )

    converter = DicomConverter(cfg)
    converter.convert_all()

    # Calculate statistics
    stats = {
        "num_series": len(converter.manifest_rows)
        // max(1, len(set(r["series_uid"] for r in converter.manifest_rows))),
        "num_slices": len(converter.manifest_rows),
        "num_with_labels": sum(1 for r in converter.manifest_rows if r["mask_path"]),
    }

    return stats


def process_all_classes(
    base_dir: Path = Path("data/nbia"), output_base: Path = Path("data/processed")
) -> None:
    """
    Process all classes (1-4) and generate combined statistics.

    Args:
        base_dir: Base directory containing NBIA downloads
        output_base: Base output directory
    """
    logging.info("\n" + "=" * 70)
    logging.info("DICOM CONVERTER - Batch Processing All Classes")
    logging.info("=" * 70)

    all_stats = {}
    all_manifests = []

    for class_num in range(1, 5):
        stats = process_class(class_num, base_dir, output_base)
        all_stats[f"class{class_num}"] = stats

        # Collect manifest for this class
        manifest_path = output_base / f"class{class_num}" / "manifest.csv"
        if manifest_path.exists():
            import csv

            with manifest_path.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["class"] = str(class_num)
                    all_manifests.append(row)

    # Write combined manifest
    if all_manifests:
        combined_manifest_path = output_base / "manifest_all.csv"
        fieldnames = list(all_manifests[0].keys())
        with combined_manifest_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in all_manifests:
                w.writerow(row)
        logging.info(f"\n✓ Wrote combined manifest: {combined_manifest_path}")

    # Print summary
    logging.info("\n" + "=" * 70)
    logging.info("SUMMARY")
    logging.info("=" * 70)
    for class_name, stats in all_stats.items():
        logging.info(f"{class_name}:")
        logging.info(f"  Series: {stats['num_series']}")
        logging.info(f"  Slices: {stats['num_slices']}")
        logging.info(f"  With labels: {stats['num_with_labels']}")

    total_slices = sum(s["num_slices"] for s in all_stats.values())
    total_with_labels = sum(s["num_with_labels"] for s in all_stats.values())
    logging.info(f"\nTotal slices: {total_slices}")
    logging.info(f"Total with labels: {total_with_labels}")
    if total_slices:
        logging.info(f"Label coverage: {100*total_with_labels/total_slices:.1f}%")
    else:
        logging.info("Label coverage: n/a (no slices)")
    logging.info("=" * 70)


def process_all_sequences(
    sequences: Dict[str, Tuple[Path, Path]] = DEFAULT_SEQUENCE_DIRS,
) -> None:
    logging.info("\n" + "=" * 70)
    logging.info("DICOM CONVERTER - Batch Processing All Sequences")
    logging.info("=" * 70)

    for sequence_name, (input_dir, output_dir) in sequences.items():
        if not input_dir.exists():
            logging.warning(
                f"Sequence '{sequence_name}' input directory not found: {input_dir} (skipping)"
            )
            continue

        logging.info("\n" + "-" * 70)
        logging.info(f"Sequence: {sequence_name}")
        logging.info("-" * 70)
        process_all_classes(input_dir, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert NBIA DICOM data to per-slice images"
    )
    parser.add_argument(
        "--class",
        dest="class_num",
        type=int,
        choices=[1, 2, 3, 4],
        help="Process specific class (1-4)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Process all classes. If --input/--output are not provided, "
            "defaults to all sequences (t2, ep2d_adc, ep2d_calc)."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input base directory (default: data/nbia)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output base directory (default: data/processed)",
    )

    args = parser.parse_args()

    default_input = DEFAULT_SEQUENCE_DIRS["t2"][0]
    default_output = DEFAULT_SEQUENCE_DIRS["t2"][1]
    input_path = Path(args.input) if args.input else default_input
    output_path = Path(args.output) if args.output else default_output

    if args.all:
        if args.input or args.output:
            process_all_classes(input_path, output_path)
        else:
            process_all_sequences()
    elif args.class_num:
        process_class(args.class_num, input_path, output_path)
    else:
        parser.print_help()
