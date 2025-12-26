#!/usr/bin/env python3
"""
Comprehensive Mask Validation Tool

Generates mask overlay visualizations for ALL cases in each class.
Creates individual overlay images per case for detailed inspection.

File structure:
    validation_results/class{n}/case_{i}/overlay_{first}_{last}.png
    validation_results/class{n}/case_{i}/overlay_ep2d_adc_{first}_{last}.png
    validation_results/class{n}/case_{i}/overlay_ep2d_calc_{first}_{last}.png

Usage:
    python tools/validate_all_masks.py
    python tools/validate_all_masks.py --class class1  # Validate specific class
"""

import sys
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import pandas as pd
import json
from typing import Dict, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

SEQUENCE_CONFIGS = {
    "t2": {
        "processed_dir": Path("data/processed"),
        "output_prefix": "overlay",
    },
    "ep2d_adc": {
        "processed_dir": Path("data/processed_ep2d_adc"),
        "output_prefix": "overlay_ep2d_adc",
    },
    "ep2d_calc": {
        "processed_dir": Path("data/processed_ep2d_calc"),
        "output_prefix": "overlay_ep2d_calc",
    },
}


def read_meta(path: Path) -> Dict:
    """Read a meta.json file if present."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def get_mask_slice_count(series_dir: Path) -> int:
    """Return the max slice count across available structures."""
    max_count = 0
    for structure_dir in series_dir.iterdir():
        if not structure_dir.is_dir():
            continue
        mask_count = len(list(structure_dir.glob("*.png")))
        max_count = max(max_count, mask_count)
    return max_count


def build_case_image_series_index(case_processed_dir: Path) -> Dict[str, Dict]:
    """Index image series metadata for a processed case."""
    index: Dict[str, Dict] = {}
    if not case_processed_dir.exists():
        return index

    for series_dir in case_processed_dir.iterdir():
        if not series_dir.is_dir():
            continue
        images_dir = series_dir / "images"
        if not images_dir.exists():
            continue

        meta = read_meta(series_dir / "meta.json")
        if "num_slices" not in meta:
            meta["num_slices"] = len(list(images_dir.glob("*.png")))
        index[series_dir.name] = meta

    return index


def select_image_series_uid(
    seg_series_uid: str,
    seg_series_dir: Path,
    t2_case_dir: Optional[Path],
    image_series_meta_index: Dict[str, Dict],
) -> Tuple[Optional[str], str]:
    """Select image series UID for a sequence using study UID or slice counts."""
    if seg_series_uid in image_series_meta_index:
        return seg_series_uid, "exact"

    seg_meta = {}
    if t2_case_dir is not None:
        seg_meta = read_meta(t2_case_dir / seg_series_uid / "meta.json")

    target_slices = seg_meta.get("num_slices") or get_mask_slice_count(seg_series_dir)
    study_uid = seg_meta.get("StudyInstanceUID")

    if study_uid:
        candidates = {
            uid: meta
            for uid, meta in image_series_meta_index.items()
            if meta.get("StudyInstanceUID") == study_uid
        }
        if candidates:
            best_uid = min(
                candidates.keys(),
                key=lambda uid: abs(int(candidates[uid].get("num_slices", 0)) - int(target_slices)),
            )
            return best_uid, "study_uid"

    if not image_series_meta_index:
        return None, "missing"

    best_uid = min(
        image_series_meta_index.keys(),
        key=lambda uid: abs(int(image_series_meta_index[uid].get("num_slices", 0)) - int(target_slices)),
    )
    return best_uid, "fallback"


def print_banner(text: str):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def validate_case_masks(
    case_dir: Path,
    series_uid: str,
    image_series_uid: str,
    manifest_df: pd.DataFrame,
    output_dir: Path,
    class_name: str,
    output_prefix: str,
    sequence_label: str,
    match_type: str,
):
    """
    Validate and visualize all masks for a single case.
    
    Args:
        case_dir: Path to case directory in processed_seg
        series_uid: Series UID to process
        manifest_df: DataFrame with image paths
        output_dir: Output directory for visualizations
        class_name: Class name (e.g., 'class1')
    """
    series_dir = case_dir / series_uid
    
    if not series_dir.exists():
        return False
    
    # Define structure colors (RGB normalized to [0, 1])
    structure_colors = {
        "prostate": (1.0, 1.0, 0.0),  # Yellow
        "target1": (1.0, 0.0, 0.0),  # Red
        "target2": (0.26, 0.45, 0.77),  # Blue (#4472C4 converted to RGB)
    }
    
    # Find available structures and their mask files
    available_structures = {}
    all_slice_numbers = set()
    
    for structure_name in structure_colors.keys():
        structure_dir = series_dir / structure_name
        if structure_dir.exists():
            mask_files = sorted(list(structure_dir.glob("*.png")))
            if mask_files:
                available_structures[structure_name] = mask_files
                # Collect all slice numbers
                for mask_file in mask_files:
                    all_slice_numbers.add(int(mask_file.stem))
    
    if not available_structures:
        print(f"    ⚠ No masks found in {series_dir}")
        return False
    
    # Sort slice numbers
    slice_numbers = sorted(list(all_slice_numbers))
    
    if not slice_numbers:
        return False
    
    # Extract case_id from case_dir name
    case_id = int(case_dir.name.split("_")[1])

    # Filter manifest for this image series
    manifest_series = manifest_df[
        (manifest_df["case_id"] == case_id)
        & (manifest_df["series_uid"] == image_series_uid)
    ]
    if manifest_series.empty:
        print(
            f"    ⚠ No manifest rows for {sequence_label} series "
            f"{image_series_uid[:30]} (match: {match_type})"
        )
        return False
    
    # Determine grid size based on number of slices
    num_slices = len(slice_numbers)
    
    # Use larger grid for better visibility
    if num_slices <= 6:
        rows, cols = 2, 3
        fig_size = (18, 12)
    elif num_slices <= 12:
        rows, cols = 3, 4
        fig_size = (24, 18)
    elif num_slices <= 20:
        rows, cols = 4, 5
        fig_size = (30, 24)
    else:
        # For very large numbers, use 6 columns
        cols = 6
        rows = (num_slices + cols - 1) // cols
        fig_size = (36, 6 * rows)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    print(
        f"    Processing {num_slices} slices from {series_dir.name[:30]} -> "
        f"{image_series_uid[:30]} ({sequence_label}, {match_type})..."
    )
    
    size_mismatch_info = None

    # Process each slice
    for idx, slice_num in enumerate(slice_numbers):
        if idx >= len(axes):
            break
            
        # Find matching image in manifest
        matching_row = manifest_df[
            (manifest_df["case_id"] == case_id)
            & (manifest_df["series_uid"] == image_series_uid)
            & (manifest_df["slice_idx"] == slice_num)
        ]
        
        if len(matching_row) > 0:
            row = matching_row.iloc[0]
            img_path = Path(row["image_path"])
            
            if img_path.exists():
                # Load and display base image
                img = np.array(Image.open(img_path).convert("L"))
                axes[idx].imshow(img, cmap="gray")
                
                # Overlay each available structure with its color
                structures_present = []
                for structure_name, color in structure_colors.items():
                    if structure_name in available_structures:
                        mask_path = series_dir / structure_name / f"{slice_num:04d}.png"
                        
                        if mask_path.exists():
                            mask = np.array(Image.open(mask_path).convert("L"))
                            if mask.shape != img.shape:
                                if size_mismatch_info is None:
                                    size_mismatch_info = (mask.shape, img.shape)
                                    print(
                                        f"    ⚠ Size mismatch {sequence_label}: "
                                        f"{mask.shape[1]}x{mask.shape[0]} -> "
                                        f"{img.shape[1]}x{img.shape[0]} (resized)"
                                    )
                                mask_img = Image.fromarray(mask)
                                mask_img = mask_img.resize(
                                    (img.shape[1], img.shape[0]), Image.NEAREST
                                )
                                mask = np.array(mask_img)
                            
                            # Create colored mask (RGBA)
                            if mask.max() > 0:
                                mask_rgba = np.zeros((*mask.shape, 4))
                                mask_rgba[..., 0] = color[0]  # R
                                mask_rgba[..., 1] = color[1]  # G
                                mask_rgba[..., 2] = color[2]  # B
                                mask_rgba[..., 3] = 0.5 * (mask / 255.0)  # Alpha
                                
                                axes[idx].imshow(mask_rgba)
                                structures_present.append(structure_name.capitalize())
                
                # Set title with structure info
                title = f"Slice {slice_num}"
                if structures_present:
                    title += f"\n({', '.join(structures_present)})"
                axes[idx].set_title(title, fontsize=10, fontweight='bold')
                axes[idx].axis("off")
            else:
                axes[idx].text(
                    0.5, 0.5, f"Image not found\nSlice {slice_num}",
                    ha="center", va="center",
                    transform=axes[idx].transAxes
                )
                axes[idx].axis("off")
        else:
            axes[idx].text(
                0.5, 0.5, f"No manifest entry\nSlice {slice_num}",
                ha="center", va="center",
                transform=axes[idx].transAxes
            )
            axes[idx].axis("off")
    
    # Hide unused subplots
    for idx in range(len(slice_numbers), len(axes)):
        axes[idx].axis("off")
    
    # Add legend for structure colors
    legend_elements = []
    for structure_name in sorted(available_structures.keys()):
        color = structure_colors[structure_name]
        legend_elements.append(
            Patch(facecolor=color, edgecolor="black", label=structure_name.capitalize())
        )
    
    # Add legend to the figure
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=len(legend_elements),
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, 0.99),
    )
    
    # Add title
    first_slice = slice_numbers[0]
    last_slice = slice_numbers[-1]
    plt.suptitle(
        f"{class_name.upper()} - {sequence_label.upper()} - Case {case_id:04d} - "
        f"Slices {first_slice} to {last_slice}\n"
        f"Available structures: {', '.join([s.capitalize() for s in available_structures.keys()])}",
        fontsize=16,
        fontweight='bold',
        y=0.97
    )
    size_note = ""
    if size_mismatch_info is not None:
        mask_shape, img_shape = size_mismatch_info
        size_note = (
            f" | Size: {mask_shape[1]}x{mask_shape[0]}->"
            f"{img_shape[1]}x{img_shape[0]}"
        )

    fig.text(
        0.98,
        0.02,
        f"Match: {match_type}{size_note}",
        ha="right",
        va="bottom",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = output_dir / f"{output_prefix}_{first_slice}_{last_slice}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"    ✓ Saved: {output_path.name}")
    print(f"      Slices: {first_slice}-{last_slice} ({num_slices} total)")
    print(f"      Structures: {', '.join([s.capitalize() for s in available_structures.keys()])}")
    
    return True


def validate_class_masks(class_name: str):
    """Validate all masks for a specific class."""
    print_banner(f"Validating {class_name.upper()}")
    
    # Load manifests per sequence
    sequence_manifests = {}
    for sequence_name, cfg in SEQUENCE_CONFIGS.items():
        manifest_path = cfg["processed_dir"] / class_name / "manifest.csv"
        if not manifest_path.exists():
            print(f"✗ Manifest not found: {manifest_path} (skipping {sequence_name})")
            continue

        df = pd.read_csv(manifest_path)
        df["case_id"] = pd.to_numeric(df["case_id"], errors="coerce")
        df["slice_idx"] = pd.to_numeric(df["slice_idx"], errors="coerce")
        df = df.dropna(subset=["case_id", "series_uid", "slice_idx"])
        df["case_id"] = df["case_id"].astype(int)
        df["slice_idx"] = df["slice_idx"].astype(int)

        print(f"✓ Loaded manifest ({sequence_name}): {len(df)} rows")
        print(f"  Cases: {df['case_id'].nunique()}")
        print(f"  Series: {df['series_uid'].nunique()}")
        sequence_manifests[sequence_name] = df

    if not sequence_manifests:
        print("✗ No manifests loaded; aborting.")
        return False
    
    # Check processed_seg directory
    processed_seg_dir = Path("data/processed_seg") / class_name
    
    if not processed_seg_dir.exists():
        print(f"✗ No processed_seg directory: {processed_seg_dir}")
        return False
    
    # Find all cases with masks
    case_dirs = sorted(list(processed_seg_dir.glob("case_*")))
    
    if not case_dirs:
        print(f"✗ No cases found in {processed_seg_dir}")
        return False
    
    print(f"✓ Found {len(case_dirs)} cases with masks")
    print()
    
    # Process each case
    success_count = 0
    
    for case_dir in case_dirs:
        case_id = int(case_dir.name.split("_")[1])
        print(f"  Processing Case {case_id:04d}...")
        
        # Create output directory for this case
        output_dir = Path("validation_results") / class_name / f"case_{case_id:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find series directories
        series_dirs = [
            d for d in case_dir.iterdir() 
            if d.is_dir() and d.name != "biopsies.json"
        ]
        
        if not series_dirs:
            print(f"    ⚠ No series directories found")
            continue
        
        t2_case_dir = SEQUENCE_CONFIGS["t2"]["processed_dir"] / class_name / case_dir.name

        # Process each series for each sequence
        series_success = False
        for sequence_name, df in sequence_manifests.items():
            processed_dir = SEQUENCE_CONFIGS[sequence_name]["processed_dir"]
            case_processed_dir = processed_dir / class_name / case_dir.name
            image_series_meta_index = build_case_image_series_index(case_processed_dir)
            if not image_series_meta_index:
                print(f"    ⚠ No image series found for {sequence_name}")
                continue

            for series_dir in series_dirs:
                series_uid = series_dir.name
                image_series_uid, match_type = select_image_series_uid(
                    series_uid,
                    series_dir,
                    t2_case_dir if t2_case_dir.exists() else None,
                    image_series_meta_index,
                )
                if image_series_uid is None:
                    print(
                        f"    ⚠ No image series match for {sequence_name} "
                        f"{series_uid[:30]}"
                    )
                    continue

                if validate_case_masks(
                    case_dir,
                    series_uid,
                    image_series_uid,
                    df,
                    output_dir,
                    class_name,
                    SEQUENCE_CONFIGS[sequence_name]["output_prefix"],
                    sequence_name,
                    match_type,
                ):
                    series_success = True
        
        if series_success:
            success_count += 1
        
        print()
    
    print(f"✓ Successfully processed {success_count}/{len(case_dirs)} cases")
    print(f"  Results saved to: validation_results/{class_name}/")
    
    return True


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate all mask overlays for dataset inspection"
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        type=str,
        help="Validate specific class only (e.g., class1, class2)",
    )
    
    args = parser.parse_args()
    
    print_banner("COMPREHENSIVE MASK VALIDATION")
    
    # Determine which classes to process
    if args.class_name:
        classes = [args.class_name]
    else:
        # Find all available classes
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            print(f"✗ Processed data directory not found: {processed_dir}")
            print("\nPlease run preprocessing first:")
            print("  python service/preprocess.py --all")
            sys.exit(1)
        
        manifests = list(processed_dir.glob("class*/manifest.csv"))
        
        if not manifests:
            print(f"✗ No manifest files found in {processed_dir}")
            sys.exit(1)
        
        classes = [m.parent.name for m in manifests]
    
    print(f"Classes to validate: {', '.join(classes)}")
    print()
    
    # Validate each class
    success_classes = []
    
    for class_name in classes:
        if validate_class_masks(class_name):
            success_classes.append(class_name)
    
    # Final summary
    print_banner("VALIDATION SUMMARY")
    
    print(f"✓ Validation complete!")
    print(f"\nSuccessfully validated: {len(success_classes)}/{len(classes)} classes")
    for class_name in success_classes:
        output_dir = Path("validation_results") / class_name
        case_dirs = list(output_dir.glob("case_*"))
        print(f"  • {class_name}: {len(case_dirs)} cases → validation_results/{class_name}/")
    
    print("\nGenerated files:")
    print("  • validation_results/class{n}/case_{i}/overlay_{first}_{last}.png")
    print("  • validation_results/class{n}/case_{i}/overlay_ep2d_adc_{first}_{last}.png")
    print("  • validation_results/class{n}/case_{i}/overlay_ep2d_calc_{first}_{last}.png")
    print("    where {first} and {last} are the first and last slice numbers")
    
    print("\nNext steps:")
    print("  1. Review each case's overlay images to verify mask quality")
    print("  2. Check for any alignment issues between images and masks")
    print("  3. Identify any cases that need mask corrections")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
