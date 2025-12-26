#!/usr/bin/env python3
"""
Visualize segmentation masks overlaid on MRI images.

This script creates visualization images showing:
- Original MRI image
- Segmentation mask overlay (colored)
- Side-by-side comparison
- Multi-mask overlay (prostate + targets)

Requirements:
    pip install pandas pyarrow numpy pillow matplotlib tqdm
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json


# Color scheme for different structures
STRUCTURE_COLORS = {
    "prostate": (255, 255, 0, 100),      # Yellow, semi-transparent
    "target1": (255, 0, 0, 150),         # Red, more opaque
    "target2": (255, 128, 0, 150),       # Orange
    "target3": (255, 0, 255, 150),       # Magenta
    "default": (0, 255, 0, 100)          # Green for unknown structures
}

SEQUENCE_PROCESSED_DIRS = {
    "t2": Path("data/processed"),
    "ep2d_adc": Path("data/processed_ep2d_adc"),
    "ep2d_calc": Path("data/processed_ep2d_calc"),
}


def find_matching_cases(
    processed_dir: Path,
    processed_seg_dir: Path,
    sequence_name: Optional[str] = None,
) -> List[Tuple[Path, Path]]:
    """
    Find matching case directories between processed and processed_seg.
    
    Args:
        processed_dir: Path to processed/ directory
        processed_seg_dir: Path to processed_seg/ directory
        sequence_name: Optional sequence label for logging
    
    Returns:
        List of tuples (processed_case_path, processed_seg_case_path)
    """
    matches = []
    
    print(f"\n{'='*80}")
    if sequence_name:
        print(f"Finding matching cases ({sequence_name})...")
    else:
        print("Finding matching cases...")
    print(f"{'='*80}")
    
    # Iterate through all class directories
    for class_dir in sorted(processed_dir.glob("class*")):
        class_name = class_dir.name
        seg_class_dir = processed_seg_dir / class_name
        
        if not seg_class_dir.exists():
            continue
        
        # Iterate through all cases in this class
        for case_dir in sorted(class_dir.glob("case_*")):
            case_name = case_dir.name
            seg_case_dir = seg_class_dir / case_name
            
            if seg_case_dir.exists():
                matches.append((case_dir, seg_case_dir))
    
    print(f"✓ Found {len(matches)} matching cases\n")
    return matches


def build_seg_series_index(case_seg: Path) -> Dict[str, Dict]:
    """Index segmentation series dirs with structure and mask counts."""
    seg_series_info: Dict[str, Dict] = {}
    for seg_series_dir in sorted(case_seg.iterdir()):
        if not seg_series_dir.is_dir():
            continue
        structure_dirs = [d for d in seg_series_dir.iterdir() if d.is_dir()]
        if not structure_dirs:
            continue
        mask_counts = [len(list(sd.glob("*.png"))) for sd in structure_dirs]
        max_mask_count = max(mask_counts) if mask_counts else 0
        if max_mask_count == 0:
            continue
        seg_series_info[seg_series_dir.name] = {
            "dir": seg_series_dir,
            "structure_dirs": structure_dirs,
            "mask_count": max_mask_count,
        }
    return seg_series_info


def read_meta(series_dir: Path) -> Dict:
    meta_path = series_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def build_seg_series_meta_index(
    seg_series_info: Dict[str, Dict], t2_case_dir: Optional[Path]
) -> Dict[str, Dict]:
    """Load T2 meta.json for segmentation series UIDs."""
    if t2_case_dir is None or not t2_case_dir.exists():
        return {}

    meta_index = {}
    for series_uid in seg_series_info.keys():
        meta = read_meta(t2_case_dir / series_uid)
        if meta:
            meta_index[series_uid] = meta
    return meta_index


def select_seg_series_entry(
    series_uid: str,
    seg_series_info: Dict[str, Dict],
    image_count: int,
    series_meta: Optional[Dict] = None,
    seg_series_meta: Optional[Dict[str, Dict]] = None,
) -> Tuple[Optional[Dict], str]:
    if series_uid in seg_series_info:
        return seg_series_info[series_uid], "exact"
    if not seg_series_info:
        return None, "missing"

    study_uid = series_meta.get("StudyInstanceUID") if series_meta else None
    if study_uid and seg_series_meta:
        candidates = [
            uid
            for uid, meta in seg_series_meta.items()
            if meta.get("StudyInstanceUID") == study_uid
        ]
        if candidates:
            target_slices = series_meta.get("num_slices") or image_count

            def score(uid: str) -> Tuple[int, int, str]:
                meta = seg_series_meta.get(uid, {})
                seg_slices = meta.get("num_slices")
                if seg_slices is None:
                    seg_slices = seg_series_info[uid]["mask_count"]
                diff = abs(int(seg_slices) - int(target_slices))
                return (diff, -int(seg_slices), uid)

            best_uid = min(candidates, key=score)
            return seg_series_info[best_uid], "study_uid"

    def score(name: str) -> Tuple[int, int, str]:
        mask_count = seg_series_info[name]["mask_count"]
        diff = abs(mask_count - image_count)
        return (diff, -mask_count, name)

    best_name = min(seg_series_info.keys(), key=score)
    return seg_series_info[best_name], "fallback"


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load an image as numpy array.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image as numpy array (H, W) or None if failed
    """
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        return None


def create_overlay(image: np.ndarray, 
                   mask: np.ndarray, 
                   color: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Create an overlay of mask on image.
    
    Args:
        image: Original image (H, W) grayscale
        mask: Binary mask (H, W)
        color: RGBA color tuple
    
    Returns:
        RGB image with overlay (H, W, 3)
    """
    # Convert grayscale to RGB
    if len(image.shape) == 2:
        img_rgb = np.stack([image, image, image], axis=-1)
    else:
        img_rgb = image.copy()
    
    # Normalize if needed
    if img_rgb.max() > 1:
        img_rgb = img_rgb.astype(np.float32) / 255.0
    
    # Create colored mask
    r, g, b, a = color
    alpha = a / 255.0
    
    # Apply mask color with alpha blending
    mask_bool = mask > 0
    if mask_bool.any():
        img_rgb[mask_bool, 0] = (1 - alpha) * img_rgb[mask_bool, 0] + alpha * (r / 255.0)
        img_rgb[mask_bool, 1] = (1 - alpha) * img_rgb[mask_bool, 1] + alpha * (g / 255.0)
        img_rgb[mask_bool, 2] = (1 - alpha) * img_rgb[mask_bool, 2] + alpha * (b / 255.0)
    
    return (img_rgb * 255).astype(np.uint8)


def visualize_case(case_processed: Path, 
                   case_seg: Path,
                   output_dir: Path,
                   max_slices: int = 10,
                   sequence_name: Optional[str] = None,
                   t2_processed_dir: Optional[Path] = None) -> Dict:
    """
    Create visualizations for a single case.
    
    Args:
        case_processed: Path to case in processed/
        case_seg: Path to case in processed_seg/
        output_dir: Output directory for visualizations
        max_slices: Maximum number of slices to visualize per series
    
    Returns:
        Statistics dict
    """
    stats = {
        "visualizations_created": 0,
        "series_processed": 0,
        "structures_found": 0,
        "series_fallback_used": 0,
        "series_missing_seg": 0,
        "series_fallback_mismatch": 0,
        "series_matched_study": 0,
    }
    
    case_name = case_processed.name
    class_name = case_processed.parent.name
    
    # Create output directory
    case_output_dir = output_dir / class_name / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build segmentation series index once per case
    seg_series_info = build_seg_series_index(case_seg)
    t2_case_dir = None
    if t2_processed_dir is not None:
        t2_case_dir = t2_processed_dir / class_name / case_name
    seg_series_meta = build_seg_series_meta_index(seg_series_info, t2_case_dir)

    # Find matching series directories
    for series_dir in sorted(case_processed.glob("*")):
        if not series_dir.is_dir():
            continue
        
        series_uid = series_dir.name
        
        # Get image directory
        images_dir = series_dir / "images"
        if not images_dir.exists():
            continue

        # Get list of image files
        image_files = sorted(images_dir.glob("*.png"))
        if not image_files:
            continue

        series_meta = read_meta(series_dir)
        seg_entry, match_type = select_seg_series_entry(
            series_uid,
            seg_series_info,
            len(image_files),
            series_meta=series_meta,
            seg_series_meta=seg_series_meta,
        )
        if seg_entry is None:
            stats["series_missing_seg"] += 1
            continue

        if match_type == "study_uid":
            stats["series_matched_study"] += 1
        elif match_type == "fallback":
            stats["series_fallback_used"] += 1
        diff = abs(seg_entry["mask_count"] - len(image_files))
        if diff > max(5, int(len(image_files) * 0.2)):
            stats["series_fallback_mismatch"] += 1
            seq_label = sequence_name or "sequence"
            print(
                f"⚠️  {seq_label} {class_name}/{case_name} {series_uid}: "
                f"mask count {seg_entry['mask_count']} vs images {len(image_files)}"
            )

        stats["series_processed"] += 1
        
        # Find all structure directories (prostate, target1, etc.)
        structure_dirs = seg_entry["structure_dirs"]
        stats["structures_found"] += len(structure_dirs)
        
        if not structure_dirs:
            continue
        
        # Sample slices evenly across the volume
        if len(image_files) > max_slices:
            indices = np.linspace(0, len(image_files)-1, max_slices, dtype=int)
            image_files = [image_files[i] for i in indices]
        
        # Process each slice
        for img_file in image_files:
            slice_num = img_file.stem
            
            # Load original image
            image = load_image(img_file)
            if image is None:
                continue
            
            # Create overlay with all available masks
            overlay_img = None
            structure_names = []
            
            for struct_dir in structure_dirs:
                struct_name = struct_dir.name
                mask_file = struct_dir / f"{slice_num}.png"
                
                if not mask_file.exists():
                    continue
                
                # Load mask
                mask = load_image(mask_file)
                if mask is None:
                    continue
                
                # Resize mask to match image if needed
                if mask.shape != image.shape:
                    mask_img = Image.fromarray(mask)
                    mask_img = mask_img.resize((image.shape[1], image.shape[0]), Image.NEAREST)
                    mask = np.array(mask_img)
                
                # Get color for this structure
                color = STRUCTURE_COLORS.get(struct_name, STRUCTURE_COLORS["default"])
                
                # Create overlay
                if overlay_img is None:
                    overlay_img = create_overlay(image, mask, color)
                else:
                    overlay_img = create_overlay(overlay_img, mask, color)
                
                structure_names.append(struct_name)
            
            if overlay_img is None:
                continue
            
            # Create visualization figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original MRI')
            axes[0].axis('off')
            
            # Overlay
            axes[1].imshow(overlay_img)
            axes[1].set_title(f'Overlay ({", ".join(structure_names)})')
            axes[1].axis('off')
            
            # Masks only
            mask_composite = np.zeros((*image.shape, 3), dtype=np.uint8)
            for struct_dir in structure_dirs:
                struct_name = struct_dir.name
                mask_file = struct_dir / f"{slice_num}.png"
                
                if mask_file.exists():
                    mask = load_image(mask_file)
                    if mask is not None:
                        if mask.shape != image.shape:
                            mask_img = Image.fromarray(mask)
                            mask_img = mask_img.resize((image.shape[1], image.shape[0]), Image.NEAREST)
                            mask = np.array(mask_img)
                        
                        color = STRUCTURE_COLORS.get(struct_name, STRUCTURE_COLORS["default"])
                        mask_bool = mask > 0
                        mask_composite[mask_bool] = color[:3]
            
            axes[2].imshow(mask_composite)
            axes[2].set_title('Masks Only')
            axes[2].axis('off')
            
            # Add legend
            legend_elements = []
            for struct_name in set(structure_names):
                color = STRUCTURE_COLORS.get(struct_name, STRUCTURE_COLORS["default"])
                legend_elements.append(patches.Patch(
                    facecolor=np.array(color[:3])/255.0, 
                    label=struct_name.capitalize()
                ))
            
            if legend_elements:
                fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements))
            
            plt.suptitle(f'{class_name}/{case_name} - Slice {slice_num}', fontsize=14)
            plt.tight_layout()
            
            # Save figure
            output_file = case_output_dir / f"slice_{slice_num}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            stats["visualizations_created"] += 1
    
    return stats


def main():
    """Main execution function."""
    
    print("="*80)
    print("Segmentation Mask Visualization")
    print("="*80)
    
    # Configuration
    processed_seg_dir = Path("data/processed_seg")
    output_base_dir = Path("data/visualizations")
    max_slices_per_series = 10  # Visualize up to 10 slices per series
    
    total_stats = {
        "visualizations_created": 0,
        "series_processed": 0,
        "structures_found": 0,
        "series_fallback_used": 0,
        "series_missing_seg": 0,
        "series_fallback_mismatch": 0,
        "series_matched_study": 0,
    }
    total_cases = 0
    sequences_processed = 0

    # Process each sequence (t2, ep2d_adc, ep2d_calc)
    for sequence_name, processed_dir in SEQUENCE_PROCESSED_DIRS.items():
        if not processed_dir.exists():
            print(f"⚠️  Skipping {sequence_name}: {processed_dir} not found")
            continue

        sequence_seg_dir = (
            processed_seg_dir.parent / f"{processed_seg_dir.name}_{sequence_name}"
        )
        if not sequence_seg_dir.exists():
            sequence_seg_dir = processed_seg_dir

        if not sequence_seg_dir.exists():
            print(f"⚠️  Skipping {sequence_name}: {sequence_seg_dir} not found")
            continue

        output_dir = output_base_dir if sequence_name == "t2" else output_base_dir / sequence_name

        # Find matching cases
        matching_cases = find_matching_cases(processed_dir, sequence_seg_dir, sequence_name)

        if not matching_cases:
            print(f"⚠️  No matching cases found for {sequence_name}")
            continue

        sequences_processed += 1
        total_cases += len(matching_cases)

        # Process each case with progress bar
        print(f"{'='*80}")
        print(f"Creating visualizations for {sequence_name}...")
        print(f"{'='*80}\n")

        pbar = tqdm(matching_cases, desc="Processing cases", unit="case")

        for case_processed, case_seg in pbar:
            case_name = case_processed.name
            class_name = case_processed.parent.name

            pbar.set_description(f"{class_name}/{case_name}")

            stats = visualize_case(
                case_processed,
                case_seg,
                output_dir,
                max_slices=max_slices_per_series,
                sequence_name=sequence_name,
                t2_processed_dir=SEQUENCE_PROCESSED_DIRS["t2"]
            )

            # Accumulate stats
            for key, value in stats.items():
                total_stats[key] += value

            pbar.set_postfix({
                'Viz': total_stats['visualizations_created'],
                'Series': total_stats['series_processed']
            })

        pbar.close()

        print(f"\nOutput directory ({sequence_name}): {output_dir}/")

    if sequences_processed == 0:
        print("❌ Error: No sequences processed (missing inputs or no matching cases).")
        return

    # Final summary
    print(f"\n{'='*80}")
    print(f"Visualization Complete!")
    print(f"{'='*80}")
    print(f"\nSummary Statistics:")
    print(f"  Sequences processed: {sequences_processed}")
    print(f"  Cases processed: {total_cases}")
    print(f"  Series processed: {total_stats['series_processed']}")
    print(f"  Structures found: {total_stats['structures_found']}")
    print(f"  Visualizations created: {total_stats['visualizations_created']}")
    print(f"  Series matched by study UID: {total_stats['series_matched_study']}")
    print(f"  Series fallback used: {total_stats['series_fallback_used']}")
    print(f"  Series missing masks: {total_stats['series_missing_seg']}")
    print(f"  Fallback mismatch (slice count): {total_stats['series_fallback_mismatch']}")
    print(f"\nOutput base directory: {output_base_dir}/")
    print(f"  Structure (t2): {output_base_dir}/class{{N}}/case_{{XXXX}}/slice_{{NNNN}}.png")
    print(f"  Structure (others): {output_base_dir}/{{sequence}}/class{{N}}/case_{{XXXX}}/slice_{{NNNN}}.png")


if __name__ == "__main__":
    main()
