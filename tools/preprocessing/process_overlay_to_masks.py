#!/usr/bin/env python3
"""
Process biopsy overlay data to create PNG masks for training.

This script:
1. Matches overlay directories with MRI Series Instance UIDs from parquet files
2. Converts STL mesh segmentations to PNG masks (slice-by-slice)
3. Copies/organizes only matching data
4. Optionally creates biopsy point annotations

Requirements:
    pip install pandas pyarrow trimesh numpy pillow SimpleITK pydicom tqdm
"""

import pandas as pd
import re
import shutil
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from PIL import Image
import trimesh
from tqdm import tqdm


def load_series_uids_from_parquet(parquet_dir: str) -> Dict[str, Dict]:
    """
    Load Series Instance UIDs and their corresponding classes/patients from parquet files.
    
    Args:
        parquet_dir: Directory containing class-organized parquet files
    
    Returns:
        Dict mapping Series UID to {class, patient_number}
    """
    series_info_map = {}
    parquet_path = Path(parquet_dir)
    
    print(f"\n{'='*80}")
    print(f"Loading Series Instance UIDs from {parquet_dir}...")
    print(f"{'='*80}")
    
    for class_dir in sorted(parquet_path.glob("class=*")):
        class_num = int(class_dir.name.split("=")[1])
        
        for parquet_file in class_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            
            # Look for Series Instance UID column
            series_col = None
            for col_name in ["Series Instance UID (MRI)", "Series Instance UID", "SeriesInstanceUID"]:
                if col_name in df.columns:
                    series_col = col_name
                    break
            
            # Look for patient number column
            patient_col = None
            for col_name in ["patient_number", "Patient ID", "PatientID"]:
                if col_name in df.columns:
                    patient_col = col_name
                    break
            
            if series_col:
                for _, row in df.iterrows():
                    series_uid = str(row[series_col])
                    
                    # Extract patient number (should be 4 digits like "0114")
                    if patient_col:
                        patient_val = str(row[patient_col])
                        # If it's "Prostate-MRI-US-Biopsy-0114", extract just "0114"
                        if "Prostate-MRI-US-Biopsy-" in patient_val:
                            patient_num = patient_val.split("Prostate-MRI-US-Biopsy-")[1].split("-")[0]
                        else:
                            # Just pad with zeros if it's already a number
                            patient_num = patient_val.zfill(4)
                    else:
                        patient_num = "0000"
                    
                    series_info_map[series_uid] = {
                        "class": class_num,
                        "patient_number": patient_num
                    }
                
                print(f"  Class {class_num}: Added {len(df)} series from {parquet_file.name}")
            else:
                print(f"  Warning: No Series UID column in {parquet_file.name}")
                print(f"    Available columns: {list(df.columns)}")
    
    print(f"\n✓ Total unique series UIDs: {len(series_info_map)}")
    return series_info_map


def extract_series_uid_from_dirname(dirname: str) -> Optional[str]:
    """
    Extract Series Instance UID from overlay directory name.
    
    Example:
        'Prostate-MRI-US-Biopsy-0001-BXmr-seriesUID-1.3.6.1.4.1.14519...'
        Returns: '1.3.6.1.4.1.14519...'
    
    Args:
        dirname: Overlay directory name
    
    Returns:
        Series Instance UID or None
    """
    # Pattern: seriesUID-{UID}
    match = re.search(r'seriesUID-([0-9.]+)', dirname)
    if match:
        return match.group(1)
    return None


def match_overlay_dirs_with_series(overlay_base_dir: str, 
                                   series_info_map: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Match overlay directories with Series Instance UIDs.
    
    Args:
        overlay_base_dir: Base directory containing overlay data
        series_info_map: Mapping of Series UID to {class, patient_number}
    
    Returns:
        Dict mapping overlay directory to metadata (series_uid, class, patient_number)
    """
    overlay_path = Path(overlay_base_dir)
    matched_dirs = {}
    unmatched_count = 0
    
    print(f"\n{'='*80}")
    print(f"Matching overlay directories with Series UIDs...")
    print(f"{'='*80}")
    
    for overlay_dir in overlay_path.iterdir():
        if not overlay_dir.is_dir():
            continue
        
        series_uid = extract_series_uid_from_dirname(overlay_dir.name)
        
        if series_uid and series_uid in series_info_map:
            info = series_info_map[series_uid]
            
            matched_dirs[str(overlay_dir)] = {
                "series_uid": series_uid,
                "class": info["class"],
                "patient_number": info["patient_number"],
                "dir_name": overlay_dir.name
            }
        else:
            unmatched_count += 1
    
    print(f"\n  ✓ Matched directories: {len(matched_dirs)}")
    print(f"  ✗ Unmatched directories: {unmatched_count}")
    
    # Show distribution by class
    class_counts = defaultdict(int)
    for info in matched_dirs.values():
        class_counts[info["class"]] += 1
    
    print(f"\n  Distribution by class:")
    for class_num in sorted(class_counts.keys()):
        print(f"    Class {class_num}: {class_counts[class_num]} cases")
    
    return matched_dirs


def load_stl_mesh(stl_path: Path) -> Optional[trimesh.Trimesh]:
    """
    Load an STL file as a trimesh object.
    
    Args:
        stl_path: Path to STL file
    
    Returns:
        Trimesh object or None if failed
    """
    try:
        mesh = trimesh.load(stl_path)
        return mesh
    except Exception as e:
        print(f"    Warning: Failed to load {stl_path.name}: {e}")
        return None


def mesh_to_voxel_grid(mesh: trimesh.Trimesh, 
                       voxel_size: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a mesh to a voxel grid.
    
    Args:
        mesh: Trimesh object
        voxel_size: Size of each voxel in mm
    
    Returns:
        Tuple of (voxel_grid, origin_point)
    """
    # Create voxel grid from mesh
    voxelized = mesh.voxelized(pitch=voxel_size)
    voxel_grid = voxelized.matrix
    origin = voxelized.transform[:3, 3]
    
    return voxel_grid, origin


def save_voxel_slices_as_png(voxel_grid: np.ndarray, 
                             output_dir: Path,
                             prefix: str = "") -> int:
    """
    Save voxel grid slices as PNG images.
    
    Args:
        voxel_grid: 3D binary numpy array
        output_dir: Output directory for PNG files
        prefix: Filename prefix (optional)
    
    Returns:
        Number of slices saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_slices = voxel_grid.shape[2]  # Assuming z-axis is last dimension
    num_saved = 0
    
    for i in range(num_slices):
        slice_data = voxel_grid[:, :, i]
        
        # Skip empty slices
        if not slice_data.any():
            continue
        
        # Convert to 8-bit image (0 or 255)
        img_data = (slice_data * 255).astype(np.uint8)
        
        # Save as PNG
        img = Image.fromarray(img_data)
        
        if prefix:
            img_path = output_dir / f"{prefix}_{i:04d}.png"
        else:
            img_path = output_dir / f"{i:04d}.png"
        
        img.save(img_path)
        num_saved += 1
    
    return num_saved


def parse_fcsv_biopsy_coords(fcsv_path: Path) -> Optional[Dict]:
    """
    Parse a .fcsv file to extract biopsy coordinates.
    
    Args:
        fcsv_path: Path to FCSV file
    
    Returns:
        Dict with biopsy info or None
    """
    try:
        # Read FCSV file
        with open(fcsv_path, 'r') as f:
            lines = f.readlines()
        
        # Find data lines (skip comments)
        data_lines = [line.strip() for line in lines if not line.startswith('#')]
        
        if len(data_lines) < 2:
            return None
        
        coords = []
        labels = []
        for line in data_lines:
            if line:
                parts = line.split(',')
                if len(parts) >= 12:
                    # Extract x, y, z coordinates and label
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    label = parts[11]
                    coords.append([x, y, z])
                    labels.append(label)
        
        if len(coords) >= 2:
            # Extract pathology from filename
            pathology = fcsv_path.stem.replace("Bx-", "").replace(str(fcsv_path.stem.split('-')[0]), "")
            
            return {
                "top": coords[0],  # Entry point
                "bottom": coords[1],  # Exit point
                "pathology": pathology,
                "filename": fcsv_path.name
            }
    except Exception as e:
        print(f"    Warning: Failed to parse {fcsv_path.name}: {e}")
    
    return None


def process_overlay_directory(overlay_dir: Path, 
                              output_base_dir: Path,
                              class_num: int,
                              patient_number: str,
                              series_uid: str,
                              voxel_size: float = 0.5) -> Dict:
    """
    Process a single overlay directory to create masks.
    
    Args:
        overlay_dir: Path to overlay directory
        output_base_dir: Base output directory
        class_num: Class number
        patient_number: Patient number (e.g., "0001")
        series_uid: Series Instance UID
        voxel_size: Voxel size for mesh rasterization
    
    Returns:
        Dict with processing statistics
    """
    stats = {
        "stl_files_found": 0,
        "stl_files_processed": 0,
        "fcsv_files_found": 0,
        "fcsv_files_processed": 0,
        "slices_created": 0
    }
    
    # Create output directory structure matching processed/
    # processed_seg/class{n}/case_{patient_number}/{series_uid}/prostate/, /target1/, etc.
    case_dir = output_base_dir / f"class{class_num}" / f"case_{patient_number}"
    series_dir = case_dir / series_uid
    
    # Look for Data subdirectory
    data_dir = overlay_dir / "Data"
    if not data_dir.exists():
        return stats
    
    # Process STL files (segmentation meshes)
    stl_files = list(data_dir.glob("*.STL")) + list(data_dir.glob("*.stl"))
    stats["stl_files_found"] = len(stl_files)
    
    for stl_file in stl_files:
        mesh = load_stl_mesh(stl_file)
        if mesh is None:
            continue
        
        # Convert mesh to voxel grid
        try:
            voxel_grid, origin = mesh_to_voxel_grid(mesh, voxel_size=voxel_size)
            
            # Save slices as PNG in subdirectory named after the structure
            # e.g., prostate/, target1/, target2/
            mask_name = stl_file.stem.lower().replace(" ", "_")
            mask_output_dir = series_dir / mask_name
            
            num_slices = save_voxel_slices_as_png(voxel_grid, mask_output_dir, prefix="")
            
            stats["slices_created"] += num_slices
            stats["stl_files_processed"] += 1
            
        except Exception as e:
            pass  # Silent fail, will show in progress bar
    
    # Process FCSV files (biopsy coordinates)
    fcsv_files = list(data_dir.glob("*.fcsv"))
    stats["fcsv_files_found"] = len(fcsv_files)
    
    biopsies_data = []
    for fcsv_file in fcsv_files:
        biopsy_info = parse_fcsv_biopsy_coords(fcsv_file)
        if biopsy_info:
            biopsies_data.append(biopsy_info)
            stats["fcsv_files_processed"] += 1
    
    # Save biopsy coordinates as JSON at case level (not series level)
    if biopsies_data:
        case_dir.mkdir(parents=True, exist_ok=True)
        biopsies_json = case_dir / "biopsies.json"
        
        # If file exists, merge with existing biopsies
        existing_biopsies = []
        if biopsies_json.exists():
            with open(biopsies_json, 'r') as f:
                existing_biopsies = json.load(f)
        
        # Add series UID to each biopsy entry
        for biopsy in biopsies_data:
            biopsy['series_uid'] = series_uid
        
        # Merge and save
        all_biopsies = existing_biopsies + biopsies_data
        with open(biopsies_json, 'w') as f:
            json.dump(all_biopsies, f, indent=2)
    
    # Copy MRML file for reference
    mrml_files = list(overlay_dir.glob("*.mrml"))
    if mrml_files:
        series_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mrml_files[0], series_dir / "scene.mrml")
    
    return stats


def main():
    """Main execution function."""
    
    print("="*80)
    print("Overlay Data Processing: Match and Convert to PNG Masks")
    print("="*80)
    
    # Configuration
    parquet_dir = "data/splitted_images"
    overlay_base_dir = "data/overlay/Biopsy Overlays (3D Slicer)"
    output_base_dir = "data/processed_seg"  # Match processed structure
    voxel_size = 0.5  # mm per voxel
    
    # Step 1: Load Series Instance UIDs from parquet files
    series_info_map = load_series_uids_from_parquet(parquet_dir)
    
    if not series_info_map:
        print("❌ Error: No Series UIDs found in parquet files!")
        return
    
    # Step 2: Match overlay directories with Series UIDs
    matched_dirs = match_overlay_dirs_with_series(overlay_base_dir, series_info_map)
    
    if not matched_dirs:
        print("❌ Error: No matching overlay directories found!")
        return
    
    # Step 3: Process each matched directory with progress bar
    print(f"\n{'='*80}")
    print(f"Processing matched overlay directories...")
    print(f"{'='*80}\n")
    
    total_stats = defaultdict(int)
    
    # Create progress bar
    pbar = tqdm(matched_dirs.items(), 
                total=len(matched_dirs),
                desc="Processing cases",
                unit="case")
    
    for overlay_dir_str, info in pbar:
        overlay_dir = Path(overlay_dir_str)
        
        # Update progress bar description
        pbar.set_description(f"Class {info['class']}, Patient {info['patient_number']}")
        
        stats = process_overlay_directory(
            overlay_dir,
            Path(output_base_dir),
            info['class'],
            info['patient_number'],
            info['series_uid'],
            voxel_size=voxel_size
        )
        
        # Accumulate stats
        for key, value in stats.items():
            total_stats[key] += value
        
        # Update progress bar postfix with current stats
        pbar.set_postfix({
            'STL': f"{total_stats['stl_files_processed']}/{total_stats['stl_files_found']}",
            'Slices': total_stats['slices_created']
        })
    
    pbar.close()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"{'='*80}")
    print(f"\nSummary Statistics:")
    print(f"  Total matched cases: {len(matched_dirs)}")
    print(f"  STL files found: {total_stats['stl_files_found']}")
    print(f"  STL files processed: {total_stats['stl_files_processed']}")
    print(f"  FCSV files found: {total_stats['fcsv_files_found']}")
    print(f"  FCSV files processed: {total_stats['fcsv_files_processed']}")
    print(f"  Total mask slices created: {total_stats['slices_created']}")
    print(f"\nOutput directory: {output_base_dir}/")
    print(f"  Structure: class{{N}}/case_{{XXXX}}/{{series_uid}}/{{structure}}/")
    print(f"  Example: class1/case_0001/{info['series_uid'][:40]}.../ prostate/")


if __name__ == "__main__":
    main()

