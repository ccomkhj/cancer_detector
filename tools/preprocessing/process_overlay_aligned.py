#!/usr/bin/env python3
"""
Process biopsy overlay data with proper DICOM-based spatial alignment.

This script properly aligns STL mesh segmentations with DICOM images by:
1. Reading DICOM metadata (spacing, origin, orientation)
2. Transforming mesh coordinates to image space
3. Rasterizing at exact image resolution

Requirements:
    pip install pandas pyarrow trimesh numpy pillow SimpleITK pydicom tqdm scipy
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
import pydicom
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes


def load_series_uids_from_parquet(parquet_dir: str) -> Dict[str, Dict]:
    """Load Series Instance UIDs and their corresponding classes/patients."""
    series_info_map = {}
    parquet_path = Path(parquet_dir)
    
    print(f"\n{'='*80}")
    print(f"Loading Series Instance UIDs from {parquet_dir}...")
    print(f"{'='*80}")
    
    for class_dir in sorted(parquet_path.glob("class=*")):
        class_num = int(class_dir.name.split("=")[1])
        
        for parquet_file in class_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            
            series_col = None
            for col_name in ["Series Instance UID (MRI)", "Series Instance UID", "SeriesInstanceUID"]:
                if col_name in df.columns:
                    series_col = col_name
                    break
            
            patient_col = None
            for col_name in ["patient_number", "Patient ID", "PatientID"]:
                if col_name in df.columns:
                    patient_col = col_name
                    break
            
            if series_col:
                for _, row in df.iterrows():
                    series_uid = str(row[series_col])
                    
                    if patient_col:
                        patient_val = str(row[patient_col])
                        if "Prostate-MRI-US-Biopsy-" in patient_val:
                            patient_num = patient_val.split("Prostate-MRI-US-Biopsy-")[1].split("-")[0]
                        else:
                            patient_num = patient_val.zfill(4)
                    else:
                        patient_num = "0000"
                    
                    series_info_map[series_uid] = {
                        "class": class_num,
                        "patient_number": patient_num
                    }
                
                print(f"  Class {class_num}: Added {len(df)} series from {parquet_file.name}")
    
    print(f"\n✓ Total unique series UIDs: {len(series_info_map)}")
    return series_info_map


def extract_series_uid_from_dirname(dirname: str) -> Optional[str]:
    """Extract Series Instance UID from overlay directory name."""
    match = re.search(r'seriesUID-([0-9.]+)', dirname)
    if match:
        return match.group(1)
    return None


def match_overlay_dirs_with_series(overlay_base_dir: str, 
                                   series_info_map: Dict[str, Dict]) -> Dict[str, Dict]:
    """Match overlay directories with Series Instance UIDs."""
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
    
    class_counts = defaultdict(int)
    for info in matched_dirs.values():
        class_counts[info["class"]] += 1
    
    print(f"\n  Distribution by class:")
    for class_num in sorted(class_counts.keys()):
        print(f"    Class {class_num}: {class_counts[class_num]} cases")
    
    return matched_dirs


def find_dicom_files(processed_dir: Path, class_num: int, patient_num: str, series_uid: str) -> Optional[Path]:
    """Find the DICOM directory for a given series."""
    # Check in processed directory structure
    nbia_base = Path("data/nbia")
    class_dir = nbia_base / f"class{class_num}"
    
    # Search for DICOM files with matching series UID
    if class_dir.exists():
        for dicom_dir in class_dir.rglob("*"):
            if dicom_dir.is_dir():
                # Check if any DICOM file in this directory has matching Series UID
                for dcm_file in dicom_dir.glob("*.dcm"):
                    try:
                        dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
                        if hasattr(dcm, 'SeriesInstanceUID') and dcm.SeriesInstanceUID == series_uid:
                            return dicom_dir
                    except:
                        continue
    
    return None


def load_dicom_geometry(dicom_dir: Path) -> Optional[Dict]:
    """
    Load DICOM geometry information from a series directory.
    
    Returns:
        Dict with spacing, origin, direction, dimensions
    """
    try:
        # Read all DICOM files and sort by instance number
        dcm_files = sorted(dicom_dir.glob("*.dcm"))
        if not dcm_files:
            return None
        
        # Read first file for metadata
        dcm = pydicom.dcmread(dcm_files[0])
        
        # Get image geometry
        spacing = np.array([
            float(dcm.PixelSpacing[0]),
            float(dcm.PixelSpacing[1]),
            float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else 1.0
        ])
        
        origin = np.array(dcm.ImagePositionPatient, dtype=float)
        
        # Get orientation matrix
        orientation = np.array(dcm.ImageOrientationPatient, dtype=float).reshape(2, 3)
        
        # Build direction matrix
        direction = np.eye(3)
        direction[:, 0] = orientation[0]  # X direction
        direction[:, 1] = orientation[1]  # Y direction
        direction[:, 2] = np.cross(orientation[0], orientation[1])  # Z direction
        
        # Get dimensions
        dimensions = np.array([
            int(dcm.Rows),
            int(dcm.Columns),
            len(dcm_files)
        ])
        
        return {
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'dimensions': dimensions,
            'num_slices': len(dcm_files)
        }
        
    except Exception as e:
        print(f"    Error loading DICOM geometry: {e}")
        return None


def load_stl_mesh(stl_path: Path) -> Optional[trimesh.Trimesh]:
    """Load an STL file as a trimesh object."""
    try:
        mesh = trimesh.load(stl_path)
        return mesh
    except Exception as e:
        return None


def transform_mesh_to_image_space(mesh: trimesh.Trimesh, 
                                  geometry: Dict) -> trimesh.Trimesh:
    """
    Transform mesh vertices from physical space to image voxel space.
    
    Args:
        mesh: Trimesh in physical coordinates (LPS)
        geometry: DICOM geometry info
    
    Returns:
        Transformed mesh in voxel coordinates
    """
    # Get geometry parameters
    origin = geometry['origin']
    spacing = geometry['spacing']
    direction = geometry['direction']
    
    # Transform: Physical -> Voxel
    # voxel = inv(direction) * (physical - origin) / spacing
    
    vertices = mesh.vertices.copy()
    
    # Subtract origin
    vertices -= origin
    
    # Apply inverse direction matrix
    direction_inv = np.linalg.inv(direction)
    vertices = vertices @ direction_inv.T
    
    # Scale by spacing
    vertices /= spacing
    
    # Create new mesh with transformed vertices
    transformed_mesh = mesh.copy()
    transformed_mesh.vertices = vertices
    
    return transformed_mesh


def rasterize_mesh_to_slices(mesh: trimesh.Trimesh, 
                             dimensions: np.ndarray) -> np.ndarray:
    """
    Rasterize mesh to 3D binary volume.
    
    Args:
        mesh: Mesh in voxel coordinates
        dimensions: [rows, cols, slices]
    
    Returns:
        3D binary numpy array
    """
    # Use trimesh voxelization but at resolution matching dimensions
    bounds = mesh.bounds
    voxel_size = 1.0  # Already in voxel space
    
    # Create voxel grid
    try:
        voxelized = mesh.voxelized(pitch=voxel_size)
        voxel_grid = voxelized.matrix
        
        # Get voxel grid bounds
        grid_origin = voxelized.transform[:3, 3]
        
        # Create output volume
        volume = np.zeros(dimensions[::-1], dtype=np.uint8)  # [slices, rows, cols]
        
        # Map voxel grid to volume
        # This is a simplified approach - may need refinement
        x_min, y_min, z_min = np.maximum(np.floor(grid_origin).astype(int), 0)
        x_max = min(x_min + voxel_grid.shape[0], dimensions[0])
        y_max = min(y_min + voxel_grid.shape[1], dimensions[1])
        z_max = min(z_min + voxel_grid.shape[2], dimensions[2])
        
        # Copy voxel data to volume
        x_end = min(voxel_grid.shape[0], x_max - x_min)
        y_end = min(voxel_grid.shape[1], y_max - y_min)
        z_end = min(voxel_grid.shape[2], z_max - z_min)
        
        if x_end > 0 and y_end > 0 and z_end > 0:
            volume[z_min:z_max, y_min:y_max, x_min:x_max] = \
                voxel_grid[:x_end, :y_end, :z_end].transpose(2, 1, 0)
        
        # Fill holes in each slice
        for i in range(volume.shape[0]):
            volume[i] = binary_fill_holes(volume[i]).astype(np.uint8)
        
        return volume
        
    except Exception as e:
        print(f"      Error rasterizing mesh: {e}")
        return np.zeros(dimensions[::-1], dtype=np.uint8)


def save_volume_as_pngs(volume: np.ndarray, 
                        output_dir: Path) -> int:
    """
    Save 3D volume as PNG slices.
    
    Args:
        volume: 3D binary array [slices, rows, cols]
        output_dir: Output directory
    
    Returns:
        Number of slices saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_saved = 0
    for i in range(volume.shape[0]):
        slice_data = volume[i]
        
        # Skip empty slices
        if not slice_data.any():
            continue
        
        # Convert to 8-bit image (0 or 255)
        img_data = (slice_data * 255).astype(np.uint8)
        
        # Save as PNG
        img = Image.fromarray(img_data)
        img_path = output_dir / f"{i:04d}.png"
        img.save(img_path)
        num_saved += 1
    
    return num_saved


def parse_fcsv_biopsy_coords(fcsv_path: Path) -> Optional[Dict]:
    """Parse a .fcsv file to extract biopsy coordinates."""
    try:
        with open(fcsv_path, 'r') as f:
            lines = f.readlines()
        
        data_lines = [line.strip() for line in lines if not line.startswith('#')]
        
        if len(data_lines) < 2:
            return None
        
        coords = []
        labels = []
        for line in data_lines:
            if line:
                parts = line.split(',')
                if len(parts) >= 12:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    label = parts[11]
                    coords.append([x, y, z])
                    labels.append(label)
        
        if len(coords) >= 2:
            pathology = fcsv_path.stem.split('-', 2)[-1] if '-' in fcsv_path.stem else "Unknown"
            
            return {
                "top": coords[0],
                "bottom": coords[1],
                "pathology": pathology,
                "filename": fcsv_path.name
            }
    except Exception as e:
        pass
    
    return None


def process_overlay_directory(overlay_dir: Path, 
                              output_base_dir: Path,
                              processed_dir: Path,
                              class_num: int,
                              patient_number: str,
                              series_uid: str) -> Dict:
    """Process a single overlay directory with proper DICOM alignment."""
    stats = {
        "stl_files_found": 0,
        "stl_files_processed": 0,
        "fcsv_files_found": 0,
        "fcsv_files_processed": 0,
        "slices_created": 0,
        "dicom_found": False
    }
    
    case_dir = output_base_dir / f"class{class_num}" / f"case_{patient_number}"
    series_dir = case_dir / series_uid
    
    data_dir = overlay_dir / "Data"
    if not data_dir.exists():
        return stats
    
    # Find DICOM files for this series
    dicom_dir = find_dicom_files(processed_dir, class_num, patient_number, series_uid)
    
    if dicom_dir is None:
        # Fallback: try to find from processed structure
        processed_series_dir = processed_dir / f"class{class_num}" / f"case_{patient_number}" / series_uid
        if not processed_series_dir.exists():
            return stats
        
        # Try to infer geometry from manifest
        # This is a fallback - won't be as accurate
        return stats
    
    stats["dicom_found"] = True
    
    # Load DICOM geometry
    geometry = load_dicom_geometry(dicom_dir)
    if geometry is None:
        return stats
    
    # Process STL files
    stl_files = list(data_dir.glob("*.STL")) + list(data_dir.glob("*.stl"))
    stats["stl_files_found"] = len(stl_files)
    
    for stl_file in stl_files:
        mesh = load_stl_mesh(stl_file)
        if mesh is None:
            continue
        
        try:
            # Transform mesh to image voxel space
            transformed_mesh = transform_mesh_to_image_space(mesh, geometry)
            
            # Rasterize to volume
            volume = rasterize_mesh_to_slices(transformed_mesh, geometry['dimensions'])
            
            # Save as PNG slices
            mask_name = stl_file.stem.lower().replace(" ", "_")
            mask_output_dir = series_dir / mask_name
            
            num_slices = save_volume_as_pngs(volume, mask_output_dir)
            
            stats["slices_created"] += num_slices
            stats["stl_files_processed"] += 1
            
        except Exception as e:
            pass
    
    # Process FCSV files
    fcsv_files = list(data_dir.glob("*.fcsv"))
    stats["fcsv_files_found"] = len(fcsv_files)
    
    biopsies_data = []
    for fcsv_file in fcsv_files:
        biopsy_info = parse_fcsv_biopsy_coords(fcsv_file)
        if biopsy_info:
            biopsies_data.append(biopsy_info)
            stats["fcsv_files_processed"] += 1
    
    if biopsies_data:
        case_dir.mkdir(parents=True, exist_ok=True)
        biopsies_json = case_dir / "biopsies.json"
        
        existing_biopsies = []
        if biopsies_json.exists():
            with open(biopsies_json, 'r') as f:
                existing_biopsies = json.load(f)
        
        for biopsy in biopsies_data:
            biopsy['series_uid'] = series_uid
        
        all_biopsies = existing_biopsies + biopsies_data
        with open(biopsies_json, 'w') as f:
            json.dump(all_biopsies, f, indent=2)
    
    # Copy MRML file
    mrml_files = list(overlay_dir.glob("*.mrml"))
    if mrml_files:
        series_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mrml_files[0], series_dir / "scene.mrml")
    
    return stats


def main():
    """Main execution function."""
    
    print("="*80)
    print("Overlay Processing with Proper DICOM Alignment")
    print("="*80)
    
    # Configuration
    parquet_dir = "data/splitted_images"
    overlay_base_dir = "data/overlay/Biopsy Overlays (3D Slicer)"
    output_base_dir = "data/processed_seg"
    processed_dir = Path("data/processed")
    
    # Step 1: Load Series UIDs
    series_info_map = load_series_uids_from_parquet(parquet_dir)
    
    if not series_info_map:
        print("❌ Error: No Series UIDs found!")
        return
    
    # Step 2: Match overlay directories
    matched_dirs = match_overlay_dirs_with_series(overlay_base_dir, series_info_map)
    
    if not matched_dirs:
        print("❌ Error: No matching overlay directories found!")
        return
    
    # Step 3: Process with progress bar
    print(f"\n{'='*80}")
    print(f"Processing with DICOM-based alignment...")
    print(f"{'='*80}\n")
    
    total_stats = defaultdict(int)
    
    pbar = tqdm(matched_dirs.items(), 
                total=len(matched_dirs),
                desc="Processing cases",
                unit="case")
    
    for overlay_dir_str, info in pbar:
        overlay_dir = Path(overlay_dir_str)
        
        pbar.set_description(f"Class {info['class']}, Patient {info['patient_number']}")
        
        stats = process_overlay_directory(
            overlay_dir,
            Path(output_base_dir),
            processed_dir,
            info['class'],
            info['patient_number'],
            info['series_uid']
        )
        
        for key, value in stats.items():
            total_stats[key] += value
        
        pbar.set_postfix({
            'STL': f"{total_stats['stl_files_processed']}/{total_stats['stl_files_found']}",
            'DICOM': total_stats['dicom_found'],
            'Slices': total_stats['slices_created']
        })
    
    pbar.close()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"{'='*80}")
    print(f"\nSummary Statistics:")
    print(f"  Total matched cases: {len(matched_dirs)}")
    print(f"  Cases with DICOM found: {total_stats['dicom_found']}")
    print(f"  STL files processed: {total_stats['stl_files_processed']}/{total_stats['stl_files_found']}")
    print(f"  FCSV files processed: {total_stats['fcsv_files_processed']}/{total_stats['fcsv_files_found']}")
    print(f"  Total mask slices created: {total_stats['slices_created']}")
    print(f"\nOutput directory: {output_base_dir}/")


if __name__ == "__main__":
    main()

