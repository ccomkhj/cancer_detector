# Overlay Data Processing Tool

## Overview

This tool matches biopsy overlay data with MRI Series Instance UIDs and converts 3D segmentation meshes (STL files) to 2D PNG masks suitable for training deep learning models.

## What It Does

### 1. **Match Overlay Data with Series UIDs**
   - Reads Series Instance UIDs from parquet files (`data/splitted_images/`)
   - Extracts Series UIDs from overlay directory names
   - Matches and filters to keep only relevant cases

### 2. **Convert STL Meshes to PNG Masks**
   - Loads 3D STL mesh files (Prostate gland, Target lesions)
   - Voxelizes meshes into 3D grids
   - Extracts 2D slices and saves as PNG images
   - Creates binary masks (0 = background, 255 = segmentation)

### 3. **Process Biopsy Coordinates**
   - Parses FCSV files containing biopsy needle coordinates
   - Extracts pathology labels (Benign, GG1, GG2+, etc.)
   - Saves as JSON for easy access

## Requirements

```bash
pip install pandas pyarrow trimesh numpy pillow SimpleITK pydicom
# or
pip install -r requirements.txt
```

## Usage

```bash
conda activate mri
python tools/process_overlay_to_masks.py
```

## Input Data

### Parquet Files
Located in: `data/splitted_images/class={1,2,3,4}/PIRADS_*.parquet`

Must contain column:
- `Series Instance UID (MRI)` - Links to overlay data

### Overlay Data
Located in: `data/overlay/Biopsy Overlays (3D Slicer)/`

Directory naming pattern:
```
Prostate-MRI-US-Biopsy-{PatientID}-{Type}-seriesUID-{SeriesInstanceUID} (US-date-{Date})/
├── BiopsyVectors.mrml
└── Data/
    ├── Prostate.STL          # Prostate gland mesh
    ├── Target1.STL           # Lesion mesh
    ├── Bx-1-Benign.fcsv     # Biopsy coordinates
    └── ...
```

## Output Structure

```
data/overlay_processed/
├── class=1/
│   ├── patient_0001/
│   │   ├── masks/
│   │   │   ├── prostate_0000.png
│   │   │   ├── prostate_0001.png
│   │   │   ├── ...
│   │   │   ├── target1_0000.png
│   │   │   ├── target1_0001.png
│   │   │   └── ...
│   │   ├── biopsies/
│   │   │   └── biopsies.json
│   │   └── scene.mrml
│   └── patient_0002/
│       └── ...
├── class=2/
├── class=3/
└── class=4/
```

## Output Files

### Mask PNGs
- **Filename format**: `{structure}_{slice_number:04d}.png`
  - `prostate_0000.png`, `prostate_0001.png`, ...
  - `target1_0000.png`, `target2_0000.png`, ...
- **Format**: Grayscale PNG (8-bit)
- **Values**: 0 (background), 255 (segmentation)
- **Empty slices**: Skipped (not saved)

### Biopsy JSON
```json
[
  {
    "top": [-5.644, 23.161, -32.457],
    "bottom": [-11.822, 39.152, -45.904],
    "pathology": "GG2+",
    "filename": "Bx-2-GG2+.fcsv"
  },
  {
    "top": [...],
    "bottom": [...],
    "pathology": "Benign",
    "filename": "Bx-1-Benign.fcsv"
  }
]
```

### MRML Scene
- Copy of original 3D Slicer scene file for reference
- Contains all visualization and annotation metadata

## Configuration

Edit these variables in `main()`:

```python
parquet_dir = "data/splitted_images"          # Source of Series UIDs
overlay_base_dir = "data/overlay/Biopsy Overlays (3D Slicer)"  # Input overlays
output_base_dir = "data/overlay_processed"    # Output directory
voxel_size = 0.5  # mm per voxel (smaller = higher resolution, larger files)
```

## Voxel Size Parameter

The `voxel_size` parameter controls the resolution of the voxelization:

- **0.5 mm** (default): High resolution, ~2x native MRI resolution
- **1.0 mm**: Medium resolution, ~native MRI resolution
- **2.0 mm**: Low resolution, faster processing

Smaller values create more detailed masks but:
- Take longer to process
- Create more PNG files
- Use more disk space

## Statistics Reported

The script reports:
- Number of matched overlay directories
- Distribution by class (1-4)
- STL files found and processed
- FCSV files found and processed
- Total mask slices created

## Example Output

```
================================================================================
Overlay Data Processing: Match and Convert to PNG Masks
================================================================================

Loading Series Instance UIDs from data/splitted_images...
  Class 1: Added 17 series UIDs from PIRADS_0.parquet
  Class 2: Added 60 series UIDs from PIRADS_3.parquet
  Class 3: Added 60 series UIDs from PIRADS_4.parquet
  Class 4: Added 60 series UIDs from PIRADS_5.parquet

✓ Total unique series UIDs: 197

Matching overlay directories with Series UIDs...

  ✓ Matched directories: 45
  ✗ Unmatched directories: 2996

  Distribution by class:
    Class 1: 5 cases
    Class 2: 15 cases
    Class 3: 12 cases
    Class 4: 13 cases

Processing matched overlay directories...

[1/45] Patient 0001 (Class 4)
  Directory: Prostate-MRI-US-Biopsy-0001-BXmr-seriesUID-1.3.6.1.4.1.14519...
    Processing Prostate.STL...
      → Created 156 mask slices
    Processing Target1.STL...
      → Created 23 mask slices
    Saved 17 biopsy annotations to biopsies.json

...

Processing Complete!

Summary Statistics:
  Total matched cases: 45
  STL files found: 95
  STL files processed: 92
  FCSV files found: 780
  FCSV files processed: 765
  Total mask slices created: 8,432

Output directory: data/overlay_processed/
```

## Next Steps

After processing:

1. **Align with DICOM slices**: Match PNG mask slices with corresponding MRI slices
2. **Create training pairs**: Combine MRI images + segmentation masks
3. **Data augmentation**: Rotate, flip, scale for training
4. **Train segmentation model**: U-Net, nnU-Net, or similar architecture

## Troubleshooting

### No matches found
- Check that parquet files contain `Series Instance UID (MRI)` column
- Verify overlay directory names follow expected pattern
- Ensure Series UIDs match exactly (no extra spaces/characters)

### STL loading errors
- Some STL files may be corrupted or use unsupported format
- Script will skip failed files and continue processing

### Out of memory
- Increase voxel_size (e.g., 1.0 or 2.0 mm)
- Process fewer cases at a time
- Use a machine with more RAM

### Empty mask slices
- Script automatically skips slices with no segmentation
- This is normal and saves disk space

## Technical Details

### Coordinate System
- Overlay data uses LPS (Left-Posterior-Superior) coordinate system
- Voxelization preserves spatial relationships
- Origin point is saved for future alignment

### Mesh Processing
- Uses `trimesh` library for robust STL loading
- Voxelization creates binary 3D grids
- Slices extracted along Z-axis (inferior-superior)

### Biopsy Coordinates
- FCSV files contain fiducial markers in 3D space
- Top/Bottom represent needle entry/exit points
- Coordinates are in mm, LPS coordinate system

