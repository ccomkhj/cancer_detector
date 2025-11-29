#!/usr/bin/env python3
"""
Diagnose alignment issues between masks and images.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

# Check a specific case
case_dir = Path("data/processed/class3/case_0045")
seg_dir = Path("data/processed_seg/class3/case_0045")

# Find a series
series_dirs = list(case_dir.glob("*"))
if series_dirs:
    series_uid = series_dirs[0].name
    print(f"Checking series: {series_uid[:50]}...")
    
    # Load an image
    img_path = case_dir / series_uid / "images" / "0013.png"
    if img_path.exists():
        img = np.array(Image.open(img_path))
        print(f"\nImage shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Image range: [{img.min()}, {img.max()}]")
    
    # Load corresponding mask
    mask_path = seg_dir / series_uid / "prostate" / "0013.png"
    if mask_path.exists():
        mask = np.array(Image.open(mask_path))
        print(f"\nMask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Mask range: [{mask.min()}, {mask.max()}]")
        print(f"Mask non-zero pixels: {np.count_nonzero(mask)}")
    else:
        print(f"\nMask not found at: {mask_path}")
    
    # Check manifest for spacing info
    manifest_path = Path("data/processed/class3/manifest.csv")
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        case_data = df[df['case_id'] == '0045'].iloc[0]
        print(f"\nDICOM spacing from manifest:")
        print(f"  X: {case_data['spacing_x']:.4f} mm")
        print(f"  Y: {case_data['spacing_y']:.4f} mm")
        print(f"  Z: {case_data['spacing_z']:.4f} mm")
        
        print(f"\nVoxel size used in STL conversion: 0.5 mm")
        print(f"Resolution mismatch factor:")
        print(f"  X: {0.5 / case_data['spacing_x']:.2f}x")
        print(f"  Y: {0.5 / case_data['spacing_y']:.2f}x")
        print(f"  Z: {0.5 / case_data['spacing_z']:.2f}x")

