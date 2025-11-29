#!/usr/bin/env python3
"""
Merge multiple data sources into enriched patient records.

Combines three Excel data sources:
1. MRI_images: Already processed parquet files with image metadata
2. MRI_target: Target lesion information
3. MRI_biopsy: Biopsy core information

Output maintains class-based partitioning with enriched data.

Usage:
    python merge_datasets.py

Requirements:
    pip install pandas openpyxl pyarrow
"""

import pandas as pd
import os
import re
from pathlib import Path
from typing import Dict, List


def normalize_patient_id(patient_id):
    """
    Normalize patient ID to integer format.
    
    Handles various formats:
    - "Prostate-MRI-US-Biopsy-0144" → 144
    - "144" → 144
    - 144 → 144
    
    Args:
        patient_id: Patient identifier (string or int)
    
    Returns:
        int: Normalized patient number
    """
    if pd.isna(patient_id):
        return None
    
    if isinstance(patient_id, str):
        # Extract number from "Prostate-MRI-US-Biopsy-0144" format
        if "Biopsy" in patient_id:
            return int(patient_id.split("-")[-1])
        # Handle string numbers
        return int(patient_id)
    
    return int(patient_id)


def load_image_data(base_dir="data/splitted_images"):
    """
    Load all parquet files from splitted_images directory.
    
    Args:
        base_dir (str): Base directory containing class partitions
    
    Returns:
        pd.DataFrame: Combined dataframe from all parquet files
    """
    base_path = Path(base_dir)
    all_dfs = []
    
    print(f"\n{'='*70}")
    print("Loading MRI Image Data")
    print(f"{'='*70}")
    
    for class_dir in sorted(base_path.glob("class=*")):
        class_num = int(class_dir.name.split("=")[1])
        print(f"\nClass {class_num}:")
        
        for parquet_file in sorted(class_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            # Store original class and PIRADS file info
            df["source_class"] = class_num
            df["source_file"] = parquet_file.name
            all_dfs.append(df)
            print(f"  Loaded {parquet_file.name}: {len(df)} rows")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Normalize patient_number to int for consistent joining
    combined_df["patient_number"] = combined_df["patient_number"].apply(normalize_patient_id)
    
    print(f"\nTotal image records: {len(combined_df)}")
    print(f"Unique patients: {combined_df['patient_number'].nunique()}")
    
    return combined_df


def load_target_data(file_path="data/raw/Target-Data_2019-12-05-2.xlsx"):
    """
    Load target lesion data from Excel file.
    
    Args:
        file_path (str): Path to target data Excel file
    
    Returns:
        pd.DataFrame: Target data with normalized patient IDs
    """
    print(f"\n{'='*70}")
    print("Loading MRI Target Data")
    print(f"{'='*70}")
    
    df = pd.read_excel(file_path)
    print(f"Total target records: {len(df)}")
    
    # Normalize patient ID
    df["patient_number_normalized"] = df["Patient ID"].apply(normalize_patient_id)
    
    print(f"Unique patients with targets: {df['patient_number_normalized'].nunique()}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Add prefix to avoid column conflicts (except join key)
    rename_cols = {col: f"target_{col}" for col in df.columns 
                   if col not in ["patient_number_normalized", "Patient ID"]}
    df = df.rename(columns=rename_cols)
    
    return df


def load_biopsy_data(file_path="data/raw/TCIA-Biopsy-Data_2020-07-14.xlsx"):
    """
    Load biopsy data from Excel file.
    
    Args:
        file_path (str): Path to biopsy data Excel file
    
    Returns:
        pd.DataFrame: Biopsy data with normalized patient IDs
    """
    print(f"\n{'='*70}")
    print("Loading MRI Biopsy Data")
    print(f"{'='*70}")
    
    df = pd.read_excel(file_path)
    print(f"Total biopsy records: {len(df)}")
    
    # Normalize patient ID (already called "Patient Number")
    df["patient_number_normalized"] = df["Patient Number"].apply(normalize_patient_id)
    
    print(f"Unique patients with biopsies: {df['patient_number_normalized'].nunique()}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Add prefix to avoid column conflicts (except join key)
    rename_cols = {col: f"biopsy_{col}" for col in df.columns 
                   if col not in ["patient_number_normalized", "Patient Number"]}
    df = df.rename(columns=rename_cols)
    
    return df


def merge_datasets(images_df, target_df, biopsy_df):
    """
    Merge all three datasets on patient_number.
    
    Performs left joins to preserve all image records.
    Handles one-to-many relationships (multiple targets/biopsies per patient).
    
    Args:
        images_df (pd.DataFrame): Base image data
        target_df (pd.DataFrame): Target lesion data
        biopsy_df (pd.DataFrame): Biopsy data
    
    Returns:
        pd.DataFrame: Merged dataframe
    """
    print(f"\n{'='*70}")
    print("Merging Datasets")
    print(f"{'='*70}")
    
    # Merge with target data
    print("\nMerging with target data...")
    merged = images_df.merge(
        target_df,
        left_on="patient_number",
        right_on="patient_number_normalized",
        how="left",
        suffixes=("", "_target_dup")
    )
    print(f"  After target merge: {len(merged)} rows")
    
    # Merge with biopsy data
    print("Merging with biopsy data...")
    merged = merged.merge(
        biopsy_df,
        left_on="patient_number",
        right_on="patient_number_normalized",
        how="left",
        suffixes=("", "_biopsy_dup")
    )
    print(f"  After biopsy merge: {len(merged)} rows")
    
    # Clean up duplicate normalized columns
    cols_to_drop = [col for col in merged.columns if "normalized" in col or "_dup" in col]
    merged = merged.drop(columns=cols_to_drop)
    
    print(f"\nFinal merged dataset:")
    print(f"  Total rows: {len(merged)}")
    print(f"  Total columns: {len(merged.columns)}")
    print(f"  Unique patients: {merged['patient_number'].nunique()}")
    
    # Statistics - find target and biopsy columns that weren't renamed
    target_id_col = "Patient ID" if "Patient ID" in merged.columns else None
    biopsy_id_col = "Patient Number" if "Patient Number" in merged.columns else None
    
    # If those don't exist, find any target_ or biopsy_ prefixed column to check for data
    if target_id_col is None:
        target_cols = [col for col in merged.columns if col.startswith("target_")]
        target_id_col = target_cols[0] if target_cols else None
    
    if biopsy_id_col is None:
        biopsy_cols = [col for col in merged.columns if col.startswith("biopsy_")]
        biopsy_id_col = biopsy_cols[0] if biopsy_cols else None
    
    print(f"\nData coverage:")
    if target_id_col:
        patients_with_targets = merged[merged[target_id_col].notna()]["patient_number"].nunique()
        print(f"  Patients with target data: {patients_with_targets}")
    if biopsy_id_col:
        patients_with_biopsies = merged[merged[biopsy_id_col].notna()]["patient_number"].nunique()
        print(f"  Patients with biopsy data: {patients_with_biopsies}")
    
    return merged


def save_merged_data(merged_df, output_dir="data/splitted_info"):
    """
    Save merged data back into class-based partitions.
    
    Args:
        merged_df (pd.DataFrame): Merged dataframe
        output_dir (str): Output directory for parquet files
    """
    print(f"\n{'='*70}")
    print("Saving Merged Data")
    print(f"{'='*70}")
    
    output_path = Path(output_dir)
    
    # Group by source class and file
    for (class_num, source_file), group_df in merged_df.groupby(["source_class", "source_file"]):
        class_dir = output_path / f"class={class_num}"
        class_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = class_dir / source_file
        group_df.to_parquet(output_file, index=False, engine="pyarrow")
        
        print(f"  Saved class={class_num}/{source_file}: {len(group_df)} rows")
    
    print(f"\n✓ All merged data saved to {output_dir}/")


def validate_merge(original_df, merged_df):
    """
    Validate the merge operation.
    
    Args:
        original_df (pd.DataFrame): Original image data
        merged_df (pd.DataFrame): Merged data
    """
    print(f"\n{'='*70}")
    print("Validation Report")
    print(f"{'='*70}")
    
    # Check patient coverage
    original_patients = set(original_df["patient_number"].unique())
    merged_patients = set(merged_df["patient_number"].unique())
    
    print(f"\nPatient coverage:")
    print(f"  Original unique patients: {len(original_patients)}")
    print(f"  Merged unique patients: {len(merged_patients)}")
    print(f"  Patients lost: {len(original_patients - merged_patients)}")
    
    if original_patients != merged_patients:
        lost_patients = original_patients - merged_patients
        print(f"  WARNING: Lost patients: {sorted(lost_patients)}")
    else:
        print(f"  ✓ All patients preserved")
    
    # Row multiplication analysis
    original_rows = len(original_df)
    merged_rows = len(merged_df)
    multiplication_factor = merged_rows / original_rows
    
    print(f"\nRow multiplication:")
    print(f"  Original rows: {original_rows}")
    print(f"  Merged rows: {merged_rows}")
    print(f"  Multiplication factor: {multiplication_factor:.2f}x")
    
    # Sample patient analysis
    sample_patient = merged_df["patient_number"].iloc[0]
    sample_rows = merged_df[merged_df["patient_number"] == sample_patient]
    
    # Find representative target and biopsy columns
    target_cols = [col for col in merged_df.columns if col.startswith("target_")]
    biopsy_cols = [col for col in merged_df.columns if col.startswith("biopsy_")]
    
    print(f"\nSample patient analysis (Patient {sample_patient}):")
    print(f"  Number of rows: {len(sample_rows)}")
    if target_cols:
        print(f"  Has target data: {sample_rows[target_cols[0]].notna().any()}")
    if biopsy_cols:
        print(f"  Has biopsy data: {sample_rows[biopsy_cols[0]].notna().any()}")
    
    # Column analysis
    print(f"\nColumn breakdown:")
    original_cols = [col for col in merged_df.columns if not col.startswith(("target_", "biopsy_", "source_"))]
    target_cols = [col for col in merged_df.columns if col.startswith("target_")]
    biopsy_cols = [col for col in merged_df.columns if col.startswith("biopsy_")]
    
    print(f"  Original columns: {len(original_cols)}")
    print(f"  Target columns: {len(target_cols)}")
    print(f"  Biopsy columns: {len(biopsy_cols)}")
    print(f"  Total columns: {len(merged_df.columns)}")


def main():
    """
    Main function to merge all datasets.
    """
    print("\n" + "="*70)
    print("MRI DATASET MERGER")
    print("="*70)
    
    # Load all data sources
    images_df = load_image_data()
    target_df = load_target_data()
    biopsy_df = load_biopsy_data()
    
    # Merge datasets
    merged_df = merge_datasets(images_df, target_df, biopsy_df)
    
    # Validate merge
    validate_merge(images_df, merged_df)
    
    # Save merged data
    save_merged_data(merged_df)
    
    print("\n" + "="*70)
    print("✓ MERGE COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review merged data in data/splitted_info/")
    print("2. Use merged data for enriched analysis")
    print("3. Generate TCIA manifests from splitted_info if needed")


if __name__ == "__main__":
    main()

