#!/usr/bin/env python3
"""
Generate TCIA manifest files for each class partition.

This script reads parquet files from data/splitted/class={1,2,3,4}/ directories,
extracts unique Series Instance UIDs (MRI), and generates separate TCIA manifest
files for each class that can be used to download DICOM data from TCIA.

Usage:
    python tcia_generator.py

Requirements:
    pip install pandas pyarrow
"""

import pandas as pd
import os
import time
from pathlib import Path
from datetime import datetime


def read_series_uids_from_class(class_num, base_dir="data/splitted"):
    """
    Read all parquet files for a given class and extract unique Series Instance UIDs.

    Args:
        class_num (int): Class number (1-4)
        base_dir (str): Base directory containing class partitions

    Returns:
        list: List of unique Series Instance UIDs
    """
    class_dir = Path(base_dir) / f"class={class_num}"

    if not class_dir.exists():
        print(f"Warning: Directory {class_dir} does not exist")
        return []

    # Find all parquet files in the class directory
    parquet_files = list(class_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"Warning: No parquet files found in {class_dir}")
        return []

    print(f"\nProcessing class {class_num}:")
    print(f"  Found {len(parquet_files)} parquet file(s)")

    # Collect all Series Instance UIDs
    all_series_uids = []

    for parquet_file in parquet_files:
        print(f"  Reading: {parquet_file.name}")
        df = pd.read_parquet(parquet_file)

        # Extract Series Instance UID (MRI) column
        if "Series Instance UID (MRI)" in df.columns:
            series_uids = df["Series Instance UID (MRI)"].dropna().tolist()
            all_series_uids.extend(series_uids)
            print(f"    Found {len(series_uids)} series UIDs")
        else:
            print(f"    Warning: 'Series Instance UID (MRI)' column not found")

    # Get unique UIDs and sort them
    unique_uids = sorted(list(set(all_series_uids)))
    print(f"  Total unique series UIDs: {len(unique_uids)}")

    return unique_uids


def generate_tcia_manifest(series_uids, output_path, class_num):
    """
    Generate a TCIA manifest file with the given series UIDs.

    Args:
        series_uids (list): List of Series Instance UIDs
        output_path (str or Path): Output path for the manifest file
        class_num (int): Class number for the manifest
    """
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for databasketId
    timestamp = int(time.time() * 1000)  # milliseconds since epoch

    # Create manifest content
    manifest_lines = [
        "downloadServerUrl=https://nbia.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet",
        "includeAnnotation=true",
        "noOfrRetry=4",
        f"databasketId=manifest-class{class_num}-{timestamp}.tcia",
        "manifestVersion=3.0",
        "ListOfSeriesToDownload=",
    ]

    # Add series UIDs
    manifest_lines.extend(series_uids)

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(manifest_lines))

    print(f"  ✓ Generated: {output_path}")
    print(f"    Contains {len(series_uids)} series UIDs")


def main():
    """
    Main function to generate TCIA manifest files for all classes.
    """
    print("=" * 70)
    print("TCIA Manifest Generator")
    print("=" * 70)

    # Configuration
    base_dir = "data/splitted_images"
    output_dir = "data/tcia"

    # Process each class (1-4)
    for class_num in range(1, 5):
        # Read series UIDs for this class
        series_uids = read_series_uids_from_class(class_num, base_dir)

        if series_uids:
            # Generate manifest file
            output_path = Path(output_dir) / f"class{class_num}.tcia"
            generate_tcia_manifest(series_uids, output_path, class_num)
        else:
            print(f"  ⚠ Skipping class {class_num} (no series UIDs found)")

    print("\n" + "=" * 70)
    print("✓ TCIA manifest generation complete!")
    print("=" * 70)
    print(f"\nManifest files saved to: {output_dir}/")
    print("\nNext steps:")
    print("1. Use the generated .tcia files to download DICOM data")
    print("2. Downloaded files will be saved to data/nbia/")


if __name__ == "__main__":
    main()
