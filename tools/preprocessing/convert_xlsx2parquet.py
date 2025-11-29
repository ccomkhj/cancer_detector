#!/usr/bin/env python3
"""
Convert Excel file with multiple sheets to separate parquet files with class-based partitioning.
Each sheet represents a PIRADS score and will be mapped to a class:
- Class 1: PIRADS 0, 1, 2 (combined)
- Class 2: PIRADS 3
- Class 3: PIRADS 4
- Class 4: PIRADS 5

Usage:
    python convert_xlsx2parquet.py

Requirements:
    pip install pandas openpyxl pyarrow
"""

import pandas as pd
import os
import re
from pathlib import Path


def extract_pirads_score(sheet_name):
    """
    Extract PIRADS score from sheet name.

    Args:
        sheet_name (str): Sheet name (e.g., 'PIRADS_5', 'PIRADS_0')

    Returns:
        int: PIRADS score or None if not found
    """
    match = re.search(r"PIRADS?_?(\d+)", sheet_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def map_pirads_to_class(pirads_score):
    """
    Map PIRADS score to class.

    Args:
        pirads_score (int): PIRADS score (0-5)

    Returns:
        int: Class (1-4)
    """
    if pirads_score in [0, 1, 2]:
        return 1
    elif pirads_score == 3:
        return 2
    elif pirads_score == 4:
        return 3
    elif pirads_score == 5:
        return 4
    else:
        return None


def convert_xlsx_to_parquet(xlsx_file, output_dir=None):
    """
    Convert each sheet in an Excel file to a separate parquet file with class partitioning.

    Args:
        xlsx_file (str): Path to the Excel file
        output_dir (str, optional): Output directory for parquet files.
                                   If None, uses the same directory as the Excel file.
    """
    # Get the base name without extension
    base_name = Path(xlsx_file).stem

    # Set output directory
    if output_dir is None:
        output_dir = Path(xlsx_file).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Read the Excel file
    print(f"Reading Excel file: {xlsx_file}")
    xl_file = pd.ExcelFile(xlsx_file)

    print(f"Found {len(xl_file.sheet_names)} sheets: {xl_file.sheet_names}")

    # Convert each sheet to parquet
    for sheet_name in xl_file.sheet_names:
        print(f"\nProcessing sheet: {sheet_name}")

        # Read the sheet
        df = pd.read_excel(xl_file, sheet_name=sheet_name)
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # Extract PIRADS score from sheet name
        pirads_score = extract_pirads_score(sheet_name)

        class_value = None
        if pirads_score is not None:
            # Map to class
            class_value = map_pirads_to_class(pirads_score)

            if class_value is not None:
                # Add class column to dataframe
                df["class"] = class_value
                print(f"  PIRADS score: {pirads_score} → Class: {class_value}")
            else:
                print(
                    f"  Warning: Could not map PIRADS score {pirads_score} to a class"
                )
        else:
            print(f"  Warning: Could not extract PIRADS score from sheet name")

        # Create output filename (sanitize sheet name for filesystem)
        safe_sheet_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in sheet_name
        )

        # Create class-based subdirectory structure
        if class_value is not None:
            class_dir = output_dir / f"class={class_value}"
            class_dir.mkdir(parents=True, exist_ok=True)
            output_file = class_dir / f"{safe_sheet_name}.parquet"
        else:
            output_file = output_dir / f"{safe_sheet_name}.parquet"

        # Save to parquet
        df.to_parquet(output_file, index=False, engine="pyarrow")
        print(f"  Saved to: {output_file}")

    print(
        f"\n✓ Successfully converted {len(xl_file.sheet_names)} sheets to parquet files"
    )


if __name__ == "__main__":
    # Configuration
    xlsx_file = "data/raw/selected_patients_3.xlsx"

    # Check if file exists
    if not os.path.exists(xlsx_file):
        print(f"Error: File '{xlsx_file}' not found!")
        exit(1)

    # Convert to parquet
    convert_xlsx_to_parquet(xlsx_file, output_dir="data/splitted")
