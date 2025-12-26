#!/usr/bin/env python3
"""
Generate TCIA manifest files organized by class for different MRI sequences.

This script reads the NBIA manifest Excel file and creates separate TCIA files
for each class based on PIRADS scores, for different sequence types:
- T2: t2_spc_rst_axial obl_Prostate
- ADC: ep2d-advdiff-3Scan-4bval_spair_std_ADC
- CALC_BVAL: ep2d-advdiff-3Scan-4bval_spair_std_CALC_BVAL

Requirements:
    pip install pandas openpyxl pyarrow
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict


# Sequence type mapping
SEQUENCE_CONFIGS = {
    "t2": {
        "series_description": "t2_spc_rst_axial obl_Prostate",
        "output_dir": "data/tcia/t2",
    },
    "ep2d_adc": {
        "series_description": "ep2d-advdiff-3Scan-4bval_spair_std_ADC",
        "output_dir": "data/tcia/ep2d_adc",
    },
    "ep2d_calc": {
        "series_description": "ep2d-advdiff-3Scan-4bval_spair_std_CALC_BVAL",
        "output_dir": "data/tcia/ep2d_calc",
    },
}


def load_patient_class_mapping(parquet_dir):
    """
    Load patient IDs and their corresponding classes from parquet files.
    
    Args:
        parquet_dir (str): Directory containing class-organized parquet files
    
    Returns:
        dict: Mapping of Patient ID to class number
    """
    patient_class_map = {}
    parquet_path = Path(parquet_dir)
    
    print(f"\nLoading patient-class mappings from {parquet_dir}...")
    
    # Iterate through class directories
    for class_dir in sorted(parquet_path.glob("class=*")):
        class_num = int(class_dir.name.split("=")[1])
        
        # Read all parquet files in this class directory
        for parquet_file in class_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            
            # Try different possible column names for Patient ID
            patient_id_col = None
            for col_name in ["patient_number", "Patient ID", "Subject ID", "PatientID", "SubjectID"]:
                if col_name in df.columns:
                    patient_id_col = col_name
                    break
            
            if patient_id_col:
                patient_ids = df[patient_id_col].dropna().unique()
                for patient_id in patient_ids:
                    patient_class_map[str(patient_id)] = class_num
                print(f"  Class {class_num}: Added {len(patient_ids)} patients from {parquet_file.name} (using column '{patient_id_col}')")
            else:
                print(f"  Warning: No patient ID column found in {parquet_file.name}")
                print(f"    Available columns: {list(df.columns)}")
    
    print(f"✓ Total patients mapped: {len(patient_class_map)}")
    return patient_class_map


def load_manifest_data(manifest_file):
    """
    Load the NBIA manifest Excel file.
    
    Args:
        manifest_file (str): Path to the manifest Excel file
    
    Returns:
        pd.DataFrame: Manifest data
    """
    print(f"\nLoading manifest file: {manifest_file}")
    
    # Try to read the first sheet
    xl_file = pd.ExcelFile(manifest_file)
    print(f"  Available sheets: {xl_file.sheet_names}")
    
    # Read the first sheet (usually contains the manifest data)
    df = pd.read_excel(xl_file, sheet_name=0)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Show unique series descriptions
    if "Series Description" in df.columns:
        series_desc_counts = df["Series Description"].value_counts()
        print(f"\n  Top 10 Series Descriptions:")
        for desc, count in series_desc_counts.head(10).items():
            print(f"    {desc}: {count}")
    
    return df


def generate_tcia_files(manifest_df, patient_class_map, sequence_configs):
    """
    Generate TCIA files organized by class for different sequences.
    
    Args:
        manifest_df (pd.DataFrame): Manifest data
        patient_class_map (dict): Patient ID to class mapping
        sequence_configs (dict): Configuration for each sequence type
    """
    
    # Group series by class and sequence type
    series_by_class_and_seq = {
        seq_name: defaultdict(list) 
        for seq_name in sequence_configs.keys()
    }
    
    print("\n" + "="*80)
    print("Processing manifest data...")
    print("="*80)
    
    # Statistics
    total_rows = len(manifest_df)
    matched_patients = 0
    unmatched_patients = set()
    
    # Iterate through manifest rows
    for idx, row in manifest_df.iterrows():
        # Try different column names for patient ID
        patient_id = (
            row.get("Subject ID") or 
            row.get("Patient ID") or 
            row.get("PatientID") or
            row.get("SubjectID")
        )
        
        # Try different column names for series UID
        series_uid = (
            row.get("Series Instance UID") or 
            row.get("Series UID") or
            row.get("SeriesInstanceUID")
        )
        
        series_desc = row.get("Series Description", "")
        
        if pd.isna(patient_id) or pd.isna(series_uid) or pd.isna(series_desc):
            continue
        
        # Convert to string to handle any type issues
        series_desc = str(series_desc)
        
        # Convert patient_id to string for matching
        patient_id = str(patient_id)
        
        # Check if patient is in our class mapping
        if patient_id not in patient_class_map:
            unmatched_patients.add(patient_id)
            continue
        
        matched_patients += 1
        class_num = patient_class_map[patient_id]
        
        # Check each sequence type
        for seq_name, config in sequence_configs.items():
            if config["series_description"] in series_desc:
                series_by_class_and_seq[seq_name][class_num].append(series_uid)
    
    print(f"\nMatching Statistics:")
    print(f"  Total manifest rows: {total_rows}")
    print(f"  Matched patient rows: {matched_patients}")
    print(f"  Unique unmatched patients: {len(unmatched_patients)}")
    
    # Generate TCIA files for each sequence type and class
    for seq_name, config in sequence_configs.items():
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Generating TCIA files for: {seq_name}")
        print(f"  Series Description: {config['series_description']}")
        print(f"  Output Directory: {output_dir}")
        print(f"{'='*80}")
        
        series_dict = series_by_class_and_seq[seq_name]
        
        if not series_dict:
            print(f"  ⚠️  No series found for {seq_name}")
            continue
        
        # Create TCIA file for each class
        for class_num in sorted(series_dict.keys()):
            series_list = series_dict[class_num]
            
            if not series_list:
                continue
            
            # Create TCIA file content
            tcia_filename = output_dir / f"class{class_num}.tcia"
            
            tcia_content = [
                "downloadServerUrl=https://nbia.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet",
                "includeAnnotation=true",
                "noOfrRetry=4",
                f"databasketId=manifest-class{class_num}-{seq_name}.tcia",
                "manifestVersion=3.0",
                "ListOfSeriesToDownload=",
            ]
            
            # Add all series UIDs
            tcia_content.extend(series_list)
            
            # Write TCIA file
            with open(tcia_filename, "w") as f:
                f.write("\n".join(tcia_content))
            
            print(f"  ✓ Class {class_num}: {len(series_list)} series → {tcia_filename}")
        
        total_series = sum(len(series_list) for series_list in series_dict.values())
        print(f"  Total series for {seq_name}: {total_series}")


def main():
    """Main execution function."""
    
    print("="*80)
    print("TCIA Manifest Generator by Class and Sequence Type")
    print("="*80)
    
    # Configuration
    manifest_file = "data/raw/Prostate-MRI-US-Biopsy-NBIA-manifest_v2_20231020-nbia-digest.xlsx"
    parquet_dir = "data/splitted_images"
    
    # Load patient-class mapping from parquet files
    patient_class_map = load_patient_class_mapping(parquet_dir)
    
    if not patient_class_map:
        print("❌ Error: No patient-class mappings found!")
        return
    
    # Load manifest data
    manifest_df = load_manifest_data(manifest_file)
    
    # Generate TCIA files
    generate_tcia_files(manifest_df, patient_class_map, SEQUENCE_CONFIGS)
    
    print("\n" + "="*80)
    print("✓ TCIA file generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()

