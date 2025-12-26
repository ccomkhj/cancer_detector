#!/usr/bin/env python3
"""
Generate TCIA manifest files by Study Instance UID organized by class.

This script creates TCIA files that download entire studies (all sequences) 
for patients who have at least one of the target sequence types:
- t2_spc_rst_axial obl_Prostate
- ep2d-advdiff-3Scan-4bval_spair_std_ADC
- ep2d-advdiff-3Scan-4bval_spair_std_CALC_BVAL

Requirements:
    pip install pandas openpyxl pyarrow
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict


# Target sequence descriptions
TARGET_SEQUENCES = [
    "t2_spc_rst_axial obl_Prostate",
    "ep2d-advdiff-3Scan-4bval_spair_std_ADC",
    "ep2d-advdiff-3Scan-4bval_spair_std_CALC_BVAL",
]


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
    
    return df


def generate_study_tcia_files(manifest_df, patient_class_map, output_dir):
    """
    Generate TCIA files by Study Instance UID organized by class.
    
    Args:
        manifest_df (pd.DataFrame): Manifest data
        patient_class_map (dict): Patient ID to class mapping
        output_dir (str): Output directory for TCIA files
    """
    
    print("\n" + "="*80)
    print("Processing manifest data for Study Instance UIDs...")
    print("="*80)
    
    # Track which studies have target sequences per patient
    study_by_class = defaultdict(set)  # class -> set of study UIDs
    patient_study_map = defaultdict(set)  # patient -> set of study UIDs
    study_has_target_seq = set()  # studies that have at least one target sequence
    
    # Statistics
    total_rows = len(manifest_df)
    matched_patients = 0
    unmatched_patients = set()
    
    # First pass: identify studies with target sequences
    print("\nFirst pass: Identifying studies with target sequences...")
    for idx, row in manifest_df.iterrows():
        # Get patient ID
        patient_id = (
            row.get("Subject ID") or 
            row.get("Patient ID") or 
            row.get("PatientID") or
            row.get("SubjectID")
        )
        
        # Get Study Instance UID
        study_uid = (
            row.get("Study Instance UID") or 
            row.get("StudyInstanceUID")
        )
        
        series_desc = row.get("Series Description", "")
        
        if pd.isna(patient_id) or pd.isna(study_uid) or pd.isna(series_desc):
            continue
        
        # Convert to strings
        patient_id = str(patient_id)
        study_uid = str(study_uid)
        series_desc = str(series_desc)
        
        # Check if patient is in our class mapping
        if patient_id not in patient_class_map:
            unmatched_patients.add(patient_id)
            continue
        
        matched_patients += 1
        
        # Track patient's studies
        patient_study_map[patient_id].add(study_uid)
        
        # Check if this series is one of our target sequences
        for target_seq in TARGET_SEQUENCES:
            if target_seq in series_desc:
                study_has_target_seq.add(study_uid)
                break
    
    # Second pass: collect studies with target sequences, organized by class
    print("\nSecond pass: Organizing studies by class...")
    for patient_id, study_uids in patient_study_map.items():
        class_num = patient_class_map[patient_id]
        
        for study_uid in study_uids:
            # Only include studies that have at least one target sequence
            if study_uid in study_has_target_seq:
                study_by_class[class_num].add(study_uid)
    
    print(f"\nMatching Statistics:")
    print(f"  Total manifest rows: {total_rows}")
    print(f"  Matched patient rows: {matched_patients}")
    print(f"  Unique unmatched patients: {len(unmatched_patients)}")
    print(f"  Studies with target sequences: {len(study_has_target_seq)}")
    
    # Generate TCIA files for each class
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"Generating TCIA files for Study Instance UIDs")
    print(f"  Output Directory: {output_path}")
    print("="*80)
    
    total_studies = 0
    for class_num in sorted(study_by_class.keys()):
        study_list = sorted(list(study_by_class[class_num]))
        
        if not study_list:
            continue
        
        # Create TCIA file content
        tcia_filename = output_path / f"class{class_num}.tcia"
        
        tcia_content = [
            "downloadServerUrl=https://nbia.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet",
            "includeAnnotation=true",
            "noOfrRetry=4",
            f"databasketId=manifest-study-class{class_num}.tcia",
            "manifestVersion=3.0",
            "ListOfSeriesToDownload=",
        ]
        
        # Add all study UIDs
        tcia_content.extend(study_list)
        
        # Write TCIA file
        with open(tcia_filename, "w") as f:
            f.write("\n".join(tcia_content))
        
        print(f"  ✓ Class {class_num}: {len(study_list)} studies → {tcia_filename}")
        total_studies += len(study_list)
    
    print(f"\n  Total studies across all classes: {total_studies}")


def main():
    """Main execution function."""
    
    print("="*80)
    print("TCIA Study Manifest Generator by Class")
    print("="*80)
    print("\nTarget Sequences:")
    for seq in TARGET_SEQUENCES:
        print(f"  - {seq}")
    
    # Configuration
    manifest_file = "data/raw/Prostate-MRI-US-Biopsy-NBIA-manifest_v2_20231020-nbia-digest.xlsx"
    parquet_dir = "data/splitted_images"
    output_dir = "data/tcia/study"
    
    # Load patient-class mapping from parquet files
    patient_class_map = load_patient_class_mapping(parquet_dir)
    
    if not patient_class_map:
        print("❌ Error: No patient-class mappings found!")
        return
    
    # Load manifest data
    manifest_df = load_manifest_data(manifest_file)
    
    # Generate TCIA files
    generate_study_tcia_files(manifest_df, patient_class_map, output_dir)
    
    print("\n" + "="*80)
    print("✓ TCIA study manifest generation complete!")
    print("="*80)
    print(f"\nOutput: {output_dir}/class{{1,2,3,4}}.tcia")
    print("\nThese TCIA files contain Study Instance UIDs, which will download")
    print("all sequences for each study, not just the target sequences.")


if __name__ == "__main__":
    main()

