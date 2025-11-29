# TCIA Manifest Generator by Sequence Type

## Overview

This script generates TCIA manifest files organized by class for different MRI sequence types. It reads the NBIA manifest Excel file, matches patients with their PIRADS class, and creates separate TCIA files for each sequence type and class.

## Sequence Types

The script processes three MRI sequence types:

1. **T2** (`t2_spc_rst_axial obl_Prostate`)
   - Output: `data/tcia/t2/class{1,2,3,4}.tcia`

2. **ADC** (`ep2d-advdiff-3Scan-4bval_spair_std_ADC`)
   - Output: `data/tcia/ep2d_adc/class{1,2,3,4}.tcia`

3. **CALC_BVAL** (`ep2d-advdiff-3Scan-4bval_spair_std_CALC_BVAL`)
   - Output: `data/tcia/ep2d_calc/class{1,2,3,4}.tcia`

## Class Mapping

- **Class 1**: PIRADS 0, 1, 2 (Low risk)
- **Class 2**: PIRADS 3 (Intermediate risk)
- **Class 3**: PIRADS 4 (High risk)
- **Class 4**: PIRADS 5 (Very high risk)

## Requirements

```bash
pip install pandas openpyxl pyarrow
```

## Usage

```bash
conda activate mri
python tools/generate_tcia_by_class.py
```

## Input Files

1. **Patient-Class Mapping**: `data/splitted_images/class={1,2,3,4}/*.parquet`
   - Contains patient IDs and their PIRADS class assignments

2. **NBIA Manifest**: `data/raw/Prostate-MRI-US-Biopsy-NBIA-manifest_v2_20231020-nbia-digest.xlsx`
   - Contains all available series with patient IDs and series descriptions

## Output Structure

```
data/tcia/
├── t2/
│   ├── class1.tcia
│   ├── class2.tcia
│   ├── class3.tcia
│   └── class4.tcia
├── ep2d_adc/
│   ├── class1.tcia
│   ├── class2.tcia
│   ├── class3.tcia
│   └── class4.tcia
└── ep2d_calc/
    ├── class1.tcia
    ├── class2.tcia
    ├── class3.tcia
    └── class4.tcia
```

## What the Script Does

1. **Load Patient-Class Mapping**: Reads parquet files to map each patient to their PIRADS class
2. **Load Manifest Data**: Reads the NBIA manifest Excel file
3. **Match and Filter**: For each row in the manifest:
   - Matches the patient ID with the class mapping
   - Filters by series description (T2, ADC, or CALC_BVAL)
   - Groups series UIDs by class and sequence type
4. **Generate TCIA Files**: Creates separate TCIA manifest files for each sequence type and class

## Next Steps

After generating the TCIA files:

1. Use NBIA Data Retriever to download DICOM files:
   ```bash
   # For T2 sequences
   open data/tcia/t2/class1.tcia  # Opens NBIA Data Retriever
   
   # For ADC sequences
   open data/tcia/ep2d_adc/class1.tcia
   
   # For CALC_BVAL sequences
   open data/tcia/ep2d_calc/class1.tcia
   ```

2. Download to corresponding directories:
   - `data/nbia/t2/class{1,2,3,4}/`
   - `data/nbia/ep2d_adc/class{1,2,3,4}/`
   - `data/nbia/ep2d_calc/class{1,2,3,4}/`

3. Convert DICOM to images using `dicom_converter.py`

## Troubleshooting

### No series found for a sequence type
- Check if the series description exactly matches in the manifest file
- Use the diagnostic output to see available series descriptions

### Patients not matching
- Verify patient ID column names in parquet files
- Check if patient IDs are formatted consistently (the script converts to strings)
- Review the "Unique unmatched patients" count in the output

### Missing columns
- The script tries multiple column name variations
- Check the column list in the diagnostic output

