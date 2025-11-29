# MRI 2.5D Segmentation Pipeline

End-to-end pipeline for training multi-class segmentation models on prostate MRI data from TCIA.



> **⚠️ Work in Progress**  
> This project is currently under active development. If you're interested in using this pipeline for your research or have questions about the implementation, please contact the author.


## 📁 Project Structure

```
mri/
├── data/
│   ├── raw/                    # Raw Excel files
│   │   ├── selected_patients_3.xlsx
│   │   └── Prostate-MRI-US-Biopsy-NBIA-manifest_v2_20231020-nbia-digest.xlsx
│   ├── splitted_images/        # Image-only records (197 rows)
│   │   └── class={1,2,3,4}/   # PIRADS-based classes
│   ├── splitted_info/          # Enriched records with targets & biopsies (10,881 rows)
│   │   └── class={1,2,3,4}/
│   ├── tcia/                   # TCIA manifest files
│   │   ├── t2/, ep2d_adc/, ep2d_calc/  # By sequence type
│   │   └── study/             # Full study downloads
│   ├── overlay/                # 3D Slicer biopsy annotations
│   │   └── Biopsy Overlays (3D Slicer)/
│   ├── nbia/                   # Downloaded DICOM files
│   │   └── class{1,2,3,4}/
│   ├── processed/              # Converted per-slice PNG images
│   │   └── class{1,2,3,4}/case_XXXX/{series_uid}/images/
│   ├── processed_seg/          # Segmentation masks (aligned)
│   │   └── class{1,2,3,4}/case_XXXX/{series_uid}/{structure}/
│   └── visualizations/         # Mask overlays on images
│       └── class{1,2,3,4}/case_XXXX/
├── service/                    # ML Pipeline Scripts
│   ├── preprocess.py          # Data preprocessing orchestration
│   ├── validate_data.py       # Visual data validation
│   ├── train.py               # Multi-class 2.5D training
│   ├── test.py                # Model evaluation
│   ├── inference.py           # Run segmentation
│   └── demo.py                # Full pipeline demo
├── tools/                      # Data Processing Tools
│   ├── convert_xlsx2parquet.py       # Excel → Parquet converter
│   ├── merge_datasets.py             # Merge multi-source data
│   ├── tcia_generator.py             # Generate TCIA manifests (by series)
│   ├── generate_tcia_by_class.py     # Generate TCIA by sequence type
│   ├── generate_tcia_by_study.py     # Generate TCIA by study (full download)
│   ├── dicom_converter.py            # DICOM → PNG converter
│   ├── process_overlay_aligned.py    # STL → PNG masks (DICOM-aligned)
│   ├── visualize_overlay_masks.py    # Visualize masks on images
│   └── dataset_2d5_multiclass.py     # Multi-class dataset loader
├── checkpoints/                # Trained models
└── requirements.txt
```

## 🚀 Quick Start (3 Commands)

### 1. Preprocess Data
```bash
conda activate mri
python service/preprocess.py
```
Runs all data conversion: Excel→Parquet, TCIA manifests, DICOM→PNG, STL→Masks

### 2. Validate Data
```bash
python service/validate_data.py
```
Checks data integrity and creates visualizations with color-coded masks

### 3. Train Model
```bash
# Option 1: Use config file (recommended)
python service/train.py --config config.yaml

# Option 2: CLI arguments
python service/train.py --manifest data/processed/class2/manifest.csv --epochs 50
```
Trains multi-class segmentation (Prostate + Target1 + Target2 together)

---

## 📋 Detailed Workflow

### Step 1: Convert Excel to Parquet

Convert raw Excel data with PIRADS scores into class-partitioned parquet files.

```bash
python tools/convert_xlsx2parquet.py
```

**Input:** `data/raw/selected_patients_3.xlsx` - MRI image metadata (197 records)

**Output:** Class-partitioned parquet files in `data/splitted_images/class={1,2,3,4}/`

**PIRADS Classification:**
- **Class 1:** PIRADS 0, 1, 2 (combined) - Low risk
- **Class 2:** PIRADS 3 - Intermediate risk
- **Class 3:** PIRADS 4 - High risk
- **Class 4:** PIRADS 5 - Very high risk

---

### Step 1b: Merge Multi-Source Data (Optional)

Enrich patient records by merging three data sources: image metadata, target lesions, and biopsy results.

```bash
python tools/merge_datasets.py
```

**Input Sources:**
- `data/raw/selected_patients_3.xlsx` - MRI image metadata (197 records)
- `data/raw/Target-Data_2019-12-05-2.xlsx` - Target lesion data (1,617 targets from 840 patients)
- `data/raw/TCIA-Biopsy-Data_2020-07-14.xlsx` - Biopsy core data (24,783 cores from 1,150 patients)

**Output:** Enriched dataset in `data/splitted_info/` with:
- **10,881 total rows** (55.23x multiplication from 197 original)
- **48 total columns** (17 original + 5 target + 24 biopsy + 2 source tracking)
- Multiple rows per patient due to one-to-many relationships
- All 197 patients preserved with full data coverage

**Key Features:**
- Left join preserves all image records
- Handles multiple targets per patient
- Handles multiple biopsy cores per patient
- Prefixed columns (`target_*`, `biopsy_*`) to avoid conflicts

---

### Step 2: Generate TCIA Manifests

Create `.tcia` manifest files for downloading DICOM data.

**Option A: By Series (T2, ADC, CALC_BVAL separately)**
```bash
python tools/generate_tcia_by_class.py
```
**Output:** `data/tcia/{t2,ep2d_adc,ep2d_calc}/class{1-4}.tcia`

**Option B: By Study (Download all sequences)**
```bash
python tools/generate_tcia_by_study.py
```
**Output:** `data/tcia/study/class{1-4}.tcia`

---

### Step 3: Download DICOM Files (Manual)

Use the NBIA Data Retriever to download DICOM files from TCIA.

1. Install [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
2. Open each `.tcia` manifest file from `data/tcia/`
3. Download to corresponding class directory in `data/nbia/`:
   - `class1.tcia` → `data/nbia/class1/`
   - `class2.tcia` → `data/nbia/class2/`
   - `class3.tcia` → `data/nbia/class3/`
   - `class4.tcia` → `data/nbia/class4/`

**Expected directory structure after download:**
```
data/nbia/class1/
  manifest-class1-.../
    Prostate-MRI-US-Biopsy/
      Prostate-MRI-US-Biopsy-0001/
        1.3.6.1.../
          1.2.840.../
            1-01.dcm
            1-02.dcm
            ...
```

---

### Step 4: Convert DICOM to PNG Images

Convert DICOM series to per-slice PNG images.

**Process All Classes (Recommended):**
```bash
python tools/dicom_converter.py --all
```

**Process Single Class:**
```bash
python tools/dicom_converter.py --class 1
```

**Output Structure:**
```
data/processed/
  class1/
    case_0001/
      {SeriesInstanceUID}/
        meta.json              # Series metadata
        images/                # Per-slice PNG images
          0000.png
          0001.png
          ...
    manifest.csv              # Per-class manifest
  class2/
    ...
  manifest_all.csv            # Combined manifest for all classes
```

**Manifest CSV Columns:**
- `case_id`, `series_uid`, `slice_idx`
- `image_path`, `mask_path` (empty if no labels)
- `num_labels`, `spacing_x`, `spacing_y`, `spacing_z`
- `modality`, `study_date`, `manufacturer`
- `class` (in combined manifest)

---

### Step 5: Process Overlay Segmentations

Convert 3D Slicer STL mesh segmentations to aligned PNG masks.

```bash
python tools/process_overlay_aligned.py
```

**Requirements:** Original DICOM files in `data/nbia/`

**Output Structure:**
```
data/processed_seg/
  class1/
    case_0001/
      {SeriesInstanceUID}/
        prostate/
          0000.png
          0001.png
          ...
        target1/
          0000.png
          ...
        target2/
          0000.png
          ...
        scene.mrml
      biopsies.json            # Biopsy coordinates + pathology
```

**Key Features:**
- Uses DICOM geometry for proper alignment
- Transforms meshes from physical space to image space
- Masks exactly match image dimensions (256×256 or 512×512)

---

### Step 6: Visualize Segmentations

Create overlay visualizations to verify mask alignment.

```bash
python tools/visualize_overlay_masks.py
```

**Output:** `data/visualizations/class{N}/case_XXXX/slice_NNNN.png`
- 3-panel images: Original | Overlay | Masks
- Color-coded: 🟡 Prostate, 🔴 Target1, 🟠 Target2
- Samples 10 slices per series

---

### Quick Alternative: Use Preprocessor

Run all preprocessing steps at once:

```bash
python service/preprocess.py
```

This orchestrates all tools in sequence (Excel→Parquet, TCIA manifests, DICOM→PNG, STL→Masks)

---

### Training & Inference

**Validate Data First:**
```bash
python service/validate_data.py
```
✓ Creates `validation_results/` with visualizations  
✓ Shows mask statistics per class  
✓ Color legend: 🟡 Prostate, 🔴 Target1, 🟠 Target2

**Train Multi-Class Model:**
```bash
# Using config file (recommended)
python service/train.py --config config.yaml

# Or with CLI args
python service/train.py \
    --manifest data/processed/class2/manifest.csv \
    --batch_size 8 \
    --epochs 50 \
    --scheduler onecycle
```
✓ Trains on all 3 classes simultaneously  
✓ Advanced LR schedulers + validation visualizations  
✓ Saves checkpoints to `checkpoints/`  
✓ Logs to Aim for experiment tracking

**Evaluate Model:**
```bash
python service/test.py \
    --checkpoint checkpoints/model_best.pt \
    --test_manifest data/processed/class2/manifest.csv
```

**Run Inference:**
```bash
python service/inference.py \
    --checkpoint checkpoints/model_best.pt \
    --input_dir data/processed/class2/case_0001/{series_uid}/images/
```

---

## 📦 Installation

```bash
conda create -n mri python=3.12 -y
conda activate mri
pip install -r requirements.txt
```

## 🎯 Model Architecture

**Input:** 2.5D stacks (5 adjacent slices)  
**Output:** 3-channel segmentation (prostate, target1, target2)  
**Network:** U-Net with encoder-decoder + skip connections

```python
Input:  (batch, 5, 256, 256)   # 5 stacked slices
Output: (batch, 3, 256, 256)   # 3 segmentation masks
```

---

## 📊 Dataset Statistics

| Dataset | Count | Description |
|---------|-------|-------------|
| **MRI Series** | 197 | T2, ADC, CALC_BVAL sequences |
| **PIRADS Classes** | 4 | Class 1 (17), Class 2 (60), Class 3 (60), Class 4 (60) |
| **Image Slices** | ~8,000 | Per-slice PNG images (256×256 or 512×512, auto-resized) |
| **Segmentation Cases** | ~135 | With aligned prostate & lesion masks |
| **Training Samples** | 1,845 | Valid 2.5D stacks (class2) |

**Data Sources:**
- `splitted_images/` - 197 MRI series for manifest generation
- `splitted_info/` - 10,881 enriched rows with targets & biopsies
- `processed/` - ~8,000 PNG image slices
- `processed_seg/` - ~135 cases with segmentation masks
- `overlay/` - 3,041 cases with biopsy annotations (STL meshes)

**Segmentation Masks:**
- Class 1: 17 cases, 566 prostate masks, 119 target1, 34 target2
- Class 2: 58 cases, 1,852 prostate masks, 322 target1, 86 target2 ⭐
- Class 3: 60 cases with multiple structures
- Class 4: Available

---

## 🔧 Advanced Options

**Custom Image Size:**
```python
# In service/train.py, modify dataset creation:
dataset = MRI25DMultiClassDataset(..., target_size=(512, 512))
```

**Configuration Management:**

All training parameters can be configured via YAML (see `config.yaml`):
- Learning rate schedulers (OneCycle, Cosine, ReduceLROnPlateau, etc.)
- Validation visualizations
- Experiment tracking with Aim

```bash
# Edit config.yaml, then run:
python service/train.py --config config.yaml

# Override specific params:
python service/train.py --config config.yaml --epochs 100 --batch_size 16

# Resume training:
python service/train.py --config config.yaml --resume checkpoints/model_epoch_25.pt
```

---

## 📚 Documentation

- **`FINAL_TRAINING_SETUP.md`** - Complete training guide
- **`MASK_DETECTION_SOLUTION.md`** - How masks are loaded
- **`DICOM_ALIGNED_PROCESSING.md`** - Mask alignment details
- **`service/README.md`** - Service layer documentation

---

## 🔗 Resources

- [TCIA Prostate-MRI-US-Biopsy](https://www.cancerimagingarchive.net/collection/prostate-mri-us-biopsy/)
- [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)

---

## 📝 Pipeline Summary

```
Excel Data (3 sources)
    ↓ [convert_xlsx2parquet.py]
Image-Only Records (data/splitted_images/) - 197 series
    ↓ [merge_datasets.py]
Enriched Records (data/splitted_info/) - 10,881 rows with targets & biopsies
    │
    ├→ [generate_tcia_by_class.py / generate_tcia_by_study.py]
    │  TCIA Manifests (.tcia files)
    │      ↓ [NBIA Data Retriever - Manual]
    │  DICOM Files (data/nbia/)
    │      ↓ [dicom_converter.py]
    │  Per-Slice Images (data/processed/)
    │
    └→ [process_overlay_aligned.py]
       3D Slicer Overlays → DICOM-Aligned Masks (data/processed_seg/)
           ↓ [visualize_overlay_masks.py]
       Visualizations (data/visualizations/)
           ↓
       [train.py] - 2.5D Multi-Class Segmentation Training
           ↓
       Trained Model (checkpoints/)
```

**Complete workflow:**
1. `preprocess.py` - Run all data conversion steps
2. `validate_data.py` - Visual data validation
3. `train.py` - Train multi-class segmentation model
4. `test.py` - Evaluate model performance
5. `inference.py` - Segment new images
