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
│   ├── nbia/                   # Downloaded DICOM files (T2)
│   │   └── class{1,2,3,4}/
│   ├── nbia_ep2d_adc/          # Downloaded DICOM files (ADC)
│   │   └── class{1,2,3,4}/
│   ├── nbia_ep2d_calc/         # Downloaded DICOM files (CALC_BVAL)
│   │   └── class{1,2,3,4}/
│   ├── processed/              # Converted per-slice PNG images (T2)
│   │   └── class{1,2,3,4}/case_XXXX/{series_uid}/images/
│   ├── processed_ep2d_adc/     # Converted per-slice PNG images (ADC)
│   │   └── class{1,2,3,4}/case_XXXX/{series_uid}/images/
│   ├── processed_ep2d_calc/    # Converted per-slice PNG images (CALC_BVAL)
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
├── tools/                      # Data Processing Tools (Organized by Category)
│   ├── preprocessing/          # Data conversion and processing
│   │   ├── convert_xlsx2parquet.py       # Excel → Parquet converter
│   │   ├── merge_datasets.py             # Merge multi-source data
│   │   ├── dicom_converter.py            # DICOM → PNG converter
│   │   ├── process_overlay_aligned.py    # STL → PNG masks (DICOM-aligned)
│   │   ├── visualize_overlay_masks.py    # Visualize masks on images
│   │   └── README_OVERLAY_PROCESSING.md  # Preprocessing documentation
│   ├── dataset/                # PyTorch dataset loaders
│   │   ├── dataset_2d5_multiclass.py     # Multi-class 2.5D dataset
│   │   ├── dataset_2d5_with_seg.py       # 2.5D with segmentation
│   │   ├── dataset_2d5.py                # Basic 2.5D dataset
│   │   ├── transforms_2d5.py             # Data augmentation
│   │   └── README_2D5_PIPELINE.md        # Dataset documentation
│   ├── deployment/             # Cloud deployment utilities
│   │   ├── data_backup.py                # Create data backup
│   │   ├── data_restore.py               # Restore data backup
│   │   └── README.md                     # Deployment documentation
│   ├── validation/             # Testing and validation scripts
│   │   ├── validate_2d5_setup.py         # Validate training setup
│   │   ├── validate_all_masks.py         # Validate mask integrity
│   │   ├── test_dataset_basic.py         # Test dataset loaders
│   │   ├── analyze_data.py               # Data distribution analysis
│   │   └── README.md                     # Validation documentation
│   └── README.md               # Tools overview and quick reference
├── checkpoints/                # Trained models
└── requirements.txt
```

TCIA manifest generation has moved to `../tcia-handler/tools/tcia`.

Expected repo layout:
```
directory/
├── mri/
└── tcia-handler/
```
If your layout differs, set `TCIA_TOOLS_DIR` (direct path to `tools/tcia`) or
`TCIA_HANDLER_ROOT` (path to the repo root) before running `service/preprocess.py`.

## 🚀 Quick Start (3 Commands)

### 1. Preprocess Data
```bash
conda activate mri
python service/preprocess.py
```
Runs all data conversion except TCIA manifest generation (see `../tcia-handler`).

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

## ☁️ Cloud Deployment

Deploy training pipeline on cloud machines (AWS, GCP, Azure):

```bash
# 1. Local: Create data backup
python tools/deployment/data_backup.py

# 2. Cloud: Clone repo and setup
git clone <repo-url> && cd <repo>
conda create -n mri python=3.12 -y && conda activate mri
pip install -r requirements.txt

# 3. Transfer and restore data
scp backup.zip user@cloud:/path/to/repo/
python tools/deployment/data_restore.py backup.zip

# 4. Start training
python service/train.py --config config.yaml
```

📖 **See [docs/CLOUD_DEPLOYMENT.md](docs/CLOUD_DEPLOYMENT.md) for detailed guide**

---

## 📋 Detailed Workflow

### Step 1: Convert Excel to Parquet

Convert raw Excel data with PIRADS scores into class-partitioned parquet files.

```bash
python tools/preprocessing/convert_xlsx2parquet.py
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
python tools/preprocessing/merge_datasets.py
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

Run these from the `mri/` repo root so outputs land in `mri/data/tcia/`.

**Option A: By Series (T2, ADC, CALC_BVAL separately)**
```bash
python ../tcia-handler/tools/tcia/generate_tcia_by_class.py
```
**Output:** `data/tcia/{t2,ep2d_adc,ep2d_calc}/class{1-4}.tcia`

**Option B: By Study (Download all sequences)**
```bash
python ../tcia-handler/tools/tcia/generate_tcia_by_study.py
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
4. Download ep2d_adc manifests to `data/nbia_ep2d_adc/`:
   - `data/tcia/ep2d_adc/class1.tcia` → `data/nbia_ep2d_adc/class1/`
   - `data/tcia/ep2d_adc/class2.tcia` → `data/nbia_ep2d_adc/class2/`
   - `data/tcia/ep2d_adc/class3.tcia` → `data/nbia_ep2d_adc/class3/`
   - `data/tcia/ep2d_adc/class4.tcia` → `data/nbia_ep2d_adc/class4/`
5. Download ep2d_calc manifests to `data/nbia_ep2d_calc/`:
   - `data/tcia/ep2d_calc/class1.tcia` → `data/nbia_ep2d_calc/class1/`
   - `data/tcia/ep2d_calc/class2.tcia` → `data/nbia_ep2d_calc/class2/`
   - `data/tcia/ep2d_calc/class3.tcia` → `data/nbia_ep2d_calc/class3/`
   - `data/tcia/ep2d_calc/class4.tcia` → `data/nbia_ep2d_calc/class4/`

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
python tools/preprocessing/dicom_converter.py --all
```
Converts T2, ep2d_adc, and ep2d_calc sequences by default and writes to the
corresponding `data/processed*` directories.

**Process Single Class:**
```bash
python tools/preprocessing/dicom_converter.py --class 1
```
To process a single sequence, pass `--input` and `--output` for that sequence.

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

data/processed_ep2d_adc/
  class1/
    case_0001/
      {SeriesInstanceUID}/
        ...
  manifest_all.csv

data/processed_ep2d_calc/
  class1/
    case_0001/
      {SeriesInstanceUID}/
        ...
  manifest_all.csv
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
python tools/preprocessing/process_overlay_aligned.py
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
python tools/preprocessing/visualize_overlay_masks.py
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

All documentation is organized in the [`docs/`](docs/) directory:

### Essential Documentation
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Essential commands and workflows
- **[Cloud Deployment](docs/CLOUD_DEPLOYMENT.md)** - Deploy to AWS, GCP, Azure
- **[Training Enhanced](docs/TRAINING_ENHANCED.md)** - Advanced training features
- **[Backup & Restore](docs/BACKUP_RESTORE_GUIDE.md)** - Data backup system

### Tool Documentation
- **[Tools Overview](tools/README.md)** - All tools organized by category
- **[Preprocessing](tools/preprocessing/)** - Data conversion tools
- **[Dataset Loaders](tools/dataset/)** - PyTorch dataset implementations
- **[Deployment](tools/deployment/)** - Cloud deployment utilities
- **[Validation](tools/validation/)** - Testing and validation scripts

### Technical Guides
- **[Complete Training Guide](docs/guides/FINAL_TRAINING_SETUP.md)** - Step-by-step training
- **[Mask Alignment](docs/guides/DICOM_ALIGNED_PROCESSING.md)** - How alignment works
- **[Mask Detection](docs/guides/MASK_DETECTION_SOLUTION.md)** - Mask loading system
- **[Experiment Tracking](service/README_AIM.md)** - Using Aim for MLOps

**📖 See [docs/README.md](docs/README.md) for complete documentation index**

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
    ├→ [tcia-handler: generate_tcia_by_class.py / generate_tcia_by_study.py]
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
