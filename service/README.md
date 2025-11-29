# Service Layer - MRI Segmentation Pipeline

This directory contains scripts to orchestrate the complete machine learning pipeline for 2.5D MRI segmentation.

## 📁 Directory Structure

```
service/
├── README.md              # This file
├── __init__.py            # Package marker
├── preprocess.py          # Data preprocessing orchestration
├── train.py               # Model training (basic - uses manifest masks)
├── train_with_seg.py      # Model training (NEW - uses processed_seg/)
├── test.py                # Model testing and evaluation
├── inference.py           # Run segmentation on new data
├── demo.py                # Full pipeline demo
└── validate_data.py       # Visual data validation
```

---

## 🚀 Quick Start

### 1. Preprocess Data
```bash
# Run all preprocessing steps
python service/preprocess.py

# Or run specific steps
python service/preprocess.py --step convert_xlsx
python service/preprocess.py --step merge_datasets
python service/preprocess.py --step generate_tcia
python service/preprocess.py --step dicom_convert
python service/preprocess.py --step process_overlays
python service/preprocess.py --step validate
```

### 2. Validate Data
```bash
# Check data integrity and visualize
python service/validate_data.py

# Output: validation_results/ with visualizations
```

### 3. Train Model
```bash
python service/train.py \
    --manifest data/processed/class2/manifest.csv \
    --epochs 50
```

### 4. Test Model
```bash
python service/test.py \
    --checkpoint checkpoints_seg/model_best.pt \
    --test_manifest data/processed/class2/manifest.csv
```

### 5. Run Inference
```bash
python service/inference.py \
    --checkpoint checkpoints_seg/model_best.pt \
    --input_dir data/processed/class2/case_0001/ \
    --output_dir predictions/
```

### 6. Full Demo
```bash
# Run entire pipeline automatically
python service/demo.py
```

---

## 📋 Script Details

### preprocess.py
**Purpose:** Orchestrates all data conversion tools

**What it does:**
1. Converts Excel files to Parquet
2. Merges datasets across classes
3. Generates TCIA manifest files
4. Converts DICOM to PNG images
5. Processes overlay STL files to masks
6. Validates 2.5D data setup

**Usage:**
```bash
# Run all steps
python service/preprocess.py

# Run specific step
python service/preprocess.py --step dicom_convert

# Skip specific step
python service/preprocess.py --skip process_overlays

# Dry run (show what would be done)
python service/preprocess.py --dry-run
```

**Output:**
- `data/splitted_images/` - Parquet files
- `data/tcia/` - TCIA manifest files
- `data/processed/` - PNG images and manifests
- `data/processed_seg/` - Segmentation masks

---

### validate_data.py
**Purpose:** Visual validation of converted data

**What it does:**
1. Checks manifest integrity
2. Validates image files
3. Tests 2.5D stacking
4. Detects masks in `processed_seg/`
5. Creates visualizations

**Usage:**
```bash
python service/validate_data.py
```

**Output:**
```
validation_results/
├── class1/
│   ├── single_slices.png        # Sample slices
│   ├── 2d5_stack_example.png    # 2.5D stack visualization
│   ├── 2d5_stack_explained.png  # Explanation diagram
│   ├── data_distribution.png    # Statistics
│   └── masks_overlay.png        # Masks overlay (if available)
├── class2/
│   └── ...
└── class3/
    └── ...
```

**Key Features:**
- ✓ Validates manifest structure
- ✓ Checks image file integrity
- ✓ Tests 2.5D compatibility
- ✓ Detects masks in `processed_seg/`
- ✓ Creates visualization grids
- ✓ Shows data distribution statistics

---

### train_with_seg.py (RECOMMENDED)
**Purpose:** Train 2.5D segmentation model with masks from `processed_seg/`

**What it does:**
1. Loads images from `data/processed/`
2. Loads masks from `data/processed_seg/`
3. Creates 2.5D stacks
4. Trains U-Net model
5. Saves checkpoints

**Usage:**
```bash
# Basic usage
python service/train_with_seg.py \
    --manifest data/processed/class2/manifest.csv \
    --mask_type prostate

# Advanced usage
python service/train_with_seg.py \
    --manifest data/processed/class2/manifest.csv \
    --mask_type prostate \
    --batch_size 16 \
    --epochs 100 \
    --lr 5e-5 \
    --loss dice_bce \
    --stack_depth 7 \
    --output_dir checkpoints_prostate

# Resume training
python service/train_with_seg.py \
    --manifest data/processed/class2/manifest.csv \
    --mask_type prostate \
    --resume checkpoints_prostate/model_epoch_50.pt
```

**Arguments:**
- `--manifest`: Path to manifest CSV
- `--mask_type`: Which mask to use (`prostate`, `target1`, `target2`)
- `--stack_depth`: Number of slices to stack (default: 5, must be odd)
- `--batch_size`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--loss`: Loss function (`dice`, `bce`, `dice_bce`)
- `--output_dir`: Checkpoint directory (default: `checkpoints_seg`)
- `--resume`: Resume from checkpoint

**Output:**
```
checkpoints_seg/
├── model_epoch_5.pt
├── model_epoch_10.pt
├── model_best.pt          # Best model by validation Dice
└── ...
```

**Available Masks:**
- **prostate**: Prostate gland segmentation (largest dataset: 1,852 slices in class2)
- **target1**: Primary lesion segmentation
- **target2**: Secondary lesion segmentation

---

### train.py (BASIC)
**Purpose:** Train 2.5D segmentation model (original version)

**Note:** This script uses masks from the `mask_path` column in manifest CSV. Use `train_with_seg.py` instead if you want to load masks from `data/processed_seg/`.

**Usage:**
```bash
python service/train.py \
    --manifest data/processed/class2/manifest.csv \
    --model smp_resunet \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4
```

---

### test.py
**Purpose:** Evaluate trained model on test set

**Usage:**
```bash
python service/test.py \
    --checkpoint checkpoints_seg/model_best.pt \
    --test_manifest data/processed/class2/manifest.csv \
    --output_dir test_results/
```

**Output:**
- Dice scores per case
- IoU scores
- Confusion matrix
- Visualizations of predictions

---

### inference.py
**Purpose:** Run segmentation on new MRI data

**Usage:**
```bash
# Inference on a single case
python service/inference.py \
    --checkpoint checkpoints_seg/model_best.pt \
    --input_dir data/processed/class2/case_0001/{series_uid}/images/ \
    --output_dir predictions/case_0001/

# Batch inference
python service/inference.py \
    --checkpoint checkpoints_seg/model_best.pt \
    --input_csv inference_list.csv \
    --output_dir predictions/
```

**Output:**
```
predictions/
└── case_0001/
    ├── 0000_pred.png
    ├── 0001_pred.png
    ├── ...
    └── overlay/
        ├── 0000_overlay.png  # Image with mask overlay
        └── ...
```

---

### demo.py
**Purpose:** Run entire pipeline end-to-end with no arguments

**Usage:**
```bash
python service/demo.py
```

**What it does:**
1. Validates existing data
2. Checks for processed images
3. Creates 2.5D datasets
4. Trains a model (2 epochs as demo)
5. Runs inference on validation set
6. Generates visualizations

**Output:**
- `demo_output/` - All results
- `demo_checkpoints/` - Model checkpoints
- `demo_visualizations/` - Prediction visualizations

---

## 🎯 Typical Workflow

### First Time Setup
```bash
# 1. Preprocess all data
python service/preprocess.py

# 2. Validate data
python service/validate_data.py

# 3. Check validation results
ls -lh validation_results/
```

### Training
```bash
# 4. Train model (RECOMMENDED: use train_with_seg.py)
python service/train_with_seg.py \
    --manifest data/processed/class2/manifest.csv \
    --mask_type prostate \
    --batch_size 8 \
    --epochs 50
```

### Evaluation
```bash
# 5. Test model
python service/test.py \
    --checkpoint checkpoints_seg/model_best.pt \
    --test_manifest data/processed/class2/manifest.csv
```

### Inference
```bash
# 6. Run inference
python service/inference.py \
    --checkpoint checkpoints_seg/model_best.pt \
    --input_dir data/processed/class2/case_0001/{series_uid}/images/ \
    --output_dir predictions/
```

---

## 📊 Data Structure

### Input (from preprocessing)
```
data/
├── processed/              # Images
│   └── class2/
│       ├── manifest.csv
│       └── case_0001/
│           └── {series_uid}/
│               └── images/
│                   ├── 0000.png
│                   └── ...
│
└── processed_seg/          # Masks
    └── class2/
        └── case_0001/
            └── {series_uid}/
                ├── prostate/
                │   ├── 0000.png
                │   └── ...
                └── target1/
                    └── ...
```

### Output (from training)
```
checkpoints_seg/
├── model_epoch_5.pt
├── model_epoch_10.pt
├── model_best.pt
└── ...
```

### Output (from inference)
```
predictions/
└── case_0001/
    ├── 0000_pred.png        # Predicted mask
    ├── 0001_pred.png
    └── overlay/
        ├── 0000_overlay.png  # Image with mask
        └── ...
```

---

## 🔧 Configuration

### Preprocessing
Edit variables in `preprocess.py`:
```python
RAW_DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
OVERLAY_DIR = "data/overlay"
```

### Training
Common configurations:

**Fast training (prototyping):**
```bash
python service/train_with_seg.py \
    --manifest data/processed/class2/manifest.csv \
    --mask_type prostate \
    --batch_size 4 \
    --epochs 10 \
    --stack_depth 3
```

**Production training (best quality):**
```bash
python service/train_with_seg.py \
    --manifest data/processed/class2/manifest.csv \
    --mask_type prostate \
    --batch_size 16 \
    --epochs 100 \
    --lr 5e-5 \
    --stack_depth 7 \
    --loss dice_bce
```

---

## 🚨 Troubleshooting

### "No masks found" during training
**Solution:** Use `train_with_seg.py` instead of `train.py`:
```bash
python service/train_with_seg.py --manifest data/processed/class2/manifest.csv --mask_type prostate
```

### "CUDA out of memory"
**Solution:** Reduce batch size or stack depth:
```bash
python service/train_with_seg.py ... --batch_size 4 --stack_depth 3
```

### "Loss is 0.0000"
**Solution:** Check that masks exist:
```bash
python service/validate_data.py
# Should show: "✓ Found masks in processed_seg/"
```

If no masks found:
```bash
python service/preprocess.py --step process_overlays
```

---

## 📚 Documentation

- **`MASK_DETECTION_SOLUTION.md`** - How masks are loaded from `processed_seg/`
- **`QUICK_START_TRAINING.md`** - Fast training guide
- **`README_2D5.md`** - Technical details on 2.5D approach
- **`OVERLAY_PROCESSING_UPDATE.md`** - How overlay masks are processed
- **`service/README.md`** - This file

---

## ✅ Validation Checklist

Before training, ensure:

- [x] Data preprocessed (`python service/preprocess.py`)
- [x] Manifest CSV exists (`data/processed/class*/manifest.csv`)
- [x] Images exist (`data/processed/class*/case_*/.../*.png`)
- [x] Masks exist (`data/processed_seg/class*/case_*/.../*.png`)
- [x] Validation passed (`python service/validate_data.py`)

---

## 🎯 Summary

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `preprocess.py` | Data conversion | Raw DICOM | PNG images + masks |
| `validate_data.py` | Data validation | Manifest CSV | Visualizations |
| `train_with_seg.py` | Training (NEW) | Images + masks | Model checkpoints |
| `train.py` | Training (basic) | Manifest CSV | Model checkpoints |
| `test.py` | Evaluation | Model + test data | Metrics |
| `inference.py` | Prediction | Model + new images | Masks |
| `demo.py` | Full pipeline | Existing data | End-to-end demo |

**Recommended workflow:**
1. `preprocess.py` → 2. `validate_data.py` → 3. `train_with_seg.py` → 4. `test.py` → 5. `inference.py`
