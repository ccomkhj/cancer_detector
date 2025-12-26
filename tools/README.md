# Tools Directory

Organized collection of utilities for the MRI 2.5D segmentation pipeline.

## 📁 Directory Structure

```
tools/
├── preprocessing/     # Data conversion and processing
├── dataset/          # PyTorch dataset loaders
├── deployment/       # Cloud deployment utilities
└── validation/       # Testing and validation scripts
```

## 🔧 Tool Categories

### [`preprocessing/`](preprocessing/) - Data Conversion & Processing
Convert raw data into training-ready format.

| Script | Purpose |
|--------|---------|
| `convert_xlsx2parquet.py` | Excel → Parquet (PIRADS classification) |
| `merge_datasets.py` | Merge multi-source data (images + targets + biopsies) |
| `dicom_converter.py` | DICOM → PNG images |
| `process_overlay_aligned.py` | STL meshes → Aligned PNG masks |
| `process_overlay_to_masks.py` | Legacy STL processing |
| `visualize_overlay_masks.py` | Create overlay visualizations |

**Quick Start:**
```bash
# Convert all data
python tools/preprocessing/convert_xlsx2parquet.py
python tools/preprocessing/dicom_converter.py --all
python tools/preprocessing/process_overlay_aligned.py
```

---

### TCIA Manifest Generation
TCIA tools moved to `../tcia-handler/tools/tcia`.

**Quick Start (run from `mri/`):**
```bash
# Generate manifests by sequence type
python ../tcia-handler/tools/tcia/generate_tcia_by_class.py

# Or by full study
python ../tcia-handler/tools/tcia/generate_tcia_by_study.py
```

---

### [`dataset/`](dataset/) - PyTorch Dataset Loaders
PyTorch dataset implementations for training.

| Script | Purpose |
|--------|---------|
| `dataset_2d5_multiclass.py` | Multi-class 2.5D dataset (recommended) |
| `dataset_2d5_with_seg.py` | 2.5D with segmentation masks |
| `dataset_2d5.py` | Basic 2.5D dataset |
| `transforms_2d5.py` | Data augmentation transforms |

**Usage:**
```python
from tools.dataset.dataset_2d5_multiclass import MRI25DMultiClassDataset

dataset = MRI25DMultiClassDataset(
    manifest_csv="data/processed/class2/manifest.csv",
    stack_depth=5
)
```

---

### [`deployment/`](deployment/) - Cloud Deployment
Tools for deploying training pipeline to cloud machines.

| Script | Purpose |
|--------|---------|
| `data_backup.py` | Create compressed data backups |
| `data_restore.py` | Restore backups on cloud machines |

**Quick Start:**
```bash
# Create backup (local machine)
python tools/deployment/data_backup.py

# Restore backup (cloud machine)
python tools/deployment/data_restore.py backup.zip
```

**See:** [docs/CLOUD_DEPLOYMENT.md](../docs/CLOUD_DEPLOYMENT.md) for complete guide

---

### [`validation/`](validation/) - Testing & Validation
Scripts for data validation, testing, and diagnostics.

| Script | Purpose |
|--------|---------|
| `validate_2d5_setup.py` | Validate 2.5D training setup |
| `validate_all_masks.py` | Validate mask integrity |
| `test_dataset_basic.py` | Test dataset loaders |
| `test_2d5_models.py` | Test model architectures |
| `analyze_data.py` | Data distribution analysis |
| `diagnose_alignment.py` | Diagnose mask alignment issues |

**Quick Start:**
```bash
# Validate complete setup
python tools/validation/validate_2d5_setup.py

# Check masks
python tools/validation/validate_all_masks.py
```

---

## 🚀 Complete Workflow

### 1. Data Preparation
```bash
# Convert Excel → Parquet
python tools/preprocessing/convert_xlsx2parquet.py

# Generate TCIA manifests
python ../tcia-handler/tools/tcia/generate_tcia_by_class.py

# Download DICOM using NBIA Data Retriever (manual)

# Convert DICOM → PNG
python tools/preprocessing/dicom_converter.py --all

# Process STL → Masks
python tools/preprocessing/process_overlay_aligned.py

# Create visualizations
python tools/preprocessing/visualize_overlay_masks.py
```

### 2. Validation
```bash
# Validate data integrity
python tools/validation/validate_2d5_setup.py
python tools/validation/validate_all_masks.py

# Analyze data distribution
python tools/validation/analyze_data.py
```

### 3. Training (Local)
```bash
# Train using dataset loaders
python service/train.py --config config.yaml
```

### 4. Deployment (Cloud)
```bash
# Create backup
python tools/deployment/data_backup.py

# Transfer to cloud
scp backup.zip user@cloud:/path/

# Restore on cloud
python tools/deployment/data_restore.py backup.zip

# Train on cloud
python service/train.py --config config.yaml
```

---

## 📚 Category READMEs

Each subdirectory contains its own README with detailed documentation:

- **[preprocessing/README.md](preprocessing/README_OVERLAY_PROCESSING.md)** - Data preprocessing details
- **[tcia/README.md](tcia/README_TCIA_GENERATOR.md)** - TCIA manifest generation
- **[dataset/README.md](dataset/README_2D5_PIPELINE.md)** - Dataset loader documentation
- **deployment/** - See [CLOUD_DEPLOYMENT.md](../CLOUD_DEPLOYMENT.md)
- **validation/** - Individual script help with `--help`

---

## 🔍 Quick Reference

### Common Commands

```bash
# Full preprocessing pipeline (alternative)
python service/preprocess.py

# Data validation
python service/validate_data.py

# Create backup
python tools/deployment/data_backup.py

# Test dataset
python tools/validation/test_dataset_basic.py
```

### Import Paths

When importing from tools in your scripts:

```python
# Dataset loaders
from tools.dataset.dataset_2d5_multiclass import MRI25DMultiClassDataset

# Transforms
from tools.dataset.transforms_2d5 import get_transforms
```

---

## 📖 Related Documentation

- **[Main README](../README.md)** - Project overview
- **[Documentation Index](../docs/README.md)** - All documentation
- **[Cloud Deployment](../docs/CLOUD_DEPLOYMENT.md)** - Cloud deployment guide
- **[Training Enhanced](../docs/TRAINING_ENHANCED.md)** - Training features
- **[Quick Reference](../docs/QUICK_REFERENCE.md)** - Command reference

---

## 💡 Tips

1. **Use service scripts first**: `service/preprocess.py` and `service/validate_data.py` run common workflows
2. **Check --help**: All scripts have help text: `python script.py --help`
3. **Read category READMEs**: Each subdirectory has detailed docs
4. **Test incrementally**: Validate after each preprocessing step
