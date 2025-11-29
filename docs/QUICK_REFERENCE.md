# Quick Reference Guide

Essential commands for the MRI 2.5D Segmentation Pipeline.

## 📦 Setup

```bash
# Clone repository
git clone <repo-url>
cd <repo>

# Create environment
conda create -n mri python=3.12 -y
conda activate mri
pip install -r requirements.txt
```

## 🔄 Data Processing

```bash
# Full preprocessing pipeline
python service/preprocess.py

# Or individual steps:
python tools/convert_xlsx2parquet.py        # Excel → Parquet
python tools/generate_tcia_by_class.py      # TCIA manifests
python tools/dicom_converter.py --all       # DICOM → PNG
python tools/process_overlay_aligned.py     # STL → Masks
python tools/visualize_overlay_masks.py     # Create overlays
```

## ✅ Validation

```bash
# Validate data integrity
python service/validate_data.py
```

## 🎯 Training

### Using Config File (Recommended)
```bash
# Basic training
python service/train.py --config config.yaml

# Fast training with OneCycle
python service/train.py --config config_onecycle.yaml

# Override parameters
python service/train.py --config config.yaml --epochs 100 --batch_size 16

# Resume training
python service/train.py --config config.yaml --resume checkpoints/model_best.pt
```

### Using CLI Arguments
```bash
python service/train.py \
    --manifest data/processed/class2/manifest.csv \
    --batch_size 16 \
    --epochs 100 \
    --scheduler onecycle \
    --lr 1e-4
```

## 📊 Monitoring

```bash
# Start Aim UI
aim up

# Access at: http://localhost:43800

# With custom port
aim up --port 8080
```

## ☁️ Cloud Deployment

### Create Backup (Local)
```bash
# Essential data only (~2-5 GB)
python tools/data_backup.py

# With checkpoints
python tools/data_backup.py --include-checkpoints

# Full backup
python tools/data_backup.py --full -y
```

### Deploy to Cloud
```bash
# 1. Transfer backup
scp backup.zip user@cloud:/path/to/repo/

# 2. On cloud: Clone and setup
git clone <repo-url>
cd <repo>
conda create -n mri python=3.12 -y && conda activate mri
pip install -r requirements.txt

# 3. Restore data
python tools/data_restore.py backup.zip

# 4. Train
python service/train.py --config config.yaml
```

## 🔧 Configuration Files

### `config.yaml` - Default Configuration
All training parameters in one file:
- Data paths and preprocessing
- Training hyperparameters
- Learning rate scheduler settings
- Visualization options

### `config_onecycle.yaml` - Fast Training
Optimized for rapid convergence with OneCycleLR.

### Custom Configs
```bash
# Create experiment config
cp config.yaml experiments/my_experiment.yaml

# Edit and run
vim experiments/my_experiment.yaml
python service/train.py --config experiments/my_experiment.yaml
```

## 📁 Important Directories

```
data/
├── processed/          # PNG images (required for training)
├── processed_seg/      # Segmentation masks (required for training)
├── tcia/              # TCIA manifests (for downloading DICOM)
├── nbia/              # Raw DICOM (large, not backed up)
└── splitted_images/   # Metadata parquet files

checkpoints/           # Trained models
.aim/                  # Experiment tracking data
validation_results/    # Validation visualizations
```

## 🎓 Training Schedulers

| Scheduler | When to Use | Key Params |
|-----------|-------------|------------|
| `onecycle` | Fast training, aggressive | `scheduler_max_lr_mult: 10` |
| `cosine` | Escape local minima | `scheduler_t0: 10` |
| `reduce_on_plateau` | Conservative, stable | `scheduler_patience: 5` |
| `step` | Traditional, predictable | `scheduler_step_size: 10` |

## 📈 Key Parameters

### Data
- `manifest`: Path to training data CSV
- `stack_depth`: Number of slices (2.5D), default: 5
- `batch_size`: Training batch size, default: 32

### Training
- `epochs`: Number of training epochs, default: 50
- `lr`: Learning rate, default: 5e-5
- `loss`: Loss function (dice/bce/dice_bce), default: dice_bce

### Scheduler
- `scheduler`: Type (see table above)
- `scheduler_*`: Scheduler-specific parameters

### Visualization
- `vis_every`: Save visualizations every N epochs, default: 5
- `num_vis_samples`: Number of samples to visualize, default: 4

### System
- `num_workers`: Data loading workers, default: 4
- `output_dir`: Checkpoint directory, default: checkpoints

## 🚨 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python service/train.py --config config.yaml --batch_size 8

# Reduce num_workers
python service/train.py --config config.yaml --num_workers 2
```

### Training Too Slow
```bash
# Use OneCycle scheduler
python service/train.py --config config_onecycle.yaml

# Increase batch size (if GPU allows)
python service/train.py --config config.yaml --batch_size 32
```

### Data Not Found
```bash
# Verify data integrity
python service/validate_data.py

# Check paths
ls -lh data/processed/class*/
ls -lh data/processed_seg/class*/
```

## 📚 Documentation

- **[Cloud Deployment](CLOUD_DEPLOYMENT.md)** - Detailed cloud deployment guide
- **[Training Enhanced](TRAINING_ENHANCED.md)** - Advanced training features
- **[Backup & Restore](BACKUP_RESTORE_GUIDE.md)** - Data backup system
- **[Experiment Tracking](../service/README_AIM.md)** - Using Aim for MLOps

## 🔗 Quick Links

- Main README: [README.md](../README.md)
- Documentation Index: [docs/README.md](README.md)
- Tools Overview: [tools/README.md](../tools/README.md)
- Cloud Guide: [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md)
- Training Guide: [TRAINING_ENHANCED.md](TRAINING_ENHANCED.md)

