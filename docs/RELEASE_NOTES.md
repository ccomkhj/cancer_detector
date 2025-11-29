# Release Notes - Cloud Deployment & Training Enhancements

## Summary

Major update adding cloud deployment support, YAML configuration management, advanced learning rate schedulers, and validation visualizations to the MRI 2.5D segmentation training pipeline.

## 🆕 New Features

### 1. Data Backup & Restore System

**Tools:**
- `tools/data_backup.py` - Create compressed data backups for cloud deployment
- `tools/data_restore.py` - Restore data backups on cloud machines

**Usage:**
```bash
# Create backup
python tools/data_backup.py

# Transfer to cloud
scp backup.zip user@cloud:/path/to/repo/

# Restore on cloud
python tools/data_restore.py backup.zip
```

**Features:**
- ✅ Intelligent data selection (essential vs. full backup)
- ✅ Compression (typically 50-70% size reduction)
- ✅ Progress tracking and integrity verification
- ✅ Metadata tracking (JSON)
- ✅ Resume capability with rsync

**Backup Sizes:**
- Essential: ~2-6 GB (compressed: ~500MB-2GB)
- With checkpoints: +500 MB per model
- Full: +1-2 GB additional

### 2. YAML Configuration Management

**Files:**
- `config.yaml` - Default training configuration
- `config_onecycle.yaml` - Fast training preset

**Usage:**
```bash
# Use config file
python service/train.py --config config.yaml

# Override specific parameters
python service/train.py --config config.yaml --epochs 100 --batch_size 16
```

**Benefits:**
- ✅ All 20+ parameters in one organized file
- ✅ Version control friendly
- ✅ Easy experiment management
- ✅ CLI arguments override config values
- ✅ Backwards compatible (CLI-only still works)

### 3. Advanced Learning Rate Schedulers

**Available Schedulers:**
- `onecycle` - Super-convergence (recommended for fast training)
- `cosine` - Cosine annealing with warm restarts
- `cosine_simple` - Simple cosine annealing
- `reduce_on_plateau` - Adaptive reduction (default, stable)
- `step` - Step decay
- `exponential` - Exponential decay
- `none` - Constant learning rate

**Usage:**
```bash
# Via config file
scheduler: onecycle
scheduler_max_lr_mult: 10

# Via CLI
python service/train.py --manifest data.csv --scheduler onecycle
```

**Features:**
- ✅ Automatic batch-level vs. epoch-level stepping
- ✅ Learning rate tracking and logging
- ✅ Checkpoint-based resume with scheduler state
- ✅ Configurable parameters for each scheduler type

### 4. Validation Visualizations

**Features:**
- Automatic generation of prediction visualizations during training
- Side-by-side comparison: Input | Ground Truth | Predictions
- Multi-class visualization with color coding:
  - 🔴 Red: Prostate
  - 🟢 Green: Target1
  - 🔵 Blue: Target2
- Both overlay and contour views
- Logged to Aim for experiment tracking

**Usage:**
```bash
# Control frequency
python service/train.py --config config.yaml --vis_every 5

# Number of samples
python service/train.py --config config.yaml --num_vis_samples 8
```

**Visualization Layout:**
- Top row: Input image, GT overlay, GT contours
- Bottom row: Input image, Prediction overlay, Prediction contours
- Legend showing class names and colors

### 5. Enhanced Experiment Tracking

**Integration:**
- All metrics logged to Aim automatically
- Learning rate tracking per epoch
- Validation visualizations stored in Aim
- Hyperparameter logging (including scheduler config)
- System info tracking (GPU, CUDA, etc.)

**View Results:**
```bash
aim up
# Open: http://localhost:43800
```

## 📚 New Documentation

1. **CLOUD_DEPLOYMENT.md** - Complete cloud deployment guide
   - AWS, GCP, Azure setup instructions
   - Transfer methods (SCP, rsync, cloud storage)
   - Cost optimization tips
   - Troubleshooting guide

2. **TRAINING_ENHANCED.md** - Advanced training features
   - Scheduler comparison and recommendations
   - Visualization examples
   - Configuration management
   - Best practices

3. **BACKUP_RESTORE_GUIDE.md** - Backup system documentation
   - Detailed backup/restore workflows
   - Transfer methods for all cloud providers
   - Security considerations
   - Performance optimization

4. **QUICK_REFERENCE.md** - Command quick reference
   - Essential commands
   - Common workflows
   - Troubleshooting quick fixes

5. **CONFIG_UPDATE.md** - YAML configuration details
   - Config file structure
   - Merge behavior
   - Example workflows

## 🔄 Updated Files

### Core Scripts
- `service/train.py` - Added config support, schedulers, visualizations
- `service/logger.py` - Enhanced with image logging support
- `requirements.txt` - Added PyYAML, ensured ML dependencies

### Configuration
- `config.yaml` - Default configuration file
- `config_onecycle.yaml` - Fast training configuration
- `.gitignore` - Updated for configs and backups

### Documentation
- `README.md` - Added cloud deployment section
- Multiple new documentation files (see above)

## 🎯 Use Cases

### Local Development
```bash
# Traditional workflow still works
python service/train.py --manifest data.csv --epochs 50
```

### Cloud Training (New!)
```bash
# 1. Local: Backup data
python tools/data_backup.py

# 2. Cloud: Clone + restore
git clone <repo> && cd <repo>
python tools/data_restore.py backup.zip

# 3. Cloud: Train
python service/train.py --config config.yaml
```

### Experiment Management (New!)
```bash
# Create experiment configs
cp config.yaml experiments/exp1.yaml
cp config.yaml experiments/exp2.yaml

# Run experiments
python service/train.py --config experiments/exp1.yaml
python service/train.py --config experiments/exp2.yaml

# Compare in Aim
aim up
```

## 📊 Performance Improvements

### Training Speed
- OneCycleLR can achieve similar results in 30-50% fewer epochs
- Batch-level LR scheduling for fine-grained control
- Optimized data loading with configurable workers

### Deployment Efficiency
- 50-70% data compression for faster transfers
- Essential backup: ~2GB vs. ~10GB raw data
- Resume capability for interrupted transfers

### Monitoring
- Real-time visualization of predictions
- Learning rate tracking
- Comprehensive experiment comparison in Aim

## 🔧 Breaking Changes

**None** - All changes are backwards compatible:
- CLI-only training still works
- Existing scripts unchanged
- Default behavior preserved

## 📝 Migration Guide

### For Existing Users

**No migration needed!** But you can benefit from new features:

1. **Start using config files:**
   ```bash
   # Create your config
   cp config.yaml my_config.yaml
   # Edit and use
   python service/train.py --config my_config.yaml
   ```

2. **Try advanced schedulers:**
   ```bash
   # Edit config.yaml
   scheduler: onecycle
   # Or via CLI
   python service/train.py --manifest data.csv --scheduler onecycle
   ```

3. **Enable visualizations:**
   ```bash
   # Already enabled by default!
   # Control with --vis_every and --num_vis_samples
   ```

### For Cloud Deployment

**New workflow:**
```bash
# Old way: Manual data transfer
scp -r data/ user@cloud:/path/  # Slow, large

# New way: Backup system
python tools/data_backup.py      # Fast, compressed
scp backup.zip user@cloud:/path/ # Quick transfer
python tools/data_restore.py backup.zip  # Organized restore
```

## 🚀 Getting Started

### Quick Test
```bash
# 1. Update code
git pull

# 2. Install new dependencies
pip install -r requirements.txt

# 3. Test config
python service/train.py --config config.yaml --epochs 1

# 4. View in Aim
aim up
```

### First Cloud Deployment
```bash
# Follow: CLOUD_DEPLOYMENT.md
# Or quick start:
python tools/data_backup.py
# ... transfer and restore on cloud
```

## 📖 Learn More

- **Quick Start**: See README.md
- **Cloud Deployment**: See CLOUD_DEPLOYMENT.md
- **Advanced Training**: See TRAINING_ENHANCED.md
- **Command Reference**: See QUICK_REFERENCE.md

## 🙏 Feedback

This is a major update. Please report:
- Issues with backup/restore
- Scheduler behavior
- Cloud deployment problems
- Documentation improvements

---

**Version**: 2.0.0  
**Date**: November 2025  
**Major Changes**: Cloud deployment, YAML config, Advanced schedulers, Visualizations

