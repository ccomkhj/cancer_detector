# Documentation

Complete documentation for the MRI 2.5D Segmentation Pipeline.

## 📚 Quick Navigation

### Getting Started
- **[Main README](../README.md)** - Project overview and quick start
- **[Quick Reference](QUICK_REFERENCE.md)** - Essential commands and workflows
- **[Release Notes](RELEASE_NOTES.md)** - Latest features and updates

### User Guides
- **[Cloud Deployment](CLOUD_DEPLOYMENT.md)** - Deploy to AWS, GCP, Azure
- **[SLURM Cluster](slurm.md)** - Submit training jobs on HPC
- **[Training Enhanced](TRAINING_ENHANCED.md)** - Advanced training features
- **[Backup & Restore](BACKUP_RESTORE_GUIDE.md)** - Data backup system

### Tools Documentation
- **[Tools Overview](../tools/README.md)** - All tools organized by category
- **[Preprocessing Tools](../tools/preprocessing/)** - Data conversion
- **[TCIA Tools](../../tcia-handler/tools/tcia/)** - Manifest generation
- **[Dataset Tools](../tools/dataset/)** - PyTorch loaders
- **[Deployment Tools](../tools/deployment/)** - Cloud deployment
- **[Validation Tools](../tools/validation/)** - Testing & validation

### Service Documentation
- **[AIM Tracking](../service/README_AIM.md)** - Experiment tracking guide
- **[View Metrics](../service/VIEW_METRICS.md)** - How to view training metrics

---

## 📖 Documentation Structure

```
docs/
├── README.md                      # This file
├── CLOUD_DEPLOYMENT.md            # Cloud deployment guide
├── TRAINING_ENHANCED.md           # Advanced training features
├── BACKUP_RESTORE_GUIDE.md        # Backup system documentation
├── QUICK_REFERENCE.md             # Command quick reference
├── RELEASE_NOTES.md               # Version history and features
├── guides/                        # Technical guides
│   ├── FINAL_TRAINING_SETUP.md   # Complete training guide
│   ├── DICOM_ALIGNED_PROCESSING.md # Mask alignment details
│   ├── MASK_DETECTION_SOLUTION.md # Mask loading system
│   └── README_2D5.md             # 2.5D architecture details
└── archive/                       # Historical documentation
    └── (development notes and legacy docs)
```

---

## 🎯 Documentation by Task

### I want to...

#### **Train a model locally**
1. [Main README - Quick Start](../README.md#-quick-start-3-commands)
2. [Quick Reference - Training](QUICK_REFERENCE.md#-training)
3. [Training Enhanced](TRAINING_ENHANCED.md)

#### **Deploy to cloud**
1. [Cloud Deployment Guide](CLOUD_DEPLOYMENT.md)
2. [Backup & Restore](BACKUP_RESTORE_GUIDE.md)
3. [Deployment Tools](../tools/deployment/README.md)

#### **Understand the data pipeline**
1. [Tools Overview](../tools/README.md)
2. [Preprocessing Guide](guides/DICOM_ALIGNED_PROCESSING.md)
3. [Mask Detection](guides/MASK_DETECTION_SOLUTION.md)

#### **Use advanced training features**
1. [Training Enhanced](TRAINING_ENHANCED.md)
2. [Config Files](../config.yaml)
3. [Quick Reference](QUICK_REFERENCE.md)

#### **Track experiments**
1. [AIM Guide](../service/README_AIM.md)
2. [View Metrics](../service/VIEW_METRICS.md)

#### **Troubleshoot issues**
1. [Quick Reference - Troubleshooting](QUICK_REFERENCE.md#-troubleshooting)
2. [Cloud Deployment - Troubleshooting](CLOUD_DEPLOYMENT.md#-troubleshooting)
3. [Validation Tools](../tools/validation/README.md)

---

## 📊 Documentation Categories

### Essential Documentation (Start Here)
These docs cover the most common use cases:

| Document | Purpose | Audience |
|----------|---------|----------|
| [Main README](../README.md) | Project overview, quick start | Everyone |
| [Quick Reference](QUICK_REFERENCE.md) | Common commands | All users |
| [Cloud Deployment](CLOUD_DEPLOYMENT.md) | Deploy to cloud | Cloud users |
| [Training Enhanced](TRAINING_ENHANCED.md) | Advanced training | ML practitioners |

### Technical Guides (Deep Dives)
Detailed technical documentation in `guides/`:

| Document | Purpose | Audience |
|----------|---------|----------|
| [Final Training Setup](guides/FINAL_TRAINING_SETUP.md) | Complete training guide | Developers |
| [DICOM Aligned Processing](guides/DICOM_ALIGNED_PROCESSING.md) | Mask alignment details | Technical users |
| [Mask Detection](guides/MASK_DETECTION_SOLUTION.md) | How masks are loaded | Developers |
| [2.5D Architecture](guides/README_2D5.md) | Model architecture | ML developers |

### Historical Documentation
Development notes and legacy documentation in `archive/`:
- Implementation notes
- Quick start guides (superseded)
- Development planning docs

---

## 🔄 Workflow Documentation

### Complete Workflow (Local → Cloud)

```
1. Data Preparation (Local)
   └─→ tools/preprocessing/ + tools/README.md

2. Validation (Local)
   └─→ service/validate_data.py + tools/validation/

3. Training (Local)
   └─→ TRAINING_ENHANCED.md + QUICK_REFERENCE.md

4. Backup (Local)
   └─→ tools/deployment/ + BACKUP_RESTORE_GUIDE.md

5. Deploy (Cloud)
   └─→ CLOUD_DEPLOYMENT.md

6. Train (Cloud)
   └─→ TRAINING_ENHANCED.md + service/README_AIM.md
```

---

## 🆕 What's New

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for:
- Latest features
- Breaking changes
- Migration guides
- Version history

---

## 💡 Tips for Using Documentation

1. **Start with the main README** - Get overview and run quick start
2. **Use Quick Reference** - Fast lookup for common commands
3. **Check tools/README.md** - Understanding data processing
4. **Read category-specific READMEs** - Detailed info for each tool category
5. **Refer to guides/** - Deep technical understanding

---

## 🔗 External Resources

- [TCIA Prostate-MRI-US-Biopsy](https://www.cancerimagingarchive.net/collection/prostate-mri-us-biopsy/)
- [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
- [3D Slicer](https://www.slicer.org/)
- [Aim Documentation](https://aimstack.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📝 Contributing to Documentation

When adding new documentation:
1. Place user-facing docs in `docs/`
2. Place tool-specific docs in `tools/[category]/README.md`
3. Place historical docs in `docs/archive/`
4. Update this index (docs/README.md)
5. Update cross-references in affected documents

---

Need help? Check the [Quick Reference](QUICK_REFERENCE.md) or open an issue on GitHub.
