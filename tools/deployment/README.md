# Deployment Tools

Tools for deploying the training pipeline to cloud machines (AWS, GCP, Azure, etc.).

## 📦 Scripts

### `data_backup.py` - Create Data Backup

Creates compressed backups of training data for cloud deployment.

**Usage:**
```bash
# Essential backup (~2-6 GB compressed)
python tools/deployment/data_backup.py

# With checkpoints
python tools/deployment/data_backup.py --include-checkpoints

# Full backup
python tools/deployment/data_backup.py --full

# Skip confirmation
python tools/deployment/data_backup.py -y
```

**Options:**
- `--output PATH` - Custom output path
- `--include-checkpoints` - Include trained models
- `--include-aim` - Include experiment tracking data
- `--full` - Include everything
- `-y, --yes` - Skip confirmation prompt

**Output:**
- `mri_data_backup_YYYYMMDD_HHMMSS.zip` - Compressed backup
- `mri_data_backup_YYYYMMDD_HHMMSS_metadata.json` - Backup metadata

---

### `data_restore.py` - Restore Data Backup

Extracts and verifies data backups on cloud machines.

**Usage:**
```bash
# Normal restore
python tools/deployment/data_restore.py backup.zip

# Preview without extracting
python tools/deployment/data_restore.py backup.zip --dry-run

# Force overwrite
python tools/deployment/data_restore.py backup.zip --force
```

**Options:**
- `--dry-run` - Show what would be extracted
- `--force` - Force overwrite existing files

---

## 🚀 Complete Deployment Workflow

### On Local Machine

```bash
# 1. Create backup
python tools/deployment/data_backup.py

# 2. Transfer to cloud
scp mri_data_backup_*.zip user@cloud:/path/to/repo/
```

### On Cloud Machine

```bash
# 1. Clone repository
git clone <repo-url>
cd <repo>

# 2. Setup environment
conda create -n mri python=3.12 -y
conda activate mri
pip install -r requirements.txt

# 3. Restore data
python tools/deployment/data_restore.py mri_data_backup_*.zip

# 4. Verify
python service/validate_data.py

# 5. Train
python service/train.py --config config.yaml
```

---

## 📊 What Gets Backed Up

### Essential (Default)
- `data/processed/` - Processed images (~2-5 GB)
- `data/processed_seg/` - Segmentation masks (~100-500 MB)
- `data/tcia/` - TCIA manifests (~1 MB)
- `data/splitted_images/` - Metadata (~10 MB)

**Total: ~2-6 GB → ~500MB-2GB compressed**

### With `--include-checkpoints`
- `checkpoints/` - Trained models (~500 MB per model)

### With `--include-aim`
- `.aim/` - Experiment tracking data (size varies)

### With `--full`
- All of the above
- `validation_results/` - Validation visualizations
- `data/splitted_info/` - Enriched metadata

---

## 🌐 Transfer Methods

### SCP (Simple)
```bash
scp backup.zip user@cloud:/path/
```

### Rsync (Resumable)
```bash
rsync -avzP backup.zip user@cloud:/path/
```

### AWS S3
```bash
aws s3 cp backup.zip s3://bucket/
aws s3 cp s3://bucket/backup.zip .
```

### GCP Cloud Storage
```bash
gsutil cp backup.zip gs://bucket/
gsutil cp gs://bucket/backup.zip .
```

---

## 📚 Documentation

For comprehensive deployment guide, see:
- **[Cloud Deployment](../../docs/CLOUD_DEPLOYMENT.md)** - Complete guide
- **[Backup & Restore Guide](../../docs/BACKUP_RESTORE_GUIDE.md)** - Detailed docs

---

## 💡 Tips

1. **Start with essential backup** - Fastest to create and transfer
2. **Use rsync for large transfers** - Resume capability
3. **Verify after restore** - Run `service/validate_data.py`
4. **Delete backup after extraction** - Save cloud storage space

