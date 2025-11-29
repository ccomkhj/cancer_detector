# Data Backup & Restore System

Complete guide for the backup and restore system designed for cloud deployment.

## 🎯 Purpose

The backup/restore system allows you to:
1. **Package** all training data into a single compressed file
2. **Transfer** data efficiently to cloud machines
3. **Restore** data with integrity verification
4. **Deploy** training pipeline on any cloud provider

## 🚀 Quick Start

### Create Backup (Local Machine)

```bash
# Essential data only (~2-5 GB compressed)
python tools/data_backup.py

# Output: mri_data_backup_YYYYMMDD_HHMMSS.zip
```

### Deploy to Cloud

```bash
# 1. Clone repository on cloud
git clone <repo-url>
cd <repo>

# 2. Setup environment
conda create -n mri python=3.12 -y
conda activate mri
pip install -r requirements.txt

# 3. Transfer backup (from local machine)
scp mri_data_backup_*.zip user@cloud:/path/to/repo/

# 4. Restore data (on cloud machine)
python tools/data_restore.py mri_data_backup_*.zip

# 5. Verify and train
python service/validate_data.py
python service/train.py --config config.yaml
```

## 📦 What Gets Backed Up

### Essential (Always Included)

| Directory | Description | Typical Size |
|-----------|-------------|--------------|
| `data/processed/` | Processed PNG images | 2-5 GB |
| `data/processed_seg/` | Segmentation masks | 100-500 MB |
| `data/tcia/` | TCIA manifests | ~1 MB |
| `data/splitted_images/` | Image metadata | ~10 MB |

**Total Essential Backup: ~2-6 GB → ~500MB-2GB compressed**

### Optional Components

#### With `--include-checkpoints`
- `checkpoints/` - Trained models (~500 MB per checkpoint)
- Use when resuming training on cloud

#### With `--include-aim`
- `.aim/` - Experiment tracking data (size varies)
- Use when continuing experiment tracking

#### With `--full`
- All of the above plus:
- `validation_results/` - Validation visualizations (~100 MB)
- `data/splitted_info/` - Enriched metadata (~20 MB)

### What's Excluded

These are NOT backed up (re-download or regenerate):

| Directory | Why Excluded | How to Get |
|-----------|--------------|------------|
| `data/nbia/` | Raw DICOM (10-50 GB) | Download from TCIA |
| `data/overlay/` | Raw STL meshes (5-10 GB) | Download from TCIA |
| `data/raw/` | Excel files | In git repository |
| `data/visualizations/` | Can be regenerated | Run visualize script |

## 🔧 Backup Options

### Basic Usage

```bash
# Essential data only (recommended)
python tools/data_backup.py

# With trained checkpoints
python tools/data_backup.py --include-checkpoints

# With Aim experiment logs
python tools/data_backup.py --include-aim

# Everything
python tools/data_backup.py --full

# Skip confirmation prompt
python tools/data_backup.py -y
```

### Advanced Options

```bash
# Custom output location
python tools/data_backup.py --output /path/to/my_backup.zip

# Full backup with custom name
python tools/data_backup.py --full --output cloud_deployment.zip -y
```

### Backup Information

The script automatically:
- ✅ Calculates total size before creating backup
- ✅ Shows what will be included
- ✅ Displays compression ratio
- ✅ Creates metadata file (JSON)
- ✅ Provides transfer instructions

## 📥 Restore Options

### Basic Usage

```bash
# Normal restore (prompts if files exist)
python tools/data_restore.py backup.zip

# Force overwrite without prompting
python tools/data_restore.py backup.zip --force

# Preview without extracting (dry run)
python tools/data_restore.py backup.zip --dry-run
```

### Restore Process

The script automatically:
- ✅ Analyzes backup contents
- ✅ Checks for file conflicts
- ✅ Extracts with progress bar
- ✅ Verifies extraction
- ✅ Provides next steps

## 🌐 Transfer Methods

### SCP (Simple, Direct)

```bash
# Basic transfer
scp backup.zip user@cloud:/path/to/repo/

# With progress
scp -v backup.zip user@cloud:/path/to/repo/

# With compression during transfer
scp -C backup.zip user@cloud:/path/to/repo/
```

### Rsync (Resumable, Efficient)

```bash
# Transfer with progress and resume capability
rsync -avzP backup.zip user@cloud:/path/to/repo/

# Resume interrupted transfer
rsync -avzP --partial backup.zip user@cloud:/path/to/repo/

# With bandwidth limit (KB/s)
rsync -avzP --bwlimit=1000 backup.zip user@cloud:/path/to/repo/
```

### AWS S3 (For AWS Deployments)

```bash
# Upload to S3 from local
aws s3 cp backup.zip s3://your-bucket/backups/

# Download on EC2 instance
aws s3 cp s3://your-bucket/backups/backup.zip .

# With progress
aws s3 cp backup.zip s3://your-bucket/backups/ \
    --storage-class STANDARD_IA
```

### Google Cloud Storage (For GCP)

```bash
# Upload from local
gsutil cp backup.zip gs://your-bucket/backups/

# Download on Compute Engine
gsutil cp gs://your-bucket/backups/backup.zip .
```

### Azure Blob Storage (For Azure)

```bash
# Upload from local
az storage blob upload \
    --account-name myaccount \
    --container-name backups \
    --file backup.zip \
    --name backup.zip

# Download on Azure VM
az storage blob download \
    --account-name myaccount \
    --container-name backups \
    --name backup.zip \
    --file backup.zip
```

## 📊 Metadata File

Each backup creates a metadata JSON file:

```json
{
  "created_at": "2024-01-15T10:30:00",
  "total_files": 5432,
  "total_size_bytes": 2147483648,
  "total_size_human": "2.00 GB",
  "included_directories": [
    "data/processed/",
    "data/processed_seg/",
    "data/tcia/"
  ],
  "backup_type": "essential",
  "includes_checkpoints": false,
  "includes_aim": false
}
```

Use this to:
- Verify backup contents
- Track backup versions
- Document what was deployed

## 🔍 Verification

### After Backup Creation

```bash
# Check backup integrity
unzip -t backup.zip

# View contents without extracting
unzip -l backup.zip | head -20
```

### After Restore

```bash
# Verify data integrity
python service/validate_data.py

# Check directory sizes
du -sh data/processed/
du -sh data/processed_seg/

# Count files
find data/processed -name "*.png" | wc -l
find data/processed_seg -name "*.png" | wc -l
```

## 💡 Best Practices

### For Initial Deployment

1. **Use essential backup only**
   ```bash
   python tools/data_backup.py
   ```
   - Fastest to create and transfer
   - Contains everything needed for training

2. **Verify locally first**
   ```bash
   python service/validate_data.py
   ```
   - Ensure data is valid before backup

3. **Test training locally**
   ```bash
   python service/train.py --config config.yaml --epochs 1
   ```
   - Verify pipeline works before cloud deployment

### For Updating Existing Deployment

1. **Include checkpoints if resuming**
   ```bash
   python tools/data_backup.py --include-checkpoints
   ```

2. **Use rsync for efficient updates**
   ```bash
   rsync -avzP backup.zip user@cloud:/path/
   ```

3. **Force overwrite on restore**
   ```bash
   python tools/data_restore.py backup.zip --force
   ```

### For Production Training

1. **Keep backup on cloud storage**
   - Upload to S3/GCS/Azure Blob
   - Enables quick redeployment

2. **Save checkpoints frequently**
   - Use `--save_every 5` in config.yaml
   - Protects against interruptions

3. **Monitor with Aim**
   - Port forward: `ssh -L 43800:localhost:43800 user@cloud`
   - Access at: http://localhost:43800

## 🚨 Troubleshooting

### Backup Too Large

**Problem:** Backup exceeds available disk space or transfer quota

**Solutions:**
```bash
# 1. Use essential only
python tools/data_backup.py

# 2. Exclude large checkpoints
python tools/data_backup.py  # Don't use --include-checkpoints

# 3. Transfer directly to cloud storage
python tools/data_backup.py
aws s3 cp backup.zip s3://bucket/  # Then delete local copy
```

### Transfer Timeout or Failure

**Problem:** Network interruption during transfer

**Solutions:**
```bash
# 1. Use rsync with resume
rsync -avzP --partial backup.zip user@cloud:/path/

# 2. Split large backups
split -b 1G backup.zip backup.part
# Transfer parts separately

# 3. Use cloud storage as intermediary
aws s3 cp backup.zip s3://bucket/
# Download from cloud machine (faster network)
```

### Extraction Fails

**Problem:** Not enough disk space or corrupted backup

**Solutions:**
```bash
# 1. Check disk space
df -h

# 2. Verify backup integrity
unzip -t backup.zip

# 3. Extract to different location
cd /mnt/large_disk/
python /path/to/tools/data_restore.py /path/to/backup.zip

# 4. Extract manually
unzip backup.zip
```

### Files Already Exist

**Problem:** Restore fails due to existing files

**Solutions:**
```bash
# 1. Preview conflicts
python tools/data_restore.py backup.zip --dry-run

# 2. Force overwrite
python tools/data_restore.py backup.zip --force

# 3. Backup existing first
mv data/processed data/processed.old
python tools/data_restore.py backup.zip
```

## 📈 Performance Tips

### Faster Backups

- **Use SSD**: Store backup on SSD for faster compression
- **Exclude large files**: Don't include checkpoints unless needed
- **Run on powerful machine**: More CPU = faster compression

### Faster Transfers

- **Compress during transfer**: `scp -C` or `rsync -z`
- **Use cloud storage**: S3/GCS often faster than direct transfer
- **Parallel transfer**: Split and transfer parts in parallel

### Faster Restore

- **SSD storage**: Extract to SSD on cloud machine
- **Parallel extraction**: Python zipfile uses single thread (future improvement)
- **Skip verification**: Use `--force` to skip prompts

## 🔐 Security Considerations

### Backup Storage

- **Encrypt backups**: Use cloud storage encryption (S3 SSE, etc.)
- **Access control**: Restrict bucket/storage access
- **Temporary URLs**: Use pre-signed URLs for transfers

### Transfer Security

- **Use SSH keys**: Better than passwords for scp/rsync
- **VPN**: Transfer through VPN for added security
- **Delete after transfer**: Remove backup from local/cloud after deployment

## 📚 Example Workflows

### Workflow 1: AWS EC2 Deployment

```bash
# Local machine
python tools/data_backup.py -y
aws s3 cp mri_data_backup_*.zip s3://my-bucket/

# EC2 instance
git clone <repo> && cd <repo>
conda create -n mri python=3.12 -y && conda activate mri
pip install -r requirements.txt
aws s3 cp s3://my-bucket/mri_data_backup_*.zip .
python tools/data_restore.py mri_data_backup_*.zip --force
python service/train.py --config config.yaml
```

### Workflow 2: Direct Transfer to GCP

```bash
# Local machine
python tools/data_backup.py --include-checkpoints -y
scp -C backup.zip user@gcp-vm:/home/user/

# GCP Compute Engine
cd /home/user
git clone <repo> && cd <repo>
conda create -n mri python=3.12 -y && conda activate mri
pip install -r requirements.txt
python tools/data_restore.py ../backup.zip --force
python service/train.py --config config_onecycle.yaml
```

### Workflow 3: Resume Training on Azure

```bash
# Local machine (with existing checkpoints)
python tools/data_backup.py --include-checkpoints --include-aim -y

# Azure VM
# ... setup as above ...
python tools/data_restore.py backup.zip --force
python service/train.py --config config.yaml \
    --resume checkpoints/model_epoch_25.pt
```

## 📖 Related Documentation

- **[CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md)** - Complete cloud deployment guide
- **[README.md](README.md)** - Main project documentation
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command reference

---

**Need Help?** Check [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) for provider-specific guides and troubleshooting.

