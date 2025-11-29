# Cloud Deployment Guide

Complete guide for deploying the MRI training pipeline on cloud machines (AWS, GCP, Azure, etc.).

## 🚀 Quick Start

### On Local Machine

1. **Create data backup:**
```bash
# Essential data only (images + masks)
python tools/data_backup.py

# Include trained checkpoints
python tools/data_backup.py --include-checkpoints

# Full backup (everything)
python tools/data_backup.py --full
```

### On Cloud Machine

2. **Setup repository:**
```bash
# Clone repository
git clone <your-repo-url>
cd <repo-name>

# Create conda environment
conda create -n mri python=3.12 -y
conda activate mri

# Install dependencies
pip install -r requirements.txt
```

3. **Transfer and restore data:**
```bash
# Transfer from local (run on local machine)
scp mri_data_backup_*.zip user@cloud-ip:/path/to/repo/

# Restore on cloud machine
python tools/data_restore.py mri_data_backup_*.zip

# Verify data integrity
python service/validate_data.py
```

4. **Start training:**
```bash
# Train with config
python service/train.py --config config.yaml

# Monitor (in another terminal)
aim up
```

---

## 📦 Backup Details

### What Gets Backed Up

#### Essential (Always Included):
- `data/processed/` - Processed PNG images (~2-5 GB)
- `data/processed_seg/` - Segmentation masks (~100-500 MB)
- `data/tcia/` - TCIA manifests (~1 MB)
- `data/splitted_images/` - Image metadata parquet files (~10 MB)

#### Optional (Flag Required):
- `checkpoints/` - Trained models (`--include-checkpoints`, ~500 MB per checkpoint)
- `.aim/` - Experiment tracking data (`--include-aim`, size varies)
- `validation_results/` - Validation visualizations (`--full`, ~100 MB)
- `data/splitted_info/` - Enriched metadata (`--full`, ~20 MB)

### What Doesn't Get Backed Up

These can be re-downloaded or regenerated:
- `data/nbia/` - Raw DICOM files (10-50 GB, download from TCIA)
- `data/overlay/` - Raw STL mesh files (5-10 GB, download from TCIA)
- `data/raw/` - Excel files (in git repository)
- `data/visualizations/` - Can be regenerated

---

## 💻 Cloud Provider Setup

### AWS EC2

```bash
# Launch instance
# Recommended: g4dn.xlarge (1 GPU, 4 vCPU, 16 GB RAM)
# Or: p3.2xlarge for serious training

# Connect
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init

# Setup (see Quick Start above)
```

### GCP Compute Engine

```bash
# Create instance
# Recommended: n1-standard-4 + 1x NVIDIA T4

# Connect
gcloud compute ssh your-instance-name

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init

# Setup (see Quick Start above)
```

### Azure VM

```bash
# Create VM
# Recommended: Standard_NC6 (1 GPU)

# Connect
ssh azureuser@your-vm-ip

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init

# Setup (see Quick Start above)
```

---

## 🔧 Advanced Options

### Backup Options

```bash
# Essential only (recommended for first deployment)
python tools/data_backup.py

# With checkpoints (if resuming training)
python tools/data_backup.py --include-checkpoints

# With Aim logs (to continue tracking)
python tools/data_backup.py --include-aim

# Everything
python tools/data_backup.py --full

# Custom output location
python tools/data_backup.py --output /path/to/backup.zip

# Skip confirmation
python tools/data_backup.py -y
```

### Restore Options

```bash
# Preview without extracting
python tools/data_restore.py backup.zip --dry-run

# Force overwrite
python tools/data_restore.py backup.zip --force

# Normal restore (will prompt if conflicts)
python tools/data_restore.py backup.zip
```

### Transfer Options

#### SCP (Simple)
```bash
# Local to cloud
scp backup.zip user@cloud:/path/to/repo/

# With progress
scp -v backup.zip user@cloud:/path/to/repo/
```

#### Rsync (Resumable)
```bash
# With progress and resume capability
rsync -avzP backup.zip user@cloud:/path/to/repo/

# Resume interrupted transfer
rsync -avzP --partial backup.zip user@cloud:/path/to/repo/
```

#### AWS S3 (For AWS)
```bash
# Upload to S3
aws s3 cp backup.zip s3://your-bucket/

# Download on EC2
aws s3 cp s3://your-bucket/backup.zip .
```

---

## 🎯 Training on Cloud

### Basic Training

```bash
# Edit config for cloud resources
vim config.yaml  # Adjust batch_size, num_workers based on GPU

# Start training
python service/train.py --config config.yaml

# Monitor in browser
# Forward port: ssh -L 43800:localhost:43800 user@cloud
# Open: http://localhost:43800
```

### Batch Training (Multiple Configs)

```bash
# Create experiment configs
cp config.yaml experiments/exp1_baseline.yaml
cp config.yaml experiments/exp2_onecycle.yaml
cp config.yaml experiments/exp3_cosine.yaml

# Run sequentially
for config in experiments/*.yaml; do
    echo "Training with $config"
    python service/train.py --config "$config"
done
```

### Long-Running Training

```bash
# Use screen or tmux to prevent disconnection
screen -S training

# Start training
python service/train.py --config config.yaml

# Detach: Ctrl+A then D
# Reattach: screen -r training
```

---

## 📊 Monitoring

### Local Monitoring (Port Forwarding)

```bash
# On local machine
ssh -L 43800:localhost:43800 user@cloud

# Start Aim on cloud
aim up

# Open on local browser
http://localhost:43800
```

### Cloud Monitoring (Direct Access)

```bash
# Start Aim with host binding
aim up --host 0.0.0.0 --port 43800

# Open in browser (configure firewall first)
http://cloud-ip:43800
```

---

## 🔍 Verification

### Check Data Integrity

```bash
# Verify all data is present
python service/validate_data.py

# Check directory sizes
du -sh data/processed/
du -sh data/processed_seg/

# Count files
find data/processed -name "*.png" | wc -l
find data/processed_seg -name "*.png" | wc -l
```

### Test Training

```bash
# Quick test (1 epoch)
python service/train.py \
    --manifest data/processed/class2/manifest.csv \
    --epochs 1 \
    --batch_size 4

# If successful, start full training
python service/train.py --config config.yaml
```

---

## 📝 Typical Workflow

### Initial Deployment

```bash
# 1. Local: Create and transfer backup
python tools/data_backup.py
scp mri_data_backup_*.zip user@cloud:/home/user/

# 2. Cloud: Clone and setup
git clone <repo>
cd <repo>
conda create -n mri python=3.12 -y
conda activate mri
pip install -r requirements.txt

# 3. Cloud: Restore data
python tools/data_restore.py ~/mri_data_backup_*.zip

# 4. Cloud: Verify
python service/validate_data.py

# 5. Cloud: Train
python service/train.py --config config.yaml
```

### Update Deployment

```bash
# 1. Cloud: Pull latest code
git pull

# 2. Local: Create incremental backup (if data changed)
python tools/data_backup.py --include-checkpoints

# 3. Transfer and restore
scp backup.zip user@cloud:/path/to/repo/
python tools/data_restore.py backup.zip --force

# 4. Resume training
python service/train.py --config config.yaml --resume checkpoints/model_best.pt
```

---

## 💰 Cost Optimization

### Spot Instances
- Use spot/preemptible instances for training
- Save checkpoints frequently
- Use `--save_every 5` or less

### Storage
- Delete backup zip after extraction
- Don't keep multiple backups on cloud
- Use S3/Cloud Storage for long-term backup

### Compute
- Shutdown when not training
- Use smaller batch sizes if GPU memory limited
- Reduce `num_workers` if CPU limited

---

## 🚨 Troubleshooting

### Backup Too Large
```bash
# Use essential only
python tools/data_backup.py

# Exclude large files manually
# Edit tools/data_backup.py to exclude specific directories
```

### Transfer Timeout
```bash
# Use rsync for resume capability
rsync -avzP --partial backup.zip user@cloud:/path/

# Or split the backup
split -b 1G backup.zip backup.zip.part
```

### Extraction Fails
```bash
# Check disk space
df -h

# Verify zip integrity
unzip -t backup.zip

# Extract to specific location
python tools/data_restore.py backup.zip --force
```

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📚 Resources

- AWS EC2: https://aws.amazon.com/ec2/
- GCP Compute: https://cloud.google.com/compute
- Azure VMs: https://azure.microsoft.com/en-us/services/virtual-machines/
- Aim Documentation: https://aimstack.readthedocs.io/

