# SLURM Job Submission Guide

This guide explains how to submit training jobs to a SLURM cluster.

## Quick Start

```bash
# 1. Set your wandb API key (optional but recommended)
export WANDB_API_KEY="your_key"
# OR save to file for persistence:
echo "your_key" > ~/.wandb_api_key

# 2. Build Singularity image (if your cluster uses containers)
./scripts/build_singularity.sh

# 3. Submit job
sbatch scripts/submit_slurm.sh
```

## Files Overview

| File | Description |
|------|-------------|
| `submit_slurm.sh` | Main SLURM job script |
| `submit_slurm_wandb.sh` | Convenience script with wandb enabled |
| `build_singularity.sh` | Build Singularity image from Docker |

## Prerequisites

### 1. Wandb API Key

Get your key from: https://wandb.ai/authorize

```bash
# Option 1: Environment variable
export WANDB_API_KEY="your_key"

# Option 2: Store in file (persists across sessions)
echo "your_key" > ~/.wandb_api_key
chmod 600 ~/.wandb_api_key
```

### 2. Singularity Image (for HPC clusters)

Most HPC clusters use Singularity instead of Docker:

```bash
# Build locally and convert
./scripts/build_singularity.sh

# Or build Docker and save as tar (then convert on cluster)
docker build -t mri-train .
docker save mri-train -o mri-train.tar
scp mri-train.tar user@hpc:/path/to/project/

# On HPC cluster:
singularity build mri-train.sif docker-archive://mri-train.tar
```

## Submitting Jobs

### Basic Training

```bash
# Default configuration
sbatch scripts/submit_slurm.sh

# Custom config file
sbatch scripts/submit_slurm.sh --config config_onecycle.yaml

# Override parameters
sbatch scripts/submit_slurm.sh --epochs 100 --batch_size 32
```

### Training with Wandb

```bash
# Using convenience script
sbatch scripts/submit_slurm_wandb.sh

# Custom project
WANDB_PROJECT=my-project sbatch scripts/submit_slurm_wandb.sh

# With entity (team/username)
WANDB_ENTITY=my-team WANDB_PROJECT=my-project sbatch scripts/submit_slurm_wandb.sh

# Using main script with wandb flags
sbatch scripts/submit_slurm.sh --wandb --wandb_project my-project
```

### Native Python (without Singularity)

```bash
# Use conda environment instead of container
USE_SINGULARITY=0 CONDA_ENV=mri sbatch scripts/submit_slurm.sh
```

## Customizing SLURM Parameters

Edit `submit_slurm.sh` to change resource requests:

```bash
#SBATCH --time=24:00:00        # Max runtime
#SBATCH --cpus-per-task=8      # CPU cores
#SBATCH --mem=64G              # Memory
#SBATCH --gres=gpu:1           # Number of GPUs
#SBATCH --partition=gpus        # Partition name
#SBATCH --account=ebrains-0000006 # Account
```

### Multi-GPU Training

```bash
#SBATCH --gres=gpu:4           # Request 4 GPUs
#SBATCH --cpus-per-task=32     # More CPUs for data loading
#SBATCH --mem=256G             # More memory
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job <job_id>

# View output in real-time
tail -f slurm-<job_id>.out

# Cancel job
scancel <job_id>
```

## Viewing Wandb Results

After training starts:

1. Go to https://wandb.ai
2. Navigate to your project
3. View metrics, images, and model artifacts

## Directory Structure on HPC

```
project/
├── mri-train.sif          # Singularity image
├── config.yaml            # Training config
├── data/                  # Data (mount point)
│   └── processed/
├── checkpoints/           # Model checkpoints
├── .aim/                  # Aim logs
├── slurm-*.out           # SLURM output logs
└── scripts/
    ├── submit_slurm.sh
    └── submit_slurm_wandb.sh
```

## Troubleshooting

### "WANDB_API_KEY not set"

```bash
# Check if key is set
echo $WANDB_API_KEY

# Set it
export WANDB_API_KEY="your_key"
```

### "Singularity image not found"

```bash
# Build the image first
./scripts/build_singularity.sh

# Or use native Python
USE_SINGULARITY=0 sbatch scripts/submit_slurm.sh
```

### "GPU not available"

```bash
# Check GPU allocation
srun --gres=gpu:1 nvidia-smi

# Verify CUDA in container
singularity exec --nv mri-train.sif python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

Reduce batch size:
```bash
sbatch scripts/submit_slurm.sh --batch_size 16
```

Or request more memory:
```bash
#SBATCH --mem=128G
```


