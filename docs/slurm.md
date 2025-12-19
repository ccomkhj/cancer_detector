# SLURM (HPC Cluster) Job Submission

This repo includes SLURM submission scripts in `scripts/` that support running training in a **Singularity/Apptainer container** with automatic dependency installation.

## Quick Start

```bash
# 1. Build container in scratch space (avoids disk quota issues)
./scripts/build_singularity.sh --scratch --fakeroot

# 2. Set wandb API key (optional)
echo "WANDB_API_KEY=your_key" > .env

# 3. Submit training job
sbatch scripts/submit_slurm.sh --wandb --wandb_project myproject
```

**Tip:** Always use `--scratch` when building containers on HPC clusters to avoid home directory disk quota limits.

## Prerequisites

### 1) Project Setup on Cluster

1. Copy/clone this repo onto the cluster (preferably on a fast filesystem like `$SCRATCH`).
2. Ensure your data is available. By default, the script looks for data in `data/` directory, but you can override this for HPC setups with separate data storage (see below).
3. Create output directories:

```bash
mkdir -p checkpoints .aim logs
```

#### Checkpoint Storage

Models are saved efficiently to conserve storage space:

- **Location**: `checkpoints/<job-id>/model_epoch_X.pt`
- **Policy**: Only the best performing model is kept (not all epochs)
- **Resume**: Use `--resume checkpoints/<job-id>/model_epoch_X.pt`

Example:
```bash
# Resume from a specific job's best checkpoint
python service/train.py --config config.yaml --resume checkpoints/755384/model_epoch_25.pt
```

#### Data Directory Configuration

For HPC setups where data is stored separately (e.g., in scratch space), you can specify the data location:

```bash
# Option 1: Set environment variable before submission
export DATA_DIR="/p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation/data"
sbatch scripts/submit_slurm_wandb.sh

# Option 2: The script defaults to this common HPC data path:
# /p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation/data
```

### 2) Wandb Setup (Optional)

Set your wandb API key using one of these methods:

```bash
# Option 1: Environment variable
export WANDB_API_KEY="your_key"

# Option 2: Store in .env file (recommended)
echo "WANDB_API_KEY=your_key" > .env

# Option 3: Store in home directory
echo "your_key" > ~/.wandb_api_key
```

**Note:** On HPC clusters, wandb automatically runs in offline mode due to network restrictions. Logs will be stored locally and can be synced later when you have internet access.

### Wandb Offline Mode Workflow

1. **During Training**: All logs are stored locally in `wandb/` directory
2. **Directory Structure**:
   ```
   wandb/
   └── offline-run-20241218_210000-abc123de/
       ├── logs/
       ├── files/
       └── run-abc123de.wandb
   ```

### Syncing Offline Runs

After training completes, sync your results to wandb servers (requires internet access):

```bash
# Navigate to your project directory
cd /path/to/your/project

# Sync all offline runs from HPC
wandb sync wandb/offline-run-*

# Or sync specific runs
wandb sync wandb/offline-run-20241218_210000-abc123de

# Check sync status
wandb sync --status

# View synced runs online
wandb runs list
```

#### Data Directory Configuration

For HPC setups where data is stored separately (e.g., in scratch space), you can specify the data location:

```bash
# Option 1: Environment variable (overrides .env)
export DATA_DIR="/p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation/data"

# Option 2: Store in .env file (recommended)
echo "DATA_DIR=/p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation/data" >> .env

# Option 3: The script defaults to the HPC data path if no .env is found
```

### 3) Singularity Container Setup

#### Building the Container Image

The container image can be built from the `singularity.def` definition file. **It's recommended to build in scratch space** to avoid home directory disk quota issues.

##### Option 1: Build in Scratch Space (Recommended)

This avoids home directory quota issues by using scratch space for both the build cache and output image:

```bash
# Auto-detect scratch directory
./scripts/build_singularity.sh --scratch --fakeroot

# Or specify custom scratch directory
./scripts/build_singularity.sh --scratch-dir /p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation --fakeroot

# With custom output name
./scripts/build_singularity.sh --scratch --output my-custom-image.sif --fakeroot
```

**Benefits:**
- Avoids home directory disk quota limits
- Apptainer cache stored in scratch space (not home directory)
- Image built directly in scratch space
- Automatically detected by submission scripts

##### Option 2: Build in Project Directory

If you have sufficient space in your home directory:

```bash
# Build in current directory
./scripts/build_singularity.sh --fakeroot

# With custom output name
./scripts/build_singularity.sh --output my-image.sif --fakeroot
```

##### Option 3: Pull Pre-built Container (Alternative)

If building fails, you can pull a pre-built PyTorch container:

```bash
# Pull PyTorch container with CUDA support
apptainer pull mri-train.sif docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
```

**Note:** If you get "not authorized to use apptainer", ensure you're in the `container` group. Contact your cluster admin if needed.

**Important:** The submission scripts automatically detect the image location:
- Project directory: `mri-train.sif`
- Scratch space: `/p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation/mri-train.sif`
- Or specify explicitly: `SINGULARITY_IMAGE=/path/to/image.sif sbatch scripts/submit_slurm.sh`

**Fallback:** If no container is found, the scripts will automatically fall back to native Python mode.

## Job Submission

### Basic Training Job

```bash
# With wandb logging (recommended) - includes account specification
./scripts/submit_slurm_wandb.sh --account=ebrains-0000006

# Without wandb logging
sbatch --account=ebrains-0000006 scripts/submit_slurm.sh
```

### Custom Configuration

```bash
# Use different config file
./scripts/submit_slurm_wandb.sh --account=ebrains-0000006 --config config_onecycle.yaml

# Override training parameters
./scripts/submit_slurm_wandb.sh --account=ebrains-0000006 --epochs 100 --batch_size 16
```

### Cluster-Specific Options

For clusters requiring account/budget specification:

```bash
./scripts/submit_slurm_wandb.sh --account=<BUDGET> --partition=<PARTITION>
```

For GPU jobs (if not using default GPU partition):

```bash
./scripts/submit_slurm_wandb.sh --account=<BUDGET> --gres=gpu:<N> --partition=<GPU_PARTITION>
```

**Note:** Use `./scripts/submit_slurm_wandb.sh` (not `sbatch`) as it's a wrapper script that passes options to sbatch internally.

## How It Works

The submission scripts automatically:

1. **Load wandb API key** from `.env` file, `~/.wandb_api_key`, or environment variable
2. **Detect Singularity container** by checking:
   - Project directory: `mri-train.sif`
   - Scratch space: `/p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation/mri-train.sif`
   - `$SCRATCH` environment variable location
   - Falls back to native Python if no container found
3. **Install dependencies** at runtime inside the container (`pip install -r requirements.txt`)
4. **Mount directories** properly for data access and output storage
5. **Handle GPU acceleration** through CUDA runtime in the container
6. **Run wandb in offline mode** automatically (for HPC clusters without internet access)

## Monitoring & Debugging

```bash
# Check job status
squeue -u $USER

# Monitor job output
tail -f slurm-<jobid>.out

# Check for errors
tail -f slurm-<jobid>.err

# Cancel job
scancel <jobid>
```

## Troubleshooting

### Common Issues

**"WANDB_API_KEY not set"**
- Add your key to `.env` file: `echo "WANDB_API_KEY=your_key" > .env`

**"No such file or directory" errors**
- Scripts use relative paths and should work from project directory
- Ensure you're submitting from the project root

**Data directory not found**
- Set `DATA_DIR` environment variable to point to your data location
- Or add to `.env` file: `echo "DATA_DIR=/path/to/data" >> .env`
- Default HPC path: `/p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation/data`

**Container image not found**
- The script automatically searches:
  - Project directory: `mri-train.sif`
  - Scratch space: `/p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation/mri-train.sif`
  - `$SCRATCH` environment variable location
- To specify custom location: `SINGULARITY_IMAGE=/path/to/image.sif sbatch scripts/submit_slurm.sh`
- Or build the image: `./scripts/build_singularity.sh --scratch --fakeroot`

**Container authorization errors**
- Ensure you're in the `container` group on your cluster
- Contact cluster admin for container access

**"Disk quota exceeded" during build**
- **Solution:** Build in scratch space using `--scratch` flag:
  ```bash
  ./scripts/build_singularity.sh --scratch --fakeroot
  ```
- This redirects Apptainer cache to scratch space (avoids home directory quota)
- Image is also built in scratch space
- Submission scripts automatically detect images in scratch space

**Singularity build failures**
- Try building with `--fakeroot` flag if you get permission errors
- Use `--scratch` to avoid disk quota issues
- If building still fails, try pulling pre-built container:
  ```bash
  apptainer pull mri-train.sif docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
  ```
- GLIBC/fakeroot issues are common - using scratch space often resolves them

### Fallback Mode

If containers don't work, jobs automatically fall back to native Python mode. Ensure dependencies are available:

```bash
# Load required modules (cluster-specific)
module load cuda anaconda

# Create and activate environment
conda create -n mri python=3.10 -y
conda activate mri
pip install -r requirements.txt

# Submit in native mode
USE_SINGULARITY=0 sbatch scripts/submit_slurm.sh
```

## Script Customization

Edit the `#SBATCH` directives in `scripts/submit_slurm.sh` for cluster-specific settings:

- `--partition`, `--account` (often required)
- `--time`, `--mem`, `--cpus-per-task`
- `--gres=gpu:<N>` for GPU requests

The scripts are designed to work out-of-the-box with minimal configuration required.