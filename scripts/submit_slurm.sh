#!/bin/bash
#SBATCH --job-name=mri-train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
#SBATCH --account=ebrains-0000006
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=ccomkhj@gmail.com

# ============================================================================
# MRI Segmentation Training - SLURM Job Script
# ============================================================================
# 
# This script supports two modes:
#   1. Singularity container (recommended for HPC)
#   2. Native Python with conda/venv
#
# Usage:
#   # Submit with default settings
#   sbatch scripts/submit_slurm.sh
#
#   # Submit with custom config
#   sbatch scripts/submit_slurm.sh --config config_onecycle.yaml
#
#   # Submit with wandb enabled
#   sbatch scripts/submit_slurm.sh --wandb --wandb_project myproject
#
#   # Use native Python instead of Singularity
#   USE_SINGULARITY=0 sbatch scripts/submit_slurm.sh
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Use Singularity container? (1=yes, 0=no/use native Python)
USE_SINGULARITY=${USE_SINGULARITY:-1}

# Singularity image path (build with: scripts/build_singularity.sh)
SINGULARITY_IMAGE=${SINGULARITY_IMAGE:-"mri-train.sif"}

# Conda environment name (if not using Singularity)
CONDA_ENV=${CONDA_ENV:-"mri"}

# Project directory (auto-detected from script location)
PROJECT_DIR=${PROJECT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}

# Data directory (can be overridden for HPC setups with separate data storage)
DATA_DIR=${DATA_DIR:-"${PROJECT_DIR}/data"}

# Load configuration from .env file
if [[ -f ".env" ]]; then
    # wandb API key (set via environment or file)
    # Option 1: Export before submitting: export WANDB_API_KEY="your_key"
    # Option 2: Store in file: echo "your_key" > ~/.wandb_api_key
    # Option 3: Store in .env file: WANDB_API_KEY=your_key
    if [[ -z "${WANDB_API_KEY}" ]]; then
        if [[ -f "${HOME}/.wandb_api_key" ]]; then
            export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
        else
            export WANDB_API_KEY=$(grep "^WANDB_API_KEY=" .env | cut -d'=' -f2-)
        fi
    fi

    # Data directory (for HPC setups with separate data storage)
    # Option 1: Export before submitting: export DATA_DIR="/path/to/data"
    # Option 2: Store in .env file: DATA_DIR=/path/to/data
    if [[ -z "${DATA_DIR}" ]]; then
        ENV_DATA_DIR=$(grep "^DATA_DIR=" .env | cut -d'=' -f2-)
        if [[ -n "${ENV_DATA_DIR}" ]]; then
            export DATA_DIR="${ENV_DATA_DIR}"
        fi
    fi
else
    # Fallback: load wandb API key from file if no .env
    if [[ -z "${WANDB_API_KEY}" ]] && [[ -f "${HOME}/.wandb_api_key" ]]; then
        export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
    fi
fi

# ============================================================================
# Print Job Info
# ============================================================================

echo "============================================================================"
echo "MRI Segmentation Training Job"
echo "============================================================================"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Job Name:        ${SLURM_JOB_NAME}"
echo "Node:            ${SLURMD_NODENAME}"
echo "CPUs:            ${SLURM_CPUS_PER_TASK}"
echo "Memory:          ${SLURM_MEM_PER_NODE}MB"
echo "GPUs:            ${SLURM_GPUS:-1}"
echo "Project Dir:     ${PROJECT_DIR}"
echo "Use Singularity: ${USE_SINGULARITY}"
echo "Start Time:      $(date)"
echo "============================================================================"

# Change to project directory
cd "${PROJECT_DIR}"

# ============================================================================
# GPU Info
# ============================================================================

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# ============================================================================
# Run Training
# ============================================================================

# Default training arguments
TRAIN_ARGS="--config config.yaml"

# Append any additional arguments passed to the script
if [[ $# -gt 0 ]]; then
    TRAIN_ARGS="${TRAIN_ARGS} $@"
fi

echo "Training Arguments: ${TRAIN_ARGS}"
echo ""

if [[ "${USE_SINGULARITY}" == "1" ]]; then
    # ========================================================================
    # Singularity Mode
    # ========================================================================
    
    # Some clusters provide Apptainer instead of Singularity
    if command -v singularity &> /dev/null; then
        CONTAINER_CMD="singularity"
    elif command -v apptainer &> /dev/null; then
        CONTAINER_CMD="apptainer"
    else
        echo "ERROR: Neither 'singularity' nor 'apptainer' found in PATH."
        echo "Load a module (e.g. 'module load apptainer') or set USE_SINGULARITY=0."
        exit 1
    fi

    if [[ ! -f "${SINGULARITY_IMAGE}" ]]; then
        echo "ERROR: Singularity image not found: ${SINGULARITY_IMAGE}"
        echo "Build it first with: scripts/build_singularity.sh"
        exit 1
    fi
    
    echo "Running with container (${CONTAINER_CMD}): ${SINGULARITY_IMAGE}"
    echo ""
    
    # Run training in Singularity container
    # Note: .aim directory not mounted to avoid version compatibility issues
    # The container will use wandb for logging instead
    "${CONTAINER_CMD}" exec --nv \
        --bind "${PROJECT_DIR}:/workspace" \
        --bind "${DATA_DIR}:/workspace/data:ro" \
        --bind "${PROJECT_DIR}/checkpoints:/workspace/checkpoints" \
        --env WANDB_API_KEY="${WANDB_API_KEY}" \
        --env WANDB_MODE="offline" \
        --env PYTHONPATH="/workspace" \
        "${SINGULARITY_IMAGE}" \
        bash -c "cd /workspace && pip install -r requirements.txt && python service/train.py ${TRAIN_ARGS}"

else
    # ========================================================================
    # Native Python Mode (Conda/venv)
    # ========================================================================
    
    echo "Running with native Python"
    echo ""
    
    # Load conda if available
    if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    elif command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
    fi
    
    # Activate conda environment
    if conda activate "${CONDA_ENV}" 2>/dev/null; then
        echo "Activated conda environment: ${CONDA_ENV}"
    else
        echo "WARNING: Could not activate conda environment: ${CONDA_ENV}"
        echo "Using current Python environment"
    fi
    
    # Show Python info
    echo "Python: $(which python)"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo ""
    
    # Run training
    python service/train.py ${TRAIN_ARGS}
fi

# ============================================================================
# Job Complete
# ============================================================================

echo ""
echo "============================================================================"
echo "Job Complete"
echo "End Time: $(date)"
echo "============================================================================"

