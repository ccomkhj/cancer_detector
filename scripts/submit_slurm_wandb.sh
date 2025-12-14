#!/bin/bash
#SBATCH --job-name=mri-train-wandb
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================================
# MRI Segmentation Training with Wandb - SLURM Job Script
# ============================================================================
#
# This is a convenience script that enables wandb logging by default.
#
# Prerequisites:
#   1. Set your wandb API key:
#      export WANDB_API_KEY="your_key"
#      # OR store in file:
#      echo "your_key" > ~/.wandb_api_key
#
#   2. Build Singularity image (if using containers):
#      ./scripts/build_singularity.sh
#
# Usage:
#   # Basic usage with default project name
#   sbatch scripts/submit_slurm_wandb.sh
#
#   # Custom project name
#   WANDB_PROJECT=my-project sbatch scripts/submit_slurm_wandb.sh
#
#   # Custom entity and project
#   WANDB_ENTITY=my-team WANDB_PROJECT=my-project sbatch scripts/submit_slurm_wandb.sh
#
#   # With additional training arguments
#   sbatch scripts/submit_slurm_wandb.sh --epochs 100 --batch_size 32
#
# ============================================================================

set -e

# Configuration
WANDB_PROJECT=${WANDB_PROJECT:-"mri-segmentation"}
WANDB_ENTITY=${WANDB_ENTITY:-""}  # Leave empty for default entity

# Load wandb API key if stored in file
if [[ -z "${WANDB_API_KEY}" ]] && [[ -f "${HOME}/.wandb_api_key" ]]; then
    export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
fi

# Check for API key
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "ERROR: WANDB_API_KEY not set!"
    echo ""
    echo "Set it with one of these methods:"
    echo "  1. export WANDB_API_KEY='your_key'"
    echo "  2. echo 'your_key' > ~/.wandb_api_key"
    echo ""
    echo "Find your key at: https://wandb.ai/authorize"
    exit 1
fi

# Build wandb arguments
WANDB_ARGS="--wandb --wandb_project ${WANDB_PROJECT}"
if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ARGS="${WANDB_ARGS} --wandb_entity ${WANDB_ENTITY}"
fi

# Forward to main submit script with wandb enabled
exec "$(dirname "${BASH_SOURCE[0]}")/submit_slurm.sh" ${WANDB_ARGS} "$@"


