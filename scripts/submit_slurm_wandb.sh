#!/bin/bash

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
# Force offline mode for wandb (HPC clusters often don't have internet access)
WANDB_MODE=${WANDB_MODE:-"offline"}

# Get project directory first so we can read .env from the correct location
# Get script directory - handle both direct execution and sbatch execution
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    # Fallback if BASH_SOURCE is not available
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to project directory to ensure .env file is found
cd "${PROJECT_DIR}" || {
    echo "ERROR: Cannot change to project directory: ${PROJECT_DIR}"
    exit 1
}

# Load configuration from .env file (now we're in the project directory)
ENV_FILE="${PROJECT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    # wandb API key
    if [[ -z "${WANDB_API_KEY}" ]]; then
        if [[ -f "${HOME}/.wandb_api_key" ]]; then
            export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
        else
            export WANDB_API_KEY=$(grep "^WANDB_API_KEY=" "${ENV_FILE}" | cut -d'=' -f2-)
        fi
    fi

    # Data directory (load from .env if not set)
    if [[ -z "${DATA_DIR}" ]]; then
        ENV_DATA_DIR=$(grep "^DATA_DIR=" "${ENV_FILE}" | cut -d'=' -f2-)
        if [[ -n "${ENV_DATA_DIR}" ]]; then
            export DATA_DIR="${ENV_DATA_DIR}"
        fi
    fi

    # Checkpoint directory (load from .env if not set)
    if [[ -z "${CHECKPOINT_DIR}" ]]; then
        ENV_CHECKPOINT_DIR=$(grep "^CHECKPOINT_DIR=" "${ENV_FILE}" | cut -d'=' -f2-)
        if [[ -n "${ENV_CHECKPOINT_DIR}" ]]; then
            export CHECKPOINT_DIR="${ENV_CHECKPOINT_DIR}"
        fi
    fi
else
    # Fallback: load wandb API key from file if no .env
    if [[ -z "${WANDB_API_KEY}" ]] && [[ -f "${HOME}/.wandb_api_key" ]]; then
        export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
    fi
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

# Submit the main script with wandb arguments
# (PROJECT_DIR and SCRIPT_DIR are already set above, and we're already in PROJECT_DIR)

# DATA_DIR is now loaded from .env file above, with fallback if not set
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"

# Separate sbatch options from script arguments
# Only match known sbatch options, not all --* arguments (to allow training args like --scheduler, --lr, etc.)
SBATCH_OPTS=""
SCRIPT_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --account=*|--partition=*|--time=*|--mem=*|--nodes=*|--ntasks=*|--cpus-per-task=*|--gres=*|--job-name=*|--output=*|--error=*|--chdir=*|--export=*|--export-file=*|--array=*|--begin=*|--dependency=*|--deadline=*|--delay-boot=*|--cpu-freq=*|--comment=*|-A*|-p*|-t*|-c*|-d*|-D*|-e*|-o*|-J*|-N*|-n*|-w*)
            SBATCH_OPTS="${SBATCH_OPTS} $arg"
            ;;
        *)
            SCRIPT_ARGS="${SCRIPT_ARGS} $arg"
            ;;
    esac
done

# Ensure submit_slurm.sh path is absolute (SCRIPT_DIR is already absolute from pwd above)
SUBMIT_SLURM_SCRIPT="${SCRIPT_DIR}/submit_slurm.sh"
if [[ ! -f "${SUBMIT_SLURM_SCRIPT}" ]]; then
    echo "ERROR: submit_slurm.sh not found at ${SUBMIT_SLURM_SCRIPT}"
    echo "SCRIPT_DIR: ${SCRIPT_DIR}"
    echo "Current directory: $(pwd)"
    exit 1
fi

# SCRIPT_DIR is already absolute, so this path is also absolute
# Use it directly to ensure sbatch can find it regardless of working directory
exec sbatch ${SBATCH_OPTS} --export=PROJECT_DIR="${PROJECT_DIR}",DATA_DIR="${DATA_DIR}",CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints}",WANDB_MODE="${WANDB_MODE}" "${SUBMIT_SLURM_SCRIPT}" ${WANDB_ARGS} ${SCRIPT_ARGS}


