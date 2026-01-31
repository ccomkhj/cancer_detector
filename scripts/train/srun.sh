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
# #SBATCH --mail-user=you@example.com

# ============================================================================
# SLURM submission wrapper for training (container via scripts/train/run.sh)
# ============================================================================
# Usage:
#   sbatch scripts/train/srun.sh --config mri/config/task/segmentation.yaml
#   sbatch scripts/train/srun.sh --config mri/config/task/classification.yaml
#   sbatch scripts/train/srun.sh --config mri/config/task/segmentation.yaml --epochs 50
#   sbatch scripts/train/srun.sh  # Auto-uses config.yaml if present
#
# Notes:
# - Requires .env in project root (same as scripts/train/run.sh)
# - Wandb API key: export WANDB_API_KEY or set in .env
# - Edit #SBATCH lines for your cluster (account/partition/resources)
# ============================================================================

set -e

# Get project directory - SLURM copies scripts to spool, so we can't rely on BASH_SOURCE
# Use SLURM_SUBMIT_DIR if available (set by sbatch to the directory where sbatch was called)
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -n "${BASH_SOURCE[0]}" ]] && [[ "${BASH_SOURCE[0]}" != /var/spool/* ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    # Fallback: try common locations
    if [[ -d "/p/home/jusers/kim27/jusuf/shared/MRI_2.5D_Segmentation" ]]; then
        PROJECT_DIR="/p/home/jusers/kim27/jusuf/shared/MRI_2.5D_Segmentation"
    else
        echo "ERROR: Cannot determine project directory"
        exit 1
    fi
fi

# Change to project directory to ensure .env file is found
cd "${PROJECT_DIR}" || {
    echo "ERROR: Cannot change to project directory: ${PROJECT_DIR}"
    exit 1
}

echo "Project directory: ${PROJECT_DIR}"

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

    # Pretrained cache directory (load from .env if not set)
    if [[ -z "${PRETRAINED_CACHE_DIR}" ]]; then
        ENV_PRETRAINED_CACHE_DIR=$(grep "^PRETRAINED_CACHE_DIR=" "${ENV_FILE}" | cut -d'=' -f2-)
        if [[ -n "${ENV_PRETRAINED_CACHE_DIR}" ]]; then
            export PRETRAINED_CACHE_DIR="${ENV_PRETRAINED_CACHE_DIR}"
        fi
    fi

    # Set up WandB directory (use scratch space to avoid home directory quota)
    if [[ -z "${WANDB_DIR}" ]]; then
        # Try WANDB_DIR first, then WANDB_PATH as fallback
        ENV_WANDB_DIR=$(grep "^WANDB_DIR=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2-)
        if [[ -z "${ENV_WANDB_DIR}" ]]; then
            ENV_WANDB_DIR=$(grep "^WANDB_PATH=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2-)
        fi
        if [[ -n "${ENV_WANDB_DIR}" ]]; then
            export WANDB_DIR="${ENV_WANDB_DIR}"
        elif [[ -d "/p/scratch/ebrains-0000006/kim27" ]]; then
            export WANDB_DIR="/p/scratch/ebrains-0000006/kim27/wandb"
        fi
    fi

    # Load Singularity image path from .env (or use default)
    if [[ -z "${SINGULARITY_IMAGE}" ]]; then
        ENV_SINGULARITY_IMAGE=$(grep "^SINGULARITY_IMAGE=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2-)
        if [[ -n "${ENV_SINGULARITY_IMAGE}" ]]; then
            SINGULARITY_IMAGE="${ENV_SINGULARITY_IMAGE}"
        elif [[ -f "${PROJECT_DIR}/mri-train.sif" ]]; then
            SINGULARITY_IMAGE="${PROJECT_DIR}/mri-train.sif"
        else
            SINGULARITY_IMAGE="mri-train.sif"
        fi
    fi
else
    # Fallback: load wandb API key from file if no .env
    if [[ -z "${WANDB_API_KEY}" ]] && [[ -f "${HOME}/.wandb_api_key" ]]; then
        export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
    fi
fi

# Defaults if not set
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb}"
WANDB_MODE=${WANDB_MODE:-"offline"}
USE_SINGULARITY=${USE_SINGULARITY:-1}
SINGULARITY_IMAGE=${SINGULARITY_IMAGE:-"${PROJECT_DIR}/mri-train.sif"}

# Check for API key (warn but don't fail - wandb can work in offline mode)
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "WARNING: WANDB_API_KEY not set! Wandb will run in offline mode."
    echo "Set it with one of these methods:"
    echo "  1. export WANDB_API_KEY='your_key'"
    echo "  2. echo 'your_key' > ~/.wandb_api_key"
    echo "  3. Add WANDB_API_KEY=your_key to .env"
    echo ""
fi

# Separate sbatch options from script arguments
# Only match known sbatch options, not all --* arguments (to allow training args like --scheduler, --lr, etc.)
SBATCH_OPTS=""
SCRIPT_ARGS=""
HAS_CONFIG=0
HAS_OUTPUT_DIR=0
for arg in "$@"; do
    case "$arg" in
        --account=*|--partition=*|--time=*|--mem=*|--nodes=*|--ntasks=*|--cpus-per-task=*|--gres=*|--job-name=*|--output=*|--error=*|--chdir=*|--export=*|--export-file=*|--array=*|--begin=*|--dependency=*|--deadline=*|--delay-boot=*|--cpu-freq=*|--comment=*|-A*|-p*|-t*|-c*|-d*|-D*|-e*|-o*|-J*|-N*|-n*|-w*)
            SBATCH_OPTS="${SBATCH_OPTS} $arg"
            ;;
        --config|--config=*)
            HAS_CONFIG=1
            SCRIPT_ARGS="${SCRIPT_ARGS} $arg"
            ;;
        --output_dir|--output_dir=*)
            HAS_OUTPUT_DIR=1
            SCRIPT_ARGS="${SCRIPT_ARGS} $arg"
            ;;
        *)
            SCRIPT_ARGS="${SCRIPT_ARGS} $arg"
            ;;
    esac
done

# Auto-add config.yaml if no config specified and it exists
if [[ ${HAS_CONFIG} -eq 0 ]] && [[ -f "${PROJECT_DIR}/config.yaml" ]]; then
    SCRIPT_ARGS="--config config.yaml ${SCRIPT_ARGS}"
fi

# Add --output_dir from CHECKPOINT_DIR if not already provided
if [[ ${HAS_OUTPUT_DIR} -eq 0 ]] && [[ -n "${CHECKPOINT_DIR}" ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --output_dir ${CHECKPOINT_DIR}"
fi

echo "============================================================================"
echo "MRI Training (SLURM)"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Start Time:  $(date)"
echo "============================================================================"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
fi

# Export environment variables for the job
export PROJECT_DIR="${PROJECT_DIR}"
export DATA_DIR="${DATA_DIR}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR}"
export PRETRAINED_CACHE_DIR="${PRETRAINED_CACHE_DIR:-${PROJECT_DIR}/pretrained_model}"
export WANDB_DIR="${WANDB_DIR}"
export WANDB_MODE="${WANDB_MODE}"
export USE_SINGULARITY="${USE_SINGULARITY}"
export SINGULARITY_IMAGE="${SINGULARITY_IMAGE}"

if command -v srun &> /dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun bash "${PROJECT_DIR}/scripts/train/run.sh" ${SCRIPT_ARGS}
else
    bash "${PROJECT_DIR}/scripts/train/run.sh" ${SCRIPT_ARGS}
fi

echo ""
echo "============================================================================"
echo "Job Complete"
echo "End Time: $(date)"
echo "============================================================================"
