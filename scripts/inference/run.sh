#!/bin/bash
# ============================================================================
# Inference via Singularity/Apptainer (HPC-friendly)
# ============================================================================
# Usage:
#   scripts/inference/run.sh --config mri/config/task/segmentation.yaml --split test
#   scripts/inference/run.sh --config mri/config/task/classification.yaml --split test
# ============================================================================

set -e

if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

# Load environment variables from .env (assumed to exist)
set -a
source "${PROJECT_DIR}/.env"
set +a

# Defaults (can be overridden by .env)
SINGULARITY_IMAGE=${SINGULARITY_IMAGE:-"${PROJECT_DIR}/mri-train.sif"}
DATA_DIR=${DATA_DIR:-"${PROJECT_DIR}/data"}
SPLITS_DIR=${SPLITS_DIR:-"${PROJECT_DIR}/data/splits"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${PROJECT_DIR}/checkpoints"}
WANDB_DIR=${WANDB_DIR:-"${PROJECT_DIR}/wandb"}
WANDB_MODE=${WANDB_MODE:-"offline"}
PRETRAINED_CACHE_DIR=${PRETRAINED_CACHE_DIR:-"${PROJECT_DIR}/pretrained_model"}

CONFIG="mri/config/task/segmentation.yaml"
SPLIT="test"
PY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        *)
            PY_ARGS+=("$1")
            shift
            ;;
    esac
done

# Determine container runtime
if command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
elif command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
else
    echo "ERROR: Neither 'singularity' nor 'apptainer' found in PATH."
    exit 1
fi

if [[ ! -f "${SINGULARITY_IMAGE}" ]]; then
    echo "ERROR: Singularity image not found: ${SINGULARITY_IMAGE}"
    echo "Build it first with: scripts/build_singularity.sh"
    exit 1
fi

mkdir -p "${CHECKPOINT_DIR}" "${WANDB_DIR}" "${PROJECT_DIR}/.aim"

# Cache directories for pretrained weights (stored in scratch to avoid quota issues)
TORCH_CACHE="${PRETRAINED_CACHE_DIR}/torch"
HF_CACHE="${PRETRAINED_CACHE_DIR}/huggingface"
mkdir -p "${TORCH_CACHE}" "${HF_CACHE}"

BIND_MOUNTS=(
    "--bind" "${PROJECT_DIR}:/workspace"
    "--bind" "${DATA_DIR}:/workspace/data:ro"
    "--bind" "${SPLITS_DIR}:/workspace/data/splits:ro"
    "--bind" "${CHECKPOINT_DIR}:/workspace/checkpoints"
    "--bind" "${PROJECT_DIR}/.aim:/workspace/.aim"
    "--bind" "${WANDB_DIR}:/workspace/wandb"
    # Mount cache directories for pretrained weights (use scratch space)
    "--bind" "${PRETRAINED_CACHE_DIR}:/workspace/pretrained_cache"
)

echo "Running inference in container: ${SINGULARITY_IMAGE}"
echo "  Pretrained weights cache: ${PRETRAINED_CACHE_DIR}"
"${CONTAINER_CMD}" exec --nv \
    "${BIND_MOUNTS[@]}" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env WANDB_MODE="${WANDB_MODE}" \
    --env WANDB_DIR="/workspace/wandb" \
    --env PYTHONPATH="/workspace" \
    --env TORCH_HOME="/workspace/pretrained_cache/torch" \
    --env HF_HOME="/workspace/pretrained_cache/huggingface" \
    "${SINGULARITY_IMAGE}" \
    bash -c "cd /workspace && pip install -r requirements.txt && python service/inference.py --config ${CONFIG} --split ${SPLIT} ${PY_ARGS[*]}"
