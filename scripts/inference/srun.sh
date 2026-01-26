#!/bin/bash
#SBATCH --job-name=mri-infer
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
#SBATCH --account=ebrains-0000006
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=you@example.com

# ============================================================================
# SLURM submission wrapper for inference (container via scripts/inference/run.sh)
# ============================================================================
# Usage:
#   sbatch scripts/inference/srun.sh --config mri/config/task/segmentation.yaml --split test
#   sbatch scripts/inference/srun.sh --config mri/config/task/classification.yaml --split test
#
# Notes:
# - Requires .env in project root (same as scripts/inference/run.sh)
# - Edit #SBATCH lines for your cluster (account/partition/resources)
# ============================================================================

set -e

if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

echo "============================================================================"
echo "MRI Inference (SLURM)"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Start Time:  $(date)"
echo "============================================================================"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
fi

if command -v srun &> /dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun bash scripts/inference/run.sh "$@"
else
    bash scripts/inference/run.sh "$@"
fi

echo ""
echo "============================================================================"
echo "Job Complete"
echo "End Time: $(date)"
echo "============================================================================"
