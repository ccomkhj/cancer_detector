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
#
# Notes:
# - Requires .env in project root (same as scripts/train/run.sh)
# - Edit #SBATCH lines for your cluster (account/partition/resources)
# ============================================================================

set -e

if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

echo "============================================================================"
echo "MRI Training (SLURM)"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Start Time:  $(date)"
echo "============================================================================"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
fi

if command -v srun &> /dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun bash scripts/train/run.sh "$@"
else
    bash scripts/train/run.sh "$@"
fi

echo ""
echo "============================================================================"
echo "Job Complete"
echo "End Time: $(date)"
echo "============================================================================"
