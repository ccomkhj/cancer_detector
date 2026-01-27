#!/bin/bash

# ============================================================================
# Simple SLURM job submission wrapper for inference
# ============================================================================
# Usage:
#   ./scripts/inference/submit.sh --split test
#   ./scripts/inference/submit.sh --config config.yaml --split test
#
# This script submits the job and exits immediately (no progress checking)
# ============================================================================

set -e

# Get script directory
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Change to project directory
cd "${PROJECT_DIR}" || {
    echo "ERROR: Cannot change to project directory: ${PROJECT_DIR}"
    exit 1
}

# Get absolute path to srun.sh
SRUN_SCRIPT="${SCRIPT_DIR}/srun.sh"

if [[ ! -f "${SRUN_SCRIPT}" ]]; then
    echo "ERROR: srun.sh not found at ${SRUN_SCRIPT}"
    exit 1
fi

# Submit job and exit immediately
sbatch "${SRUN_SCRIPT}" "$@"
