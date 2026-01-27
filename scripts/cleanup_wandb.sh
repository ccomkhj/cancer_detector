#!/bin/bash
# ============================================================================
# Cleanup Old WandB Runs
# ============================================================================
# 
# This script helps clean up old WandB offline runs to free up disk space.
# Useful when hitting disk quota limits on HPC systems.
#
# Usage:
#   # Clean up runs older than 7 days (default)
#   ./scripts/cleanup_wandb.sh
#
#   # Clean up runs older than 1 day
#   ./scripts/cleanup_wandb.sh --days 1
#
#   # Clean up all runs (be careful!)
#   ./scripts/cleanup_wandb.sh --days 0
#
#   # Dry run (show what would be deleted without deleting)
#   ./scripts/cleanup_wandb.sh --dry-run
#
#   # Clean up from a specific directory
#   ./scripts/cleanup_wandb.sh --wandb-dir /path/to/wandb
#
# ============================================================================

set -e

# Default values
DAYS=7
DRY_RUN=0
WANDB_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --days)
            DAYS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --wandb-dir)
            WANDB_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--days N] [--dry-run] [--wandb-dir PATH]"
            echo ""
            echo "Options:"
            echo "  --days N         Delete runs older than N days (default: 7)"
            echo "  --dry-run        Show what would be deleted without deleting"
            echo "  --wandb-dir PATH Clean up from specific directory (default: ./wandb)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set default wandb directory if not provided
if [[ -z "${WANDB_DIR}" ]]; then
    WANDB_DIR="${PROJECT_DIR}/wandb"
fi

# Check if wandb directory exists
if [[ ! -d "${WANDB_DIR}" ]]; then
    echo "WandB directory not found: ${WANDB_DIR}"
    exit 1
fi

echo "============================================================================"
echo "WandB Cleanup Script"
echo "============================================================================"
echo "WandB directory: ${WANDB_DIR}"
echo "Delete runs older than: ${DAYS} days"
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "Mode: DRY RUN (no files will be deleted)"
else
    echo "Mode: DELETE (files will be permanently removed)"
fi
echo "============================================================================"
echo ""

# Find all offline run directories
RUN_DIRS=$(find "${WANDB_DIR}" -maxdepth 1 -type d -name "offline-run-*" 2>/dev/null || true)

if [[ -z "${RUN_DIRS}" ]]; then
    echo "No offline runs found in ${WANDB_DIR}"
    exit 0
fi

# Count total runs
TOTAL_RUNS=$(echo "${RUN_DIRS}" | wc -l)
echo "Found ${TOTAL_RUNS} offline run(s)"
echo ""

# Calculate cutoff time
if [[ ${DAYS} -eq 0 ]]; then
    CUTOFF_TIME=0
else
    CUTOFF_TIME=$(date -d "${DAYS} days ago" +%s 2>/dev/null || date -v-${DAYS}d +%s 2>/dev/null || echo "0")
fi

DELETED_COUNT=0
KEPT_COUNT=0
TOTAL_SIZE=0

# Process each run directory
while IFS= read -r RUN_DIR; do
    if [[ -z "${RUN_DIR}" ]]; then
        continue
    fi
    
    RUN_NAME=$(basename "${RUN_DIR}")
    
    # Get modification time
    if [[ ${DAYS} -eq 0 ]]; then
        SHOULD_DELETE=1
    else
        MOD_TIME=$(stat -c %Y "${RUN_DIR}" 2>/dev/null || stat -f %m "${RUN_DIR}" 2>/dev/null || echo "0")
        if [[ ${MOD_TIME} -lt ${CUTOFF_TIME} ]]; then
            SHOULD_DELETE=1
        else
            SHOULD_DELETE=0
        fi
    fi
    
    # Get directory size
    DIR_SIZE=$(du -sk "${RUN_DIR}" 2>/dev/null | cut -f1 || echo "0")
    TOTAL_SIZE=$((TOTAL_SIZE + DIR_SIZE))
    
    if [[ ${SHOULD_DELETE} -eq 1 ]]; then
        SIZE_MB=$((DIR_SIZE / 1024))
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "[WOULD DELETE] ${RUN_NAME} (${SIZE_MB} MB)"
        else
            echo "[DELETING] ${RUN_NAME} (${SIZE_MB} MB)"
            rm -rf "${RUN_DIR}"
        fi
        DELETED_COUNT=$((DELETED_COUNT + 1))
    else
        KEPT_COUNT=$((KEPT_COUNT + 1))
    fi
done <<< "${RUN_DIRS}"

# Clean up symlinks pointing to deleted directories
if [[ ${DRY_RUN} -eq 0 ]]; then
    find "${WANDB_DIR}" -maxdepth 1 -type l -name "*.log" -exec sh -c 'if [ ! -e "$(readlink "$1")" ]; then rm "$1"; fi' _ {} \; 2>/dev/null || true
    if [[ -L "${WANDB_DIR}/latest-run" ]] && [[ ! -e "$(readlink "${WANDB_DIR}/latest-run")" ]]; then
        rm -f "${WANDB_DIR}/latest-run"
    fi
fi

# Summary
echo ""
echo "============================================================================"
echo "Summary"
echo "============================================================================"
echo "Total runs found: ${TOTAL_RUNS}"
echo "Runs to delete: ${DELETED_COUNT}"
echo "Runs to keep: ${KEPT_COUNT}"
TOTAL_SIZE_MB=$((TOTAL_SIZE / 1024))
echo "Total size of runs to delete: ${TOTAL_SIZE_MB} MB"

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo ""
    echo "This was a dry run. No files were deleted."
    echo "Run without --dry-run to actually delete the files."
else
    echo ""
    echo "Cleanup complete!"
fi
echo "============================================================================"

