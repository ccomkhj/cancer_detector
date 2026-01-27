#!/bin/bash
# ============================================================================
# Cleanup Old WandB Staging Files
# ============================================================================
# 
# This script cleans up old WandB artifact staging files that are no longer
# needed. Staging files are temporary files used during artifact creation.
# Old staging files that are no longer referenced can be safely removed.
#
# Usage:
#   ./scripts/cleanup_wandb_staging.sh              # Dry run (show what would be deleted)
#   ./scripts/cleanup_wandb_staging.sh --execute    # Actually delete files
#
# ============================================================================

set -e

# Default values
EXECUTE=0
STAGING_DIR="/p/scratch/ebrains-0000006/kim27/wandb/cache/.local/share/wandb/artifacts/staging"
DAYS_OLD=7  # Files older than 7 days

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --execute)
            EXECUTE=1
            shift
            ;;
        --days)
            DAYS_OLD="$2"
            shift 2
            ;;
        --staging-dir)
            STAGING_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--execute] [--days N] [--staging-dir PATH]"
            echo ""
            echo "Options:"
            echo "  --execute       Actually delete files (default: dry run)"
            echo "  --days N        Delete files older than N days (default: 7)"
            echo "  --staging-dir   Staging directory path"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================================================"
echo "WandB Staging Files Cleanup"
echo "============================================================================"
echo "Staging directory: ${STAGING_DIR}"
echo "Delete files older than: ${DAYS_OLD} days"
if [[ ${EXECUTE} -eq 1 ]]; then
    echo "Mode: EXECUTE (files will be deleted)"
else
    echo "Mode: DRY RUN (no files will be deleted)"
fi
echo ""

# Check if staging directory exists
if [[ ! -d "${STAGING_DIR}" ]]; then
    echo "⚠️  Staging directory does not exist: ${STAGING_DIR}"
    exit 0
fi

# Get total size before cleanup
TOTAL_SIZE=$(du -sh "${STAGING_DIR}" 2>/dev/null | cut -f1 || echo "unknown")
echo "Current staging directory size: ${TOTAL_SIZE}"
echo ""

# Calculate cutoff time
if [[ ${DAYS_OLD} -eq 0 ]]; then
    CUTOFF_TIME=0
else
    CUTOFF_TIME=$(date -d "${DAYS_OLD} days ago" +%s 2>/dev/null || date -v-${DAYS_OLD}d +%s 2>/dev/null || echo "0")
fi

# Find old files
OLD_FILES=$(find "${STAGING_DIR}" -type f -mtime +${DAYS_OLD} 2>/dev/null || true)

if [[ -z "${OLD_FILES}" ]]; then
    echo "No files older than ${DAYS_OLD} days found."
    exit 0
fi

FILE_COUNT=$(echo "${OLD_FILES}" | wc -l)
echo "Found ${FILE_COUNT} file(s) older than ${DAYS_OLD} days"
echo ""

# Calculate total size of old files
TOTAL_OLD_SIZE=0
DELETED_COUNT=0

while IFS= read -r FILE; do
    if [[ -z "${FILE}" ]]; then
        continue
    fi
    
    FILE_SIZE=$(stat -c%s "${FILE}" 2>/dev/null || stat -f%z "${FILE}" 2>/dev/null || echo "0")
    TOTAL_OLD_SIZE=$((TOTAL_OLD_SIZE + FILE_SIZE))
    
    if [[ ${EXECUTE} -eq 1 ]]; then
        rm -f "${FILE}"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done <<< "${OLD_FILES}"

# Convert size to human readable
TOTAL_OLD_SIZE_MB=$((TOTAL_OLD_SIZE / 1024 / 1024))

echo "============================================================================"
echo "Summary"
echo "============================================================================"
echo "Files found: ${FILE_COUNT}"
if [[ ${EXECUTE} -eq 1 ]]; then
    echo "Files deleted: ${DELETED_COUNT}"
    echo "Space freed: ~${TOTAL_OLD_SIZE_MB} MB"
    
    # Get new size
    NEW_SIZE=$(du -sh "${STAGING_DIR}" 2>/dev/null | cut -f1 || echo "unknown")
    echo "New staging directory size: ${NEW_SIZE}"
else
    echo "Space that would be freed: ~${TOTAL_OLD_SIZE_MB} MB"
    echo ""
    echo "Run with --execute to actually delete these files."
fi
echo ""

