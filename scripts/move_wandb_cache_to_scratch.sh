#!/bin/bash
# ============================================================================
# Move WandB Cache to Scratch Space
# ============================================================================
# 
# This script moves the existing WandB artifacts staging cache from home
# directory to scratch space to free up home directory quota.
#
# Usage:
#   ./scripts/move_wandb_cache_to_scratch.sh
#
# ============================================================================

set -e

# Source and destination directories
HOME_WANDB_STAGING="/p/home/jusers/kim27/jusuf/.local/share/wandb"
SCRATCH_WANDB_BASE="/p/scratch/ebrains-0000006/kim27/wandb"
SCRATCH_WANDB_CACHE="${SCRATCH_WANDB_BASE}/cache"
SCRATCH_WANDB_SHARE="${SCRATCH_WANDB_CACHE}/.local/share/wandb"

echo "============================================================================"
echo "Move WandB Cache to Scratch Space"
echo "============================================================================"
echo ""
echo "Source (home): ${HOME_WANDB_STAGING}"
echo "Destination (scratch): ${SCRATCH_WANDB_SHARE}"
echo ""

# Check if source exists
if [[ ! -d "${HOME_WANDB_STAGING}" ]]; then
    echo "⚠️  Source directory does not exist: ${HOME_WANDB_STAGING}"
    echo "   Nothing to move."
    exit 0
fi

# Check source size
if [[ -d "${HOME_WANDB_STAGING}/artifacts/staging" ]]; then
    STAGING_SIZE=$(du -sh "${HOME_WANDB_STAGING}/artifacts/staging" 2>/dev/null | cut -f1 || echo "unknown")
    echo "Current staging cache size: ${STAGING_SIZE}"
    echo ""
fi

# Create destination directory structure
echo "Creating destination directory structure..."
mkdir -p "${SCRATCH_WANDB_SHARE}/artifacts"
echo "✓ Created: ${SCRATCH_WANDB_SHARE}/artifacts"
echo ""

# Move the staging directory
if [[ -d "${HOME_WANDB_STAGING}/artifacts/staging" ]]; then
    echo "Moving artifacts staging directory..."
    if [[ -d "${SCRATCH_WANDB_SHARE}/artifacts/staging" ]]; then
        echo "⚠️  Destination staging directory already exists. Merging..."
        # Move files that don't exist in destination
        rsync -av --ignore-existing "${HOME_WANDB_STAGING}/artifacts/staging/" "${SCRATCH_WANDB_SHARE}/artifacts/staging/"
        echo "✓ Merged staging files"
    else
        mv "${HOME_WANDB_STAGING}/artifacts/staging" "${SCRATCH_WANDB_SHARE}/artifacts/staging"
        echo "✓ Moved staging directory"
    fi
    echo ""
fi

# Move other wandb data if it exists
if [[ -d "${HOME_WANDB_STAGING}" ]] && [[ "$(ls -A ${HOME_WANDB_STAGING} 2>/dev/null)" ]]; then
    echo "Moving other WandB data..."
    # Move any other directories/files in .local/share/wandb
    for item in "${HOME_WANDB_STAGING}"/*; do
        if [[ -e "${item}" ]] && [[ "$(basename "${item}")" != "artifacts" ]]; then
            ITEM_NAME=$(basename "${item}")
            if [[ -d "${SCRATCH_WANDB_SHARE}/${ITEM_NAME}" ]]; then
                echo "  Merging ${ITEM_NAME}..."
                rsync -av "${item}/" "${SCRATCH_WANDB_SHARE}/${ITEM_NAME}/"
            else
                echo "  Moving ${ITEM_NAME}..."
                mv "${item}" "${SCRATCH_WANDB_SHARE}/${ITEM_NAME}"
            fi
        fi
    done
    echo "✓ Moved other WandB data"
    echo ""
fi

# Create symlink from home to scratch (so existing references still work)
echo "Creating symlink from home to scratch..."
if [[ -L "${HOME_WANDB_STAGING}" ]]; then
    echo "⚠️  Symlink already exists: ${HOME_WANDB_STAGING}"
    echo "   Removing old symlink..."
    rm "${HOME_WANDB_STAGING}"
elif [[ -d "${HOME_WANDB_STAGING}" ]]; then
    # Check if directory is now empty
    if [[ -z "$(ls -A ${HOME_WANDB_STAGING} 2>/dev/null)" ]]; then
        rmdir "${HOME_WANDB_STAGING}"
    else
        echo "⚠️  Directory not empty, creating backup..."
        mv "${HOME_WANDB_STAGING}" "${HOME_WANDB_STAGING}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
fi

# Create parent directories for symlink
mkdir -p "$(dirname "${HOME_WANDB_STAGING}")"

# Create symlink
ln -s "${SCRATCH_WANDB_SHARE}" "${HOME_WANDB_STAGING}"
echo "✓ Created symlink: ${HOME_WANDB_STAGING} -> ${SCRATCH_WANDB_SHARE}"
echo ""

# Verify
echo "============================================================================"
echo "Verification"
echo "============================================================================"
if [[ -L "${HOME_WANDB_STAGING}" ]]; then
    LINK_TARGET=$(readlink -f "${HOME_WANDB_STAGING}")
    echo "✓ Symlink verified: ${HOME_WANDB_STAGING} -> ${LINK_TARGET}"
else
    echo "⚠️  Symlink verification failed"
fi

if [[ -d "${SCRATCH_WANDB_SHARE}/artifacts/staging" ]]; then
    NEW_SIZE=$(du -sh "${SCRATCH_WANDB_SHARE}/artifacts/staging" 2>/dev/null | cut -f1 || echo "unknown")
    echo "✓ Staging cache in scratch: ${NEW_SIZE}"
else
    echo "⚠️  Staging directory not found in scratch"
fi

echo ""
echo "============================================================================"
echo "Migration Complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "1. The cache has been moved to: ${SCRATCH_WANDB_SHARE}"
echo "2. A symlink has been created so existing references still work"
echo "3. Future runs will use scratch space (via WANDB_CACHE_DIR and XDG_DATA_HOME)"
echo "4. You can now sync your runs with: wandb sync /p/scratch/ebrains-0000006/kim27/wandb/wandb/offline-run-*"
echo ""

