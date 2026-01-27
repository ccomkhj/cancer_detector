#!/bin/bash
# ============================================================================
# Sync WandB Runs from Scratch Space
# ============================================================================
# 
# This script syncs WandB offline runs from scratch space to wandb.ai servers.
# It handles missing artifact staging files gracefully.
#
# Usage:
#   ./scripts/sync_wandb_runs.sh                    # Sync all runs
#   ./scripts/sync_wandb_runs.sh --run <run-id>     # Sync specific run
#   ./scripts/sync_wandb_runs.sh --status           # Check sync status
#
# ============================================================================

set -e

# Default values
WANDB_DIR="/p/scratch/ebrains-0000006/kim27/wandb"
SYNC_SPECIFIC_RUN=""
CHECK_STATUS=0
WANDB_API_KEY="8febfa084f6334dd743ecec30c3fcaa4de05cab1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run)
            SYNC_SPECIFIC_RUN="$2"
            shift 2
            ;;
        --status)
            CHECK_STATUS=1
            shift
            ;;
        --wandb-dir)
            WANDB_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--run RUN_ID] [--status] [--wandb-dir PATH]"
            echo ""
            echo "Options:"
            echo "  --run RUN_ID    Sync specific run (e.g., offline-run-20241218_210000-abc123de)"
            echo "  --status        Check sync status of all runs"
            echo "  --wandb-dir     WandB directory (default: /p/scratch/ebrains-0000006/kim27/wandb)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if wandb is installed
if ! command -v python -m wandb &> /dev/null; then
    echo "ERROR: wandb command not found. Please install wandb: pip install wandb"
    exit 1
fi

# Check if WANDB_API_KEY is set
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "ERROR: WANDB_API_KEY not set."
    echo "Set it with: export WANDB_API_KEY='<your_key>'"
    echo "Find your key at: https://wandb.ai/authorize"
    exit 1
fi

# Set up environment to use scratch space for staging
export WANDB_DIR="${WANDB_DIR}"
export WANDB_CACHE_DIR="${WANDB_DIR}/cache"
export XDG_DATA_HOME="${WANDB_DIR}/cache"

# Ensure the staging directory exists and is accessible via symlink
STAGING_DIR="${WANDB_DIR}/cache/.local/share/wandb/artifacts/staging"
if [[ ! -d "${STAGING_DIR}" ]]; then
    echo "Creating staging directory: ${STAGING_DIR}"
    mkdir -p "${STAGING_DIR}"
fi

# Check symlink
HOME_WANDB_STAGING="/p/home/jusers/kim27/jusuf/.local/share/wandb"
if [[ ! -L "${HOME_WANDB_STAGING}" ]] || [[ ! -e "${HOME_WANDB_STAGING}" ]]; then
    echo "⚠️  Symlink missing or broken. Recreating..."
    if [[ -d "${HOME_WANDB_STAGING}" ]] && [[ ! -L "${HOME_WANDB_STAGING}" ]]; then
        # Backup if it's a real directory
        mv "${HOME_WANDB_STAGING}" "${HOME_WANDB_STAGING}.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
    fi
    mkdir -p "$(dirname "${HOME_WANDB_STAGING}")"
    ln -sf "${WANDB_DIR}/cache/.local/share/wandb" "${HOME_WANDB_STAGING}"
    echo "✓ Symlink created: ${HOME_WANDB_STAGING} -> ${WANDB_DIR}/cache/.local/share/wandb"
fi

echo "============================================================================"
echo "WandB Sync Script"
echo "============================================================================"
echo "WandB directory: ${WANDB_DIR}"
echo "Staging directory: ${STAGING_DIR}"
echo ""

# Check status
if [[ ${CHECK_STATUS} -eq 1 ]]; then
    echo "Checking sync status..."
    cd "${WANDB_DIR}/wandb" 2>/dev/null || cd "${WANDB_DIR}"
    python -m wandb sync --status 2>&1 | grep -v "ERROR.*FileNotFoundError" || true
    exit 0
fi

# Sync runs
WANDB_RUNS_DIR="${WANDB_DIR}/wandb"
if [[ ! -d "${WANDB_RUNS_DIR}" ]]; then
    echo "ERROR: WandB runs directory not found: ${WANDB_RUNS_DIR}"
    exit 1
fi

cd "${WANDB_RUNS_DIR}"

if [[ -n "${SYNC_SPECIFIC_RUN}" ]]; then
    # Sync specific run
    RUN_PATH="${SYNC_SPECIFIC_RUN}"
    if [[ ! -d "${RUN_PATH}" ]]; then
        # Try to find it
        RUN_PATH=$(find . -maxdepth 1 -type d -name "*${SYNC_SPECIFIC_RUN}*" | head -1)
        if [[ -z "${RUN_PATH}" ]]; then
            echo "ERROR: Run not found: ${SYNC_SPECIFIC_RUN}"
            exit 1
        fi
    fi
    
    echo "Syncing run: ${RUN_PATH}"
    echo ""
    # Use wandb sync with error handling - ignore missing artifact file errors
    python -mwandb sync "${RUN_PATH}" 2>&1 | grep -v "ERROR.*FileNotFoundError.*artifacts/staging" || {
        EXIT_CODE=${PIPESTATUS[0]}
        if [[ ${EXIT_CODE} -eq 0 ]]; then
            echo "✓ Sync completed (some missing artifact files were ignored)"
        else
            echo "⚠️  Sync completed with warnings (missing artifact files were ignored)"
        fi
    }
else
    # Sync all offline runs
    echo "Finding offline runs..."
    OFFLINE_RUNS=$(find . -maxdepth 1 -type d -name "offline-run-*" 2>/dev/null | sort)
    
    if [[ -z "${OFFLINE_RUNS}" ]]; then
        echo "No offline runs found in ${WANDB_RUNS_DIR}"
        exit 0
    fi
    
    RUN_COUNT=$(echo "${OFFLINE_RUNS}" | wc -l)
    echo "Found ${RUN_COUNT} offline run(s)"
    echo ""
    
    SYNCED=0
    FAILED=0
    
    while IFS= read -r RUN_DIR; do
        if [[ -z "${RUN_DIR}" ]]; then
            continue
        fi
        
        RUN_NAME=$(basename "${RUN_DIR}")
        echo "============================================================================"
        echo "Syncing: ${RUN_NAME}"
        echo "============================================================================"
        
        # Sync with error filtering - ignore missing artifact staging file errors
        if python -m wandb sync "${RUN_DIR}" 2>&1 | grep -v "ERROR.*FileNotFoundError.*artifacts/staging"; then
            SYNCED=$((SYNCED + 1))
            echo "✓ Synced: ${RUN_NAME}"
        else
            # Check if sync actually succeeded despite the errors
            if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
                SYNCED=$((SYNCED + 1))
                echo "✓ Synced: ${RUN_NAME} (with warnings about missing artifact files)"
            else
                FAILED=$((FAILED + 1))
                echo "⚠️  Failed to sync: ${RUN_NAME}"
            fi
        fi
        echo ""
    done <<< "${OFFLINE_RUNS}"
    
    echo "============================================================================"
    echo "Sync Summary"
    echo "============================================================================"
    echo "Total runs: ${RUN_COUNT}"
    echo "Successfully synced: ${SYNCED}"
    echo "Failed: ${FAILED}"
    echo ""
    
    if [[ ${FAILED} -gt 0 ]]; then
        echo "Note: Some runs may have failed due to missing artifact staging files."
        echo "This is normal if staging files were cleaned up. The run data itself should be synced."
    fi
fi

echo "Done!"

