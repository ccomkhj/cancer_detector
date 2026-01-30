#!/bin/bash
# ============================================================================
# Build Singularity/Apptainer Image for HPC Training
# ============================================================================
#
# This script builds a Singularity image with all dependencies baked in.
# Run this on a LOGIN NODE (with internet access), not on compute nodes.
#
# Usage:
#   ./scripts/build_image.sh                    # Build with defaults
#   ./scripts/build_image.sh --scratch          # Use scratch space (recommended)
#   ./scripts/build_image.sh --output my.sif    # Custom output name
#
# Prerequisites:
#   - Run on a login node with internet access
#   - Singularity or Apptainer available (module load apptainer)
#
# ============================================================================

set -e

# Get project directory first to load .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load configuration from .env
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    set -a
    source "${PROJECT_DIR}/.env"
    set +a
fi

# Configuration (SINGULARITY_IMAGE loaded from .env, or use default)
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${PROJECT_DIR}/mri-train.sif}"
USE_SCRATCH=0
SCRATCH_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output|-o)
            SINGULARITY_IMAGE="$2"
            shift 2
            ;;
        --scratch|-s)
            USE_SCRATCH=1
            shift
            ;;
        --scratch-dir)
            SCRATCH_DIR="$2"
            USE_SCRATCH=1
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output, -o IMAGE.sif   Output image path (default: from .env or ./mri-train.sif)"
            echo "  --scratch, -s            Use scratch space for build (recommended)"
            echo "  --scratch-dir DIR        Use specific scratch directory"
            echo "  --help, -h               Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --scratch                    # Build in scratch space"
            echo "  $0 --output custom.sif          # Custom output name"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to project directory
cd "${PROJECT_DIR}"

# Determine container runtime
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "ERROR: Neither 'singularity' nor 'apptainer' found."
    echo ""
    echo "Load the module first:"
    echo "  module load apptainer"
    echo "  # or"
    echo "  module load singularity"
    exit 1
fi

# Set up build cache directory
if [[ ${USE_SCRATCH} -eq 1 ]]; then
    if [[ -n "${SCRATCH_DIR}" ]]; then
        BUILD_DIR="${SCRATCH_DIR}"
    elif [[ -n "${SCRATCH}" ]]; then
        BUILD_DIR="${SCRATCH}/mri-build"
    else
        # Use parent directory of SINGULARITY_IMAGE as scratch
        BUILD_DIR="$(dirname "${SINGULARITY_IMAGE}")/build-cache"
    fi

    mkdir -p "${BUILD_DIR}"
    export APPTAINER_CACHEDIR="${BUILD_DIR}/.cache"
    export SINGULARITY_CACHEDIR="${BUILD_DIR}/.cache"
    mkdir -p "${APPTAINER_CACHEDIR}"

    echo "Using scratch space for build cache:"
    echo "  Build dir: ${BUILD_DIR}"
    echo "  Cache dir: ${APPTAINER_CACHEDIR}"
fi

# Ensure output directory exists
OUTPUT_DIR="$(dirname "${SINGULARITY_IMAGE}")"
mkdir -p "${OUTPUT_DIR}"

# Check definition file exists
SINGULARITY_DEF="${PROJECT_DIR}/singularity.def"
if [[ ! -f "${SINGULARITY_DEF}" ]]; then
    echo "ERROR: Definition file not found: ${SINGULARITY_DEF}"
    exit 1
fi

# Check required files exist
for f in requirements.txt mri service tools; do
    if [[ ! -e "${PROJECT_DIR}/${f}" ]]; then
        echo "ERROR: Required file/directory not found: ${f}"
        exit 1
    fi
done

echo "============================================================================"
echo "Building Singularity Image"
echo "============================================================================"
echo "Container tool:   ${CONTAINER_CMD}"
echo "Definition file:  ${SINGULARITY_DEF}"
echo "Output image:     ${SINGULARITY_IMAGE}"
echo "============================================================================"
echo ""

# Remove existing image if present
if [[ -f "${SINGULARITY_IMAGE}" ]]; then
    echo "Removing existing image..."
    rm -f "${SINGULARITY_IMAGE}"
fi

# Build the image
echo "Building image (this may take several minutes)..."
echo ""

if ${CONTAINER_CMD} build "${SINGULARITY_IMAGE}" "${SINGULARITY_DEF}"; then
    echo ""
    echo "============================================================================"
    echo "Build Complete!"
    echo "============================================================================"
    echo "Image: ${SINGULARITY_IMAGE}"
    echo "Size:  $(du -h "${SINGULARITY_IMAGE}" | cut -f1)"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Update .env with the image path:"
    echo "     SINGULARITY_IMAGE=${SINGULARITY_IMAGE}"
    echo ""
    echo "  2. Submit a training job:"
    echo "     ./scripts/train/submit.sh"
    echo ""
    echo "  3. Or test interactively:"
    echo "     ${CONTAINER_CMD} shell --nv ${SINGULARITY_IMAGE}"
    echo ""
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "ERROR: Build failed!"
    echo "============================================================================"
    echo ""
    echo "Common solutions:"
    echo ""
    echo "  1. Use scratch space to avoid quota issues:"
    echo "     $0 --scratch"
    echo ""
    echo "  2. If permission errors, try with sudo (if available):"
    echo "     sudo $0 --scratch"
    echo ""
    echo "  3. Check network connectivity (needed to pull base image)"
    echo ""
    exit 1
fi
