#!/bin/bash
# ============================================================================
# Build Singularity/Apptainer Image from Definition File
# ============================================================================
#
# This script builds a Singularity/Apptainer image directly from singularity.def.
# It pulls the base Docker image from Docker Hub automatically (no Docker required).
# Most HPC clusters use Singularity/Apptainer instead of Docker for security reasons.
#
# Usage:
#   # Build with default settings
#   ./scripts/build_singularity.sh
#
#   # Pull pre-built container first, then add project files (recommended for HPC)
#   ./scripts/build_singularity.sh --pull --scratch
#
#   # Specify output name
#   ./scripts/build_singularity.sh --output my-image.sif
#
#   # Use fakeroot mode (if available and needed)
#   ./scripts/build_singularity.sh --fakeroot
#
# Prerequisites:
#   - Singularity or Apptainer installed (load module if needed: module load apptainer)
#   - Internet access to pull base image from Docker Hub
#   - Proper permissions (may require fakeroot or sudo depending on cluster setup)
#
# ============================================================================

set -e

if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

# Configuration
SINGULARITY_IMAGE="mri-train.sif"
SINGULARITY_DEF="singularity.def"
USE_FAKEROOT=0
SCRATCH_DIR=""
USE_SCRATCH=0
USE_PULL=0
BASE_IMAGE="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
BASE_IMAGE_LOCAL=""

# Determine available container runtime command (singularity/apptainer)
if command -v singularity &> /dev/null; then
    SINGULARITY_CMD="singularity"
elif command -v apptainer &> /dev/null; then
    SINGULARITY_CMD="apptainer"
else
    SINGULARITY_CMD=""
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            SINGULARITY_IMAGE="$2"
            shift 2
            ;;
        --fakeroot)
            USE_FAKEROOT=1
            shift
            ;;
        --scratch)
            USE_SCRATCH=1
            shift
            ;;
        --scratch-dir)
            SCRATCH_DIR="$2"
            USE_SCRATCH=1
            shift 2
            ;;
        --pull)
            USE_PULL=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output IMAGE.sif      Specify output image name (default: mri-train.sif)"
            echo "  --pull                  Pull pre-built base container first, then add project files"
            echo "                          (recommended when building from definition fails)"
            echo "  --fakeroot              Use fakeroot mode (may fail on older GLIBC systems)"
            echo "  --scratch               Use scratch space for cache and output (auto-detect)"
            echo "  --scratch-dir DIR       Use specific scratch directory"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Pull base container and build in scratch space (recommended for HPC)"
            echo "  $0 --pull --scratch"
            echo ""
            echo "  # Build from definition in scratch space WITHOUT fakeroot"
            echo "  $0 --scratch"
            echo ""
            echo "  # Build in scratch space WITH fakeroot (if GLIBC >= 2.33)"
            echo "  $0 --scratch --fakeroot"
            echo ""
            echo "  # Build with custom scratch directory"
            echo "  $0 --scratch-dir /p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation"
            echo ""
            echo "Note: If you get GLIBC errors, use --pull to pull pre-built container first."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Determine scratch directory
if [[ ${USE_SCRATCH} -eq 1 ]]; then
    if [[ -n "${SCRATCH_DIR}" ]]; then
        # Use specified scratch directory
        BUILD_DIR="${SCRATCH_DIR}"
    elif [[ -n "${SCRATCH}" ]]; then
        # Use $SCRATCH environment variable
        BUILD_DIR="${SCRATCH}/MRI_2.5D_Segmentation"
    elif [[ -d "/p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation" ]]; then
        # Use known scratch location
        BUILD_DIR="/p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation"
    else
        echo "WARNING: --scratch specified but no scratch directory found."
        echo "  Tried: \$SCRATCH_DIR, \$SCRATCH, and /p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation"
        echo "  Falling back to project directory."
        BUILD_DIR="${PROJECT_DIR}"
    fi
    
    # Create build directory if it doesn't exist
    mkdir -p "${BUILD_DIR}"
    
    # Set Apptainer cache to scratch space (to avoid home directory quota issues)
    export APPTAINER_CACHEDIR="${BUILD_DIR}/.apptainer_cache"
    export SINGULARITY_CACHEDIR="${BUILD_DIR}/.apptainer_cache"
    
    # If output is relative, make it relative to build directory
    if [[ "${SINGULARITY_IMAGE}" != /* ]]; then
        SINGULARITY_IMAGE="${BUILD_DIR}/${SINGULARITY_IMAGE}"
    fi
    
    echo "Using scratch space for build:"
    echo "  Build directory: ${BUILD_DIR}"
    echo "  Cache directory: ${APPTAINER_CACHEDIR}"
    echo "  Output image:    ${SINGULARITY_IMAGE}"
    echo ""
else
    BUILD_DIR="${PROJECT_DIR}"
fi

# Check if Singularity/Apptainer is available
if [[ -z "${SINGULARITY_CMD}" ]]; then
    echo "ERROR: Neither 'singularity' nor 'apptainer' found in PATH."
    echo ""
    echo "Please load the appropriate module:"
    echo "  module load apptainer"
    echo "  # or"
    echo "  module load singularity"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

# Make definition file path absolute
if [[ "${SINGULARITY_DEF}" != /* ]]; then
    SINGULARITY_DEF="${PROJECT_DIR}/${SINGULARITY_DEF}"
fi

# Check if definition file exists
if [[ ! -f "${SINGULARITY_DEF}" ]]; then
    echo "ERROR: Definition file not found: ${SINGULARITY_DEF}"
    echo ""
    echo "The singularity.def file should be in the project root directory."
    exit 1
fi

echo "============================================================================"
echo "Building Singularity/Apptainer Image"
echo "============================================================================"
if [[ ${USE_PULL} -eq 1 ]]; then
    echo "Build mode:        Pull base container, then add project files"
else
    echo "Build mode:        Build from definition file"
fi
echo "Definition file:  ${SINGULARITY_DEF}"
echo "Output image:     ${SINGULARITY_IMAGE}"
echo "Container tool:   ${SINGULARITY_CMD}"
echo "Fakeroot mode:    $([ ${USE_FAKEROOT} -eq 1 ] && echo 'Yes (may fail on older GLIBC)' || echo 'No')"
if [[ ${USE_SCRATCH} -eq 1 ]]; then
    echo "Cache directory:  ${APPTAINER_CACHEDIR:-${SINGULARITY_CACHEDIR}}"
fi
echo "============================================================================"
echo ""

# Check if output image already exists
if [[ -f "${SINGULARITY_IMAGE}" ]]; then
    echo "WARNING: Output image already exists: ${SINGULARITY_IMAGE}"
    echo "It will be overwritten."
    echo ""
    # Only prompt if running interactively
    if [[ -t 0 ]]; then
        read -p "Continue? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Build cancelled."
            exit 0
        fi
    else
        echo "Non-interactive mode: overwriting existing image."
    fi
    rm -f "${SINGULARITY_IMAGE}"
fi

# If --pull is used, pull base container first, then build from it
if [[ ${USE_PULL} -eq 1 ]]; then
    echo "============================================================================"
    echo "Step 1: Pulling Base Container"
    echo "============================================================================"
    echo "Pulling base image: ${BASE_IMAGE}"
    echo ""
    
    # Determine where to store the base image (use BUILD_DIR which is already set)
    if [[ ${USE_SCRATCH} -eq 1 ]]; then
        BASE_IMAGE_LOCAL="${BUILD_DIR}/base-pytorch.sif"
    else
        BASE_IMAGE_LOCAL="${PROJECT_DIR}/base-pytorch.sif"
    fi
    
    # Make sure BUILD_DIR exists if using scratch
    if [[ ${USE_SCRATCH} -eq 1 ]]; then
        mkdir -p "${BUILD_DIR}"
    fi
    
    # Check if base image already exists
    if [[ -f "${BASE_IMAGE_LOCAL}" ]]; then
        echo "Base image already exists: ${BASE_IMAGE_LOCAL}"
        echo "Reusing existing base image (to re-pull, delete it first: rm ${BASE_IMAGE_LOCAL})"
        echo ""
    else
        # Pull the base container
        echo "Running: ${SINGULARITY_CMD} pull ${BASE_IMAGE_LOCAL} docker://${BASE_IMAGE}"
        echo ""
        
        if ${SINGULARITY_CMD} pull "${BASE_IMAGE_LOCAL}" "docker://${BASE_IMAGE}"; then
            echo ""
            echo "✓ Base container pulled successfully!"
            echo ""
        else
            echo ""
            echo "ERROR: Failed to pull base container!"
            exit 1
        fi
    fi
    
    echo "Note: If building from this base image fails due to permissions, you can:"
    echo "  1. Use sudo: sudo $0 --pull --scratch"
    echo "  2. Or use the base container directly (requirements installed at runtime by submit script)"
    echo ""
    
    # Create temporary definition file that uses the local image as base
    TEMP_DEF="${PROJECT_DIR}/.singularity.def.tmp"
    # Use absolute path for base image in definition file
    BASE_IMAGE_ABS="$(cd "$(dirname "${BASE_IMAGE_LOCAL}")" && pwd)/$(basename "${BASE_IMAGE_LOCAL}")"
    cat > "${TEMP_DEF}" << EOF
Bootstrap: localimage
From: ${BASE_IMAGE_ABS}

%labels
    Author MRI Segmentation Team
    Version 1.0

%environment
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1

%files
    requirements.txt /app/requirements.txt
    service /app/service
    tools /app/tools
    config.yaml /app/config.yaml

%post
    apt-get update && apt-get install -y --no-install-recommends \
        git curl vim && \
        rm -rf /var/lib/apt/lists/*
    
    pip install --no-cache-dir --upgrade pip
    pip install --no-cache-dir -r /app/requirements.txt
    
    mkdir -p /app/checkpoints /app/logs /app/.aim
    chmod -R 755 /app

%runscript
    cd /app
    exec python service/train.py "\$@"
EOF
    
    SINGULARITY_DEF="${TEMP_DEF}"
    echo "Created temporary definition file using pulled base image."
    echo ""
fi

# Build the image
if [[ ${USE_PULL} -eq 1 ]]; then
    echo "============================================================================"
    echo "Step 2: Building Custom Image from Base Container"
    echo "============================================================================"
else
    echo "Building image from ${SINGULARITY_DEF}..."
fi
echo "This may take several minutes as it installs dependencies."
echo ""

BUILD_CMD="${SINGULARITY_CMD} build"

# Add fakeroot if requested and available
# Note: When using --pull, we avoid fakeroot unless explicitly requested (due to GLIBC issues)
if [[ ${USE_FAKEROOT} -eq 1 ]]; then
    # Check if fakeroot is supported (both Apptainer and Singularity 3.x+ support it)
    if ${SINGULARITY_CMD} build --help 2>&1 | grep -q "fakeroot"; then
        BUILD_CMD="${BUILD_CMD} --fakeroot"
        echo "Using fakeroot mode for unprivileged build."
        if [[ ${USE_PULL} -eq 1 ]]; then
            echo "WARNING: Using fakeroot with --pull may still fail on systems with older GLIBC."
            echo "If this fails, try building without --fakeroot or use sudo."
        fi
    else
        echo "WARNING: Fakeroot mode requested but may not be supported by this version."
        echo "Attempting build anyway..."
    fi
elif [[ ${USE_PULL} -eq 1 ]]; then
    # When using --pull, try without fakeroot first (avoids GLIBC issues)
    # If this fails, the error message will suggest using sudo
    echo "Building from pulled base image without fakeroot (recommended)."
    echo "If you get permission errors, try: sudo $0 --pull --scratch"
fi

# Build the image
set +e
${BUILD_CMD} "${SINGULARITY_IMAGE}" "${SINGULARITY_DEF}" 2>&1 | tee /tmp/singularity_build.log
BUILD_STATUS=$?
BUILD_OUTPUT=$(cat /tmp/singularity_build.log)
set -e

# Verify image was actually created
IMAGE_EXISTS=0
if [[ -f "${SINGULARITY_IMAGE}" ]]; then
    IMAGE_EXISTS=1
fi

# Check for build failures - either non-zero exit or missing image
if [[ ${BUILD_STATUS} -ne 0 ]] || [[ ${IMAGE_EXISTS} -eq 0 ]]; then
    # Check for GLIBC version errors (common fakeroot issue)
    if echo "${BUILD_OUTPUT}" | grep -qiE "GLIBC.*not found.*faked|fakeroot.*error|FATAL.*while performing build|FATAL.*while running.*post"; then
        echo ""
        echo "============================================================================"
        echo "GLIBC Version Mismatch Detected"
        echo "============================================================================"
        echo ""
        echo "Fakeroot requires GLIBC 2.33+ but your system has an older version."
        echo "This is a common issue on HPC clusters."
        echo ""
        echo "Solutions:"
        echo ""
        echo "1. Use sudo with --pull (if you have sudo access):"
        echo "   sudo $0 --pull --scratch-dir /p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation --output my-custom-image.sif"
        echo ""
        echo "2. Pull pre-built container WITHOUT fakeroot (if you used --fakeroot, remove it):"
        echo "   $0 --pull --scratch-dir /p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation --output my-custom-image.sif"
        echo ""
        echo "3. Alternative: Use the pulled base container directly and install requirements at runtime:"
        echo "   ${SINGULARITY_CMD} pull ${SINGULARITY_IMAGE} docker://${BASE_IMAGE}"
        echo "   (The submission script will install requirements automatically)"
        echo ""
        echo "4. Build on a machine with newer GLIBC, then transfer the .sif file"
        echo ""
        echo "5. Run without containers:"
        echo "   USE_SINGULARITY=0 sbatch scripts/submit_slurm.sh"
        echo ""
        
        # Clean up partial build if it exists
        if [[ -f "${SINGULARITY_IMAGE}" ]]; then
            echo "Removing incomplete image file..."
            rm -f "${SINGULARITY_IMAGE}"
        fi
        
        rm -f /tmp/singularity_build.log
        exit ${BUILD_STATUS}
    fi
fi

# If build failed for other reasons or image doesn't exist, show general error
if [[ ${BUILD_STATUS} -ne 0 ]] || [[ ${IMAGE_EXISTS} -eq 0 ]]; then
    echo ""
    echo "============================================================================"
    echo "ERROR: Build failed with exit code ${BUILD_STATUS}"
    echo "============================================================================"
    echo ""
    echo "Common issues and solutions:"
    echo ""
    echo "1. GLIBC version mismatch or permission issues:"
    if [[ ${USE_PULL} -eq 1 ]]; then
        echo "   - Use sudo with --pull: sudo $0 --pull --scratch"
        echo "   - Or use the base container directly (requirements installed at runtime)"
    else
        echo "   - Use --pull to pull pre-built container first: $0 --pull --scratch"
        echo "   - Then use sudo if needed: sudo $0 --pull --scratch"
    fi
    echo ""
    echo "2. Build WITHOUT fakeroot:"
    echo "   - Try: $0 --scratch"
    echo "   - Or use sudo: sudo $0 --scratch"
    echo ""
    echo "3. Permission denied / not in /etc/subuid:"
    if [[ ${USE_PULL} -eq 1 ]]; then
        echo "   - Use sudo: sudo $0 --pull --scratch"
    else
        echo "   - Use --pull: $0 --pull --scratch"
        echo "   - Or use sudo: sudo $0 --pull --scratch"
    fi
    echo "   - Or request access from cluster admins"
    echo ""
    echo "4. Disk quota exceeded (cache in home directory):"
    echo "   - Use scratch space: $0 --scratch"
    echo "   - Or specify custom scratch: $0 --scratch-dir /path/to/scratch"
    echo ""
    echo "5. Network issues (cannot pull base image):"
    echo "   - Check internet connectivity"
    echo "   - Some clusters require proxy settings"
    echo ""
    echo "6. Missing files in definition:"
    echo "   - Ensure requirements.txt, service/, tools/, and config.yaml exist"
    echo ""
    echo "7. Alternative: Build on another machine with Docker/Singularity, then transfer:"
    echo "   - Build Docker image: docker build -t mri-train:latest ."
    echo "   - Save: docker save mri-train:latest -o mri-train.tar"
    echo "   - On cluster: ${SINGULARITY_CMD} build ${SINGULARITY_IMAGE} docker-archive://mri-train.tar"
    echo ""
    echo "8. Run without containers:"
    echo "   USE_SINGULARITY=0 sbatch scripts/submit_slurm.sh"
    echo ""
    
    # Clean up partial build if it exists
    if [[ -f "${SINGULARITY_IMAGE}" ]]; then
        echo "Removing incomplete image file..."
        rm -f "${SINGULARITY_IMAGE}"
    fi
    
    rm -f /tmp/singularity_build.log
    exit ${BUILD_STATUS}
fi

# Final verification that image exists
if [[ ! -f "${SINGULARITY_IMAGE}" ]]; then
    echo ""
    echo "ERROR: Build reported success but image file not found: ${SINGULARITY_IMAGE}"
    echo "This should not happen. Please check the build logs above."
    rm -f /tmp/singularity_build.log
    exit 1
fi

rm -f /tmp/singularity_build.log

# Clean up temporary files
if [[ ${USE_PULL} -eq 1 ]] && [[ -f "${TEMP_DEF}" ]]; then
    rm -f "${TEMP_DEF}"
    echo "Cleaned up temporary definition file."
    echo ""
fi

echo ""
echo "============================================================================"
echo "Build Complete!"
echo "============================================================================"
echo "Singularity image: ${SINGULARITY_IMAGE}"
if [[ -f "${SINGULARITY_IMAGE}" ]]; then
    echo "Image size: $(du -h "${SINGULARITY_IMAGE}" | cut -f1)"
    
    # Optionally clean up base image if it was pulled
    if [[ ${USE_PULL} -eq 1 ]] && [[ -f "${BASE_IMAGE_LOCAL}" ]]; then
        echo ""
        echo "Base image used: ${BASE_IMAGE_LOCAL}"
        echo "You can remove it to save space: rm ${BASE_IMAGE_LOCAL}"
    fi
else
    echo "WARNING: Image file not found!"
fi
echo ""
echo "Next steps:"
echo "  1. Verify the image:"
echo "     ${SINGULARITY_CMD} inspect ${SINGULARITY_IMAGE}"
echo ""
echo "  2. Submit a training job:"
echo "     sbatch scripts/submit_slurm.sh"
echo ""
echo "  3. Or test interactively:"
echo "     ${SINGULARITY_CMD} shell ${SINGULARITY_IMAGE}"
echo ""
echo "============================================================================"







