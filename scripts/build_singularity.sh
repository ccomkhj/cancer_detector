#!/bin/bash
# ============================================================================
# Build Singularity Image from Docker
# ============================================================================
#
# This script builds a Singularity/Apptainer image from the Dockerfile.
# Most HPC clusters use Singularity instead of Docker for security reasons.
#
# Usage:
#   # Build locally (requires sudo for Docker)
#   ./scripts/build_singularity.sh
#
#   # Build on a system with Singularity remote builder
#   ./scripts/build_singularity.sh --remote
#
#   # Specify output name
#   ./scripts/build_singularity.sh --output my-image.sif
#
# ============================================================================

set -e

if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

# Configuration
DOCKER_IMAGE_NAME="mri-train"
DOCKER_IMAGE_TAG="latest"
SINGULARITY_IMAGE="mri-train.sif"
BUILD_REMOTE=0

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
        --remote)
            BUILD_REMOTE=1
            shift
            ;;
        --output)
            SINGULARITY_IMAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "============================================================================"
echo "Building Singularity Image"
echo "============================================================================"
echo "Docker Image:      ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
echo "Singularity Image: ${SINGULARITY_IMAGE}"
echo "Build Mode:        $([ ${BUILD_REMOTE} -eq 1 ] && echo 'Remote' || echo 'Local')"
echo "============================================================================"
echo ""

# Method 1: Build Docker image first, then convert to Singularity
if [[ ${BUILD_REMOTE} -eq 0 ]]; then
    if ! command -v docker &> /dev/null; then
        echo "WARNING: 'docker' not found. Switching to remote build mode."
        echo "If your site disables remote builds, build the Docker image on a machine with Docker,"
        echo "then transfer a tarball and run:"
        echo "  ${SINGULARITY_CMD:-apptainer} build ${SINGULARITY_IMAGE} docker-archive://mri-train.tar"
        echo ""
        BUILD_REMOTE=1
    fi
fi

if [[ ${BUILD_REMOTE} -eq 0 ]]; then
    echo "Step 1: Building Docker image..."
    docker build -t "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" .
    
    echo ""
    echo "Step 2: Converting to Singularity image..."
    
    if [[ -z "${SINGULARITY_CMD}" ]]; then
        echo "ERROR: Neither singularity nor apptainer found."
        echo ""
        echo "Alternative: Save Docker image and convert on HPC cluster:"
        echo "  docker save ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG} -o ${DOCKER_IMAGE_NAME}.tar"
        echo ""
        echo "Then on HPC cluster:"
        echo "  singularity build ${SINGULARITY_IMAGE} docker-archive://${DOCKER_IMAGE_NAME}.tar"
        
        # Save Docker image as tar for manual conversion
        echo ""
        echo "Saving Docker image as tar file..."
        docker save "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" -o "${DOCKER_IMAGE_NAME}.tar"
        echo "Saved to: ${DOCKER_IMAGE_NAME}.tar"
        exit 0
    fi
    
    # Build Singularity image from Docker
    ${SINGULARITY_CMD} build "${SINGULARITY_IMAGE}" "docker-daemon://${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"

else
    # Method 2: Build directly with Singularity from definition file
    echo "Building with Singularity remote builder..."
    if [[ -z "${SINGULARITY_CMD}" ]]; then
        echo "ERROR: Neither 'singularity' nor 'apptainer' found in PATH."
        echo "Load the module provided by your cluster (e.g. 'module load apptainer') and re-run."
        exit 1
    fi
    echo "Note: Some clusters restrict Apptainer/Singularity usage (e.g. require a 'container' group)."
    
    # Create temporary Singularity definition file
    cat > singularity.def << 'EOF'
Bootstrap: docker
From: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

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
    exec python service/train.py "$@"
EOF
    
    set +e
    ${SINGULARITY_CMD} build --remote "${SINGULARITY_IMAGE}" singularity.def
    BUILD_STATUS=$?
    set -e
    if [[ ${BUILD_STATUS} -ne 0 ]]; then
        echo ""
        echo "ERROR: Remote build failed."
        echo "If you see an authorization/group error, request access from your cluster admins."
        echo "Alternative options:"
        echo "  - Build the .sif on another machine with apptainer/singularity, then copy it to the cluster"
        echo "  - Run without containers: USE_SINGULARITY=0 sbatch scripts/submit_slurm.sh"
        exit ${BUILD_STATUS}
    fi
    rm -f singularity.def
fi

echo ""
echo "============================================================================"
echo "Build Complete!"
echo "Singularity image: ${SINGULARITY_IMAGE}"
echo ""
echo "Transfer to HPC cluster:"
echo "  scp ${SINGULARITY_IMAGE} user@hpc-cluster:/path/to/project/"
echo ""
echo "Submit job:"
echo "  sbatch scripts/submit_slurm.sh"
echo "============================================================================"
