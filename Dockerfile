# MRI Segmentation Training Container
# Supports GPU training with PyTorch and wandb logging

# Use NVIDIA PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # wandb API key (pass at runtime)
    WANDB_API_KEY="" \
    # Disable wandb console output for cleaner logs
    WANDB_SILENT=true

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for outputs
RUN mkdir -p /app/checkpoints /app/logs /app/.aim

# Set permissions for potential non-root execution (common in HPC)
RUN chmod -R 755 /app

# Default command - can be overridden
CMD ["python", "service/train.py", "--config", "config.yaml"]


