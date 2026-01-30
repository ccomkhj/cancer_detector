#!/usr/bin/env python3
"""
Pre-download pretrained encoder weights for SMP models.
Run this on a machine with internet access (e.g., login node).

Usage:
    python scripts/download_pretrained_weights.py

The weights will be cached in PRETRAINED_CACHE_DIR from .env
(defaults to /p/scratch/ebrains-0000006/kim27/MRI_2.5D_Segmentation/pretrained_model)
"""

import os
import sys
from pathlib import Path

# Get project root and load .env
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_env_file():
    """Load environment variables from .env file."""
    env_file = PROJECT_ROOT / ".env"
    env_vars = {}
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    
    return env_vars


def download_weights():
    print("="*60)
    print("Downloading pretrained encoder weights for SMP models")
    print("="*60)
    
    # Load cache directory from .env
    env_vars = load_env_file()
    cache_dir = env_vars.get('PRETRAINED_CACHE_DIR', str(PROJECT_ROOT / "pretrained_model"))
    
    torch_cache = os.path.join(cache_dir, "torch")
    hf_cache = os.path.join(cache_dir, "huggingface")
    
    # Create cache directories
    os.makedirs(torch_cache, exist_ok=True)
    os.makedirs(hf_cache, exist_ok=True)
    
    # Set environment variables to use scratch space
    os.environ['TORCH_HOME'] = torch_cache
    os.environ['HF_HOME'] = hf_cache
    
    print(f"\nCache directory: {cache_dir}")
    print(f"  TORCH_HOME: {torch_cache}")
    print(f"  HF_HOME: {hf_cache}")
    
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("\nERROR: segmentation_models_pytorch not installed")
        print("Install with: pip install segmentation-models-pytorch")
        sys.exit(1)
    
    # List of encoders to download
    encoders = [
        "resnet34",
        "resnet50",
        "resnet101",
        "efficientnet-b0",
        "efficientnet-b4",
    ]
    
    print(f"\nDownloading {len(encoders)} encoder weights...")
    print("This may take a few minutes.\n")
    
    success = 0
    failed = 0
    
    for encoder in encoders:
        print(f"Downloading {encoder}...", end=" ", flush=True)
        try:
            # Create a dummy model to trigger weight download
            # Using in_channels=3 for standard ImageNet weights
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
            print("✓")
            success += 1
            del model
        except Exception as e:
            print(f"✗ ({e})")
            failed += 1
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print(f"\nResults: {success} successful, {failed} failed")
    print(f"\nWeights are cached in: {cache_dir}")
    print("  - torch/hub/checkpoints/ (PyTorch hub)")
    print("  - huggingface/ (HuggingFace hub)")
    print("\nYou can now run training on compute nodes.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(download_weights())
