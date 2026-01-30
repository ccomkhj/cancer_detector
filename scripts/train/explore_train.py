#!/usr/bin/env python3
"""
Hyperparameter & Model Exploration Script for MRI 2.5D Segmentation

This script submits multiple SLURM jobs with different hyperparameter configurations
and model architectures to explore the best training setup.

Usage:
    python scripts/train/explore_train.py
    python scripts/train/explore_train.py --dry-run  # Preview without submitting
    python scripts/train/explore_train.py --max-jobs 10  # Limit number of jobs

The script will:
    1. Submit multiple jobs with different model/hyperparameter combinations
    2. Track job IDs and their corresponding configurations
    3. Save results to a YAML file (exploration_runs_<timestamp>.yaml)
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file() -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    env_file = PROJECT_ROOT / ".env"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
    
    return env_vars


def load_base_config(config_path: str = "config.yaml") -> Dict:
    """Load the base configuration file."""
    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def submit_job(config_overrides: Dict, dry_run: bool = False) -> Optional[str]:
    """
    Submit a SLURM job with the given configuration overrides.
    
    Args:
        config_overrides: Dictionary of parameters to override in the config
        dry_run: If True, just print the command without submitting
        
    Returns:
        Job ID if successful, None otherwise
    """
    # Use the new submit script
    script_path = SCRIPT_DIR / "srun.sh"
    
    if not script_path.exists():
        raise FileNotFoundError(f"Submission script not found: {script_path}")
    
    # Build CLI arguments from config overrides
    cli_args = []
    
    # Load CHECKPOINT_DIR from .env if --output_dir not already specified
    if 'output_dir' not in config_overrides:
        env_vars = load_env_file()
        checkpoint_dir = env_vars.get('CHECKPOINT_DIR')
        if checkpoint_dir:
            cli_args.extend(['--output_dir', checkpoint_dir])
    
    for key, value in config_overrides.items():
        arg_key = f"--{key}"
        # Handle list values (e.g., ft_alpha, ft_beta, ft_class_weights)
        if isinstance(value, list):
            cli_args.append(arg_key)
            cli_args.extend([str(v) for v in value])
        else:
            cli_args.extend([arg_key, str(value)])
    
    # Build sbatch command
    cmd = ["sbatch", str(script_path)] + cli_args
    
    print(f"\n{'='*80}")
    print(f"Submitting job with config:")
    for key, value in config_overrides.items():
        print(f"  {key}: {value}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    if dry_run:
        print("  [DRY RUN - not submitted]")
        return "dry_run"
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout.strip() + "\n" + result.stderr.strip()
        match = re.search(r'Submitted batch job (\d+)', output)
        
        if match:
            job_id = match.group(1)
            print(f"✓ Job submitted successfully! Job ID: {job_id}")
            return job_id
        else:
            if result.stdout.strip():
                print(f"stdout: {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"stderr: {result.stderr.strip()}")
            print(f"⚠ Warning: Could not extract job ID from output")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Error submitting job:")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None


# ============================================================================
# Model Configurations
# ============================================================================
# Available models with their expected performance characteristics

MODELS = [
    # Simple baseline
    {"model": "simple_unet", "description": "Simple U-Net (baseline)"},
    
    # SMP U-Net variants with different encoders
    {"model": "smp_unet_resnet34", "description": "U-Net + ResNet34 encoder"},
    {"model": "smp_unet_resnet50", "description": "U-Net + ResNet50 encoder"},
    {"model": "smp_unet_efficientnet-b0", "description": "U-Net + EfficientNet-B0"},
    {"model": "smp_unet_efficientnet-b4", "description": "U-Net + EfficientNet-B4"},
    
    # U-Net++ (nested U-Net)
    {"model": "smp_unetplusplus_resnet34", "description": "U-Net++ + ResNet34"},
    {"model": "smp_unetplusplus_resnet50", "description": "U-Net++ + ResNet50"},
    
    # DeepLabV3+ (good for multi-scale features)
    {"model": "smp_deeplabv3plus_resnet34", "description": "DeepLabV3+ + ResNet34"},
    {"model": "smp_deeplabv3plus_resnet50", "description": "DeepLabV3+ + ResNet50"},
    
    # FPN (Feature Pyramid Network)
    {"model": "smp_fpn_resnet34", "description": "FPN + ResNet34"},
    {"model": "smp_fpn_resnet50", "description": "FPN + ResNet50"},
    
    # MAnet (Multi-scale Attention)
    {"model": "smp_manet_resnet34", "description": "MAnet + ResNet34"},
    
    # Linknet (fast inference)
    {"model": "smp_linknet_resnet34", "description": "LinkNet + ResNet34"},
]


# ============================================================================
# Hyperparameter Configurations
# ============================================================================

HYPERPARAMETERS = {
    # Learning rates to explore
    "lr": [0.0001, 0.00005, 0.00001, 0.0005],
    
    # Schedulers
    "schedulers": [
        {"scheduler": "cosine", "scheduler_t0": 15, "scheduler_tmult": 2, "scheduler_min_lr": 1e-6},
        {"scheduler": "onecycle", "scheduler_max_lr_mult": 10.0, "scheduler_warmup_pct": 0.3},
        {"scheduler": "reduce_on_plateau", "scheduler_patience": 5, "scheduler_factor": 0.5},
    ],
    
    # Loss functions
    "losses": [
        {"loss": "focal_tversky", "ft_gamma": 1.33, "ft_alpha": [0.6, 0.8], "ft_beta": [0.4, 0.2]},
        {"loss": "focal_tversky", "ft_gamma": 1.5, "ft_alpha": [0.7, 0.9], "ft_beta": [0.3, 0.1]},
        {"loss": "dice_bce"},
    ],
    
    # Batch sizes
    "batch_size": [4, 8, 16],
    
    # Stack depths
    "stack_depth": [3, 5, 7],
}


def generate_exploration_configs(max_configs: int = 50) -> List[Dict]:
    """
    Generate exploration configurations combining models and hyperparameters.
    
    Strategy:
    1. Test each model with baseline hyperparameters
    2. Test best-looking models with different hyperparameters
    3. Ensure diverse coverage of the search space
    """
    configs = []
    
    # Base hyperparameters (from config.yaml defaults)
    # Note: Pretrained weights will be loaded from local cache
    # Run scripts/download_pretrained_weights.py on login node first!
    base_hp = {
        "epochs": 50,  # Shorter for exploration
        "lr": 0.0001,
        "loss": "focal_tversky",
        "scheduler": "cosine",
        "scheduler_t0": 15,
        "scheduler_tmult": 2,
        "scheduler_min_lr": 1e-6,
        # encoder_weights defaults to "imagenet" - loaded from local cache
    }
    
    # ========================================================================
    # PART 1: Test all models with baseline hyperparameters (13 configs)
    # ========================================================================
    for model_info in MODELS:
        config = {
            "name": f"model_{model_info['model']}",
            "model": model_info["model"],
            **base_hp,
        }
        configs.append(config)
    
    # ========================================================================
    # PART 2: Test promising models with different learning rates (12 configs)
    # ========================================================================
    promising_models = [
        "simple_unet",
        "smp_unet_resnet34",
        "smp_unetplusplus_resnet34",
        "smp_deeplabv3plus_resnet34",
    ]
    
    for model in promising_models:
        for lr in [0.00005, 0.00001, 0.0005]:
            config = {
                "name": f"{model}_lr{lr:.0e}".replace(".", "_"),
                "model": model,
                "lr": lr,
                "epochs": 50,
                "loss": "focal_tversky",
                "scheduler": "cosine",
                "scheduler_t0": 15,
                "scheduler_tmult": 2,
                "scheduler_min_lr": 1e-6,
            }
            configs.append(config)
    
    # ========================================================================
    # PART 3: Test different schedulers with best models (9 configs)
    # ========================================================================
    for model in ["smp_unet_resnet34", "smp_unetplusplus_resnet34", "smp_deeplabv3plus_resnet34"]:
        # OneCycle scheduler
        configs.append({
            "name": f"{model}_onecycle",
            "model": model,
            "lr": 0.00005,
            "epochs": 50,
            "loss": "focal_tversky",
            "scheduler": "onecycle",
            "scheduler_max_lr_mult": 10.0,
            "scheduler_warmup_pct": 0.3,
        })
        
        # ReduceLROnPlateau
        configs.append({
            "name": f"{model}_plateau",
            "model": model,
            "lr": 0.0001,
            "epochs": 50,
            "loss": "focal_tversky",
            "scheduler": "reduce_on_plateau",
            "scheduler_patience": 5,
            "scheduler_factor": 0.5,
        })
        
        # Cosine with longer cycles
        configs.append({
            "name": f"{model}_cosine_long",
            "model": model,
            "lr": 0.0001,
            "epochs": 50,
            "loss": "focal_tversky",
            "scheduler": "cosine",
            "scheduler_t0": 25,
            "scheduler_tmult": 1,
            "scheduler_min_lr": 1e-7,
        })
    
    # ========================================================================
    # PART 4: Test different loss configurations (6 configs)
    # ========================================================================
    for model in ["smp_unet_resnet34", "smp_unetplusplus_resnet34"]:
        # Dice+BCE loss
        configs.append({
            "name": f"{model}_dice_bce",
            "model": model,
            "lr": 0.0001,
            "epochs": 50,
            "loss": "dice_bce",
            "scheduler": "cosine",
            "scheduler_t0": 15,
            "scheduler_tmult": 2,
            "scheduler_min_lr": 1e-6,
        })
        
        # Focal Tversky with higher gamma (harder focusing)
        configs.append({
            "name": f"{model}_ft_high_gamma",
            "model": model,
            "lr": 0.0001,
            "epochs": 50,
            "loss": "focal_tversky",
            "ft_gamma": 2.0,
            "ft_alpha": [0.7, 0.9],
            "ft_beta": [0.3, 0.1],
            "scheduler": "cosine",
            "scheduler_t0": 15,
            "scheduler_tmult": 2,
            "scheduler_min_lr": 1e-6,
        })
        
        # Focal Tversky with class weights favoring target
        configs.append({
            "name": f"{model}_ft_target_focus",
            "model": model,
            "lr": 0.0001,
            "epochs": 50,
            "loss": "focal_tversky",
            "ft_gamma": 1.5,
            "ft_alpha": [0.5, 0.9],
            "ft_beta": [0.5, 0.1],
            "ft_class_weights": [1.0, 3.0],
            "scheduler": "cosine",
            "scheduler_t0": 15,
            "scheduler_tmult": 2,
            "scheduler_min_lr": 1e-6,
        })
    
    # ========================================================================
    # PART 5: Test different batch sizes (4 configs)
    # ========================================================================
    for batch_size in [4, 16]:
        for model in ["smp_unet_resnet34", "smp_unetplusplus_resnet34"]:
            configs.append({
                "name": f"{model}_bs{batch_size}",
                "model": model,
                "lr": 0.0001,
                "batch_size": batch_size,
                "epochs": 50,
                "loss": "focal_tversky",
                "scheduler": "cosine",
                "scheduler_t0": 15,
                "scheduler_tmult": 2,
                "scheduler_min_lr": 1e-6,
            })
    
    # ========================================================================
    # PART 6: Test different stack depths (4 configs)
    # ========================================================================
    for stack_depth in [3, 7]:
        for model in ["smp_unet_resnet34", "smp_unetplusplus_resnet34"]:
            configs.append({
                "name": f"{model}_stack{stack_depth}",
                "model": model,
                "lr": 0.0001,
                "stack_depth": stack_depth,
                "epochs": 50,
                "loss": "focal_tversky",
                "scheduler": "cosine",
                "scheduler_t0": 15,
                "scheduler_tmult": 2,
                "scheduler_min_lr": 1e-6,
            })
    
    # ========================================================================
    # PART 7: Combined best practices (2 configs) - longer training
    # ========================================================================
    configs.append({
        "name": "best_combo_unet_resnet34",
        "model": "smp_unet_resnet34",
        "lr": 0.00005,
        "epochs": 100,
        "loss": "focal_tversky",
        "ft_gamma": 1.5,
        "ft_alpha": [0.7, 0.9],
        "ft_beta": [0.3, 0.1],
        "ft_class_weights": [1.0, 2.0],
        "scheduler": "onecycle",
        "scheduler_max_lr_mult": 15.0,
        "scheduler_warmup_pct": 0.3,
    })
    
    configs.append({
        "name": "best_combo_unetplusplus",
        "model": "smp_unetplusplus_resnet34",
        "lr": 0.00005,
        "epochs": 100,
        "loss": "focal_tversky",
        "ft_gamma": 1.5,
        "ft_alpha": [0.7, 0.9],
        "ft_beta": [0.3, 0.1],
        "ft_class_weights": [1.0, 2.0],
        "scheduler": "onecycle",
        "scheduler_max_lr_mult": 15.0,
        "scheduler_warmup_pct": 0.3,
    })
    
    # Limit to max_configs
    return configs[:max_configs]


def main():
    parser = argparse.ArgumentParser(description="Model & Hyperparameter Exploration")
    parser.add_argument("--dry-run", action="store_true", help="Preview without submitting")
    parser.add_argument("--max-jobs", type=int, default=50, help="Maximum number of jobs to submit")
    args = parser.parse_args()
    
    print("="*80)
    print("Model & Hyperparameter Exploration Script")
    print("="*80)
    print(f"Project root: {PROJECT_ROOT}")
    
    # Generate configurations
    configs = generate_exploration_configs(max_configs=args.max_jobs)
    print(f"Total configurations: {len(configs)}")
    
    # Load and display CHECKPOINT_DIR from .env
    env_vars = load_env_file()
    checkpoint_dir = env_vars.get('CHECKPOINT_DIR')
    if checkpoint_dir:
        print(f"Checkpoint Dir: {checkpoint_dir} (from .env)")
    else:
        print(f"Checkpoint Dir: (not set in .env, will use default)")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No jobs will be submitted ***\n")
    
    print("="*80)
    
    # Track all runs
    runs = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Submit each configuration
    for idx, config in enumerate(configs, 1):
        print(f"\n[{idx}/{len(configs)}] Processing: {config.get('name', 'unnamed')}")
        
        # Remove 'name' key as it's not a training parameter
        config_overrides = {k: v for k, v in config.items() if k != 'name'}
        run_name = config.get('name', 'unnamed')
        
        # Submit the job
        job_id = submit_job(config_overrides, dry_run=args.dry_run)
        
        run_info = {
            "job_id": job_id,
            "run_name": run_name,
            "config": config_overrides,
            "timestamp": datetime.now().isoformat(),
            "status": "submitted" if job_id else "failed"
        }
        runs.append(run_info)
    
    # Save results to YAML file (use scratch space to avoid quota issues)
    checkpoint_dir = env_vars.get('CHECKPOINT_DIR', str(PROJECT_ROOT))
    output_file = Path(checkpoint_dir) / f"exploration_runs_{timestamp}.yaml"
    output_data = {
        "timestamp": timestamp,
        "total_runs": len(runs),
        "successful_submissions": sum(1 for r in runs if r["job_id"] is not None),
        "failed_submissions": sum(1 for r in runs if r["job_id"] is None),
        "dry_run": args.dry_run,
        "runs": runs
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total runs: {len(runs)}")
    print(f"Successful submissions: {sum(1 for r in runs if r['job_id'] is not None)}")
    print(f"Failed submissions: {sum(1 for r in runs if r['job_id'] is None)}")
    print(f"\nResults saved to: {output_file}")
    print("="*80)
    
    # Print job IDs for quick reference
    if any(r["job_id"] and r["job_id"] != "dry_run" for r in runs):
        print("\nJob IDs:")
        for run in runs:
            if run["job_id"] and run["job_id"] != "dry_run":
                print(f"  {run['job_id']:>8} - {run['run_name']}")
    
    # Print configuration summary
    print("\n" + "="*80)
    print("Configuration Categories:")
    print("="*80)
    print("  - Model architectures: 13 configs")
    print("  - Learning rate variations: 12 configs")
    print("  - Scheduler variations: 9 configs")
    print("  - Loss function variations: 6 configs")
    print("  - Batch size variations: 4 configs")
    print("  - Stack depth variations: 4 configs")
    print("  - Best combinations: 2 configs")
    print(f"  Total: {len(configs)} configs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
