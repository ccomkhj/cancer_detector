#!/usr/bin/env python3
"""
Hyperparameter Exploration Script v2 - Based on Analysis Results

This script focuses on the winning configurations from explore_train.py analysis:
- Best scheduler: onecycle (significantly outperformed cosine and reduce_on_plateau)
- Best learning rate: 5e-05 (with some exploration around it)
- Best loss: focal_tversky
- Longer training (100 epochs) helped achieve best results

Strategy:
1. Test multiple model architectures with the winning hyperparameters
2. Fine-tune learning rate around the best value (5e-05)
3. Explore onecycle scheduler variations
4. Test longer training durations (100-150 epochs)

Usage:
    python scripts/train/explore_train2.py
    python scripts/train/explore_train2.py --dry-run
    python scripts/train/explore_train2.py --max-jobs 10
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


def submit_job(config_overrides: Dict, dry_run: bool = False) -> Optional[str]:
    """
    Submit a SLURM job with the given configuration overrides.

    Args:
        config_overrides: Dictionary of parameters to override in the config
        dry_run: If True, just print the command without submitting

    Returns:
        Job ID if successful, None otherwise
    """
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
        if isinstance(value, list):
            cli_args.append(arg_key)
            cli_args.extend([str(v) for v in value])
        else:
            cli_args.extend([arg_key, str(value)])

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
# Model Configurations - Based on Analysis Results
# ============================================================================
# Analysis showed simple_unet performed well, but we should also test
# other architectures with the winning hyperparameters

MODELS_TO_TEST = [
    # Proven performer
    {"model": "simple_unet", "description": "Simple U-Net (best performer in v1)"},

    # SMP U-Net variants - test with winning hyperparameters
    {"model": "smp_unet_resnet34", "description": "U-Net + ResNet34 (popular choice)"},
    {"model": "smp_unet_resnet50", "description": "U-Net + ResNet50 (deeper)"},
    {"model": "smp_unet_efficientnet-b0", "description": "U-Net + EfficientNet-B0 (efficient)"},

    # U-Net++ variants
    {"model": "smp_unetplusplus_resnet34", "description": "U-Net++ + ResNet34"},

    # DeepLabV3+ (good for multi-scale)
    {"model": "smp_deeplabv3plus_resnet34", "description": "DeepLabV3+ + ResNet34"},
]


def generate_exploration_configs_v2(max_configs: int = 20) -> List[Dict]:
    """
    Generate exploration configurations based on v1 analysis results.

    Key findings from v1:
    - onecycle scheduler: best (mean=0.574, max=0.606)
    - Learning rate 5e-05: best (mean=0.520, max=0.606)
    - focal_tversky loss: dominant
    - Longer training (100 epochs) achieved best results at epoch 36
    """
    configs = []

    # ========================================================================
    # WINNING HYPERPARAMETERS (from analysis)
    # ========================================================================
    winning_hp = {
        "loss": "focal_tversky",
        "scheduler": "onecycle",
        "scheduler_max_lr_mult": 15.0,  # Peak LR = lr * 15
        "scheduler_warmup_pct": 0.3,    # 30% warmup
        "ft_gamma": 1.33,
        "ft_alpha": [0.6, 0.8],
        "ft_beta": [0.4, 0.2],
        "ft_class_weights": [1.0, 2.0],
    }

    # ========================================================================
    # PART 1: Test all model architectures with winning hyperparameters
    # (6 configs) - 100 epochs each
    # ========================================================================
    for model_info in MODELS_TO_TEST:
        config = {
            "name": f"v2_{model_info['model']}_baseline",
            "model": model_info["model"],
            "epochs": 100,
            "lr": 0.00005,  # Best LR from analysis
            **winning_hp,
        }
        configs.append(config)

    # ========================================================================
    # PART 2: Fine-tune learning rate around 5e-05 with best model
    # (4 configs) - Test nearby LRs
    # ========================================================================
    lr_variants = [0.00003, 0.00007, 0.0001, 0.00002]
    for lr in lr_variants:
        config = {
            "name": f"v2_simple_unet_lr{lr:.0e}".replace(".", "_"),
            "model": "simple_unet",
            "epochs": 100,
            "lr": lr,
            **winning_hp,
        }
        configs.append(config)

    # ========================================================================
    # PART 3: Test longer training (150 epochs) for top models
    # (3 configs) - See if more epochs helps
    # ========================================================================
    long_train_models = ["simple_unet", "smp_unet_resnet34", "smp_unetplusplus_resnet34"]
    for model in long_train_models:
        config = {
            "name": f"v2_{model}_150ep",
            "model": model,
            "epochs": 150,
            "lr": 0.00005,
            **winning_hp,
        }
        configs.append(config)

    # ========================================================================
    # PART 4: OneCycle scheduler variations
    # (3 configs) - Fine-tune the scheduler
    # ========================================================================
    onecycle_variants = [
        {"scheduler_max_lr_mult": 10.0, "scheduler_warmup_pct": 0.2, "name_suffix": "oc_mild"},
        {"scheduler_max_lr_mult": 20.0, "scheduler_warmup_pct": 0.3, "name_suffix": "oc_aggressive"},
        {"scheduler_max_lr_mult": 15.0, "scheduler_warmup_pct": 0.4, "name_suffix": "oc_long_warmup"},
    ]
    for variant in onecycle_variants:
        name_suffix = variant.pop("name_suffix")
        config = {
            "name": f"v2_simple_unet_{name_suffix}",
            "model": "simple_unet",
            "epochs": 100,
            "lr": 0.00005,
            "loss": "focal_tversky",
            "scheduler": "onecycle",
            "ft_gamma": 1.33,
            "ft_alpha": [0.6, 0.8],
            "ft_beta": [0.4, 0.2],
            "ft_class_weights": [1.0, 2.0],
            **variant,
        }
        configs.append(config)

    # ========================================================================
    # PART 5: Focal Tversky loss variations
    # (2 configs) - Fine-tune loss parameters
    # ========================================================================
    # Higher gamma = more focus on hard examples
    configs.append({
        "name": "v2_simple_unet_ft_high_focus",
        "model": "simple_unet",
        "epochs": 100,
        "lr": 0.00005,
        "loss": "focal_tversky",
        "scheduler": "onecycle",
        "scheduler_max_lr_mult": 15.0,
        "scheduler_warmup_pct": 0.3,
        "ft_gamma": 2.0,  # Higher gamma
        "ft_alpha": [0.7, 0.9],  # More recall-focused
        "ft_beta": [0.3, 0.1],
        "ft_class_weights": [1.0, 3.0],  # Higher target weight
    })

    # Balanced alpha/beta
    configs.append({
        "name": "v2_simple_unet_ft_balanced",
        "model": "simple_unet",
        "epochs": 100,
        "lr": 0.00005,
        "loss": "focal_tversky",
        "scheduler": "onecycle",
        "scheduler_max_lr_mult": 15.0,
        "scheduler_warmup_pct": 0.3,
        "ft_gamma": 1.5,
        "ft_alpha": [0.5, 0.5],  # Balanced
        "ft_beta": [0.5, 0.5],
        "ft_class_weights": [1.0, 2.0],
    })

    # ========================================================================
    # PART 6: Best combinations with SMP models
    # (2 configs) - Apply winning HP to promising architectures
    # ========================================================================
    configs.append({
        "name": "v2_smp_unet_resnet34_optimized",
        "model": "smp_unet_resnet34",
        "epochs": 150,
        "lr": 0.00003,  # Slightly lower for pretrained
        "loss": "focal_tversky",
        "scheduler": "onecycle",
        "scheduler_max_lr_mult": 10.0,  # More conservative for pretrained
        "scheduler_warmup_pct": 0.3,
        "ft_gamma": 1.5,
        "ft_alpha": [0.6, 0.8],
        "ft_beta": [0.4, 0.2],
        "ft_class_weights": [1.0, 2.0],
    })

    configs.append({
        "name": "v2_smp_unetplusplus_optimized",
        "model": "smp_unetplusplus_resnet34",
        "epochs": 150,
        "lr": 0.00003,
        "loss": "focal_tversky",
        "scheduler": "onecycle",
        "scheduler_max_lr_mult": 10.0,
        "scheduler_warmup_pct": 0.3,
        "ft_gamma": 1.5,
        "ft_alpha": [0.6, 0.8],
        "ft_beta": [0.4, 0.2],
        "ft_class_weights": [1.0, 2.0],
    })

    # Limit to max_configs
    return configs[:max_configs]


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Exploration v2 (Based on Analysis Results)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without submitting")
    parser.add_argument("--max-jobs", type=int, default=20, help="Maximum number of jobs")
    args = parser.parse_args()

    print("="*80)
    print("Hyperparameter Exploration v2 - Based on Analysis Results")
    print("="*80)
    print(f"Project root: {PROJECT_ROOT}")
    print()
    print("Key findings from v1 analysis:")
    print("  - Best scheduler: onecycle (max dice: 0.606)")
    print("  - Best learning rate: 5e-05")
    print("  - Best loss: focal_tversky")
    print("  - Best model: simple_unet (but testing others with winning HP)")
    print("  - Longer training (100 epochs) helped")
    print()

    # Generate configurations
    configs = generate_exploration_configs_v2(max_configs=args.max_jobs)
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

    # Save results to YAML file
    checkpoint_dir = env_vars.get('CHECKPOINT_DIR', str(PROJECT_ROOT))
    output_file = Path(checkpoint_dir) / f"exploration_v2_runs_{timestamp}.yaml"
    output_data = {
        "version": 2,
        "timestamp": timestamp,
        "total_runs": len(runs),
        "successful_submissions": sum(1 for r in runs if r["job_id"] is not None),
        "failed_submissions": sum(1 for r in runs if r["job_id"] is None),
        "dry_run": args.dry_run,
        "analysis_notes": {
            "based_on": "explore_train.py v1 analysis",
            "best_v1_dice": 0.6058,
            "best_v1_config": {
                "model": "simple_unet",
                "scheduler": "onecycle",
                "lr": 0.00005,
                "loss": "focal_tversky",
            }
        },
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
    print("  - Model architectures with winning HP: 6 configs (100 epochs)")
    print("  - Learning rate fine-tuning: 4 configs")
    print("  - Longer training (150 epochs): 3 configs")
    print("  - OneCycle scheduler variations: 3 configs")
    print("  - Focal Tversky loss variations: 2 configs")
    print("  - Optimized SMP models: 2 configs")
    print(f"  Total: {len(configs)} configs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
