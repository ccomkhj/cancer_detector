#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for MRI 2.5D Segmentation

This script submits multiple SLURM jobs with different hyperparameter configurations.
Each job uses the base config.yaml and overrides specific parameters via CLI arguments.

Usage:
    python scripts/hyper_parameter_tuning.py

The script will:
    1. Submit multiple jobs with different hyperparameter combinations
    2. Track job IDs and their corresponding configurations
    3. Save results to a YAML file (hyperparameter_runs.yaml)

You can customize the hyperparameter search space by modifying the `HYPERPARAMETER_CONFIGS` list.
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file() -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    env_file = PROJECT_ROOT / ".env"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
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


def submit_job(config_overrides: Dict, base_config: Dict) -> Optional[str]:
    """
    Submit a SLURM job with the given configuration overrides.
    
    Args:
        config_overrides: Dictionary of parameters to override in the config
        base_config: Base configuration dictionary
        
    Returns:
        Job ID if successful, None otherwise
    """
    # Build the command to submit the job
    # Call the script directly (not via sbatch) - it will call sbatch internally
    script_path = PROJECT_ROOT / "scripts" / "submit_slurm_wandb.sh"
    
    if not script_path.exists():
        raise FileNotFoundError(f"Submission script not found: {script_path}")
    
    # Make sure script is executable
    if not os.access(script_path, os.X_OK):
        os.chmod(script_path, 0o755)
    
    # Build SLURM options as arguments to pass to the script
    # The script will forward these to sbatch
    sbatch_opts = []
    for opt in SLURM_OPTIONS:
        if isinstance(opt, list):
            if len(opt) == 2:
                # Format as --option=value for sbatch options
                sbatch_opts.append(f"{opt[0]}={opt[1]}")
            elif len(opt) == 1:
                sbatch_opts.append(opt[0])
        else:
            sbatch_opts.append(opt)
    
    # Build CLI arguments from config overrides (training arguments)
    cli_args = []
    
    # Check if --output_dir is already in config_overrides
    has_output_dir = 'output_dir' in config_overrides
    
    # Load CHECKPOINT_DIR from .env if --output_dir not already specified
    if not has_output_dir:
        env_vars = load_env_file()
        checkpoint_dir = env_vars.get('CHECKPOINT_DIR')
        if checkpoint_dir:
            cli_args.extend(['--output_dir', checkpoint_dir])
    
    for key, value in config_overrides.items():
        # Convert config key format (scheduler_factor) to CLI format (--scheduler_factor)
        arg_key = f"--{key}"
        cli_args.extend([arg_key, str(value)])
    
    # Call the script directly - it will handle calling sbatch internally
    # Format: ./scripts/submit_slurm_wandb.sh [SBATCH_OPTS] [TRAINING_ARGS]
    script_path_str = str(script_path.resolve())
    cmd = [script_path_str] + sbatch_opts + cli_args
    
    print(f"\n{'='*80}")
    print(f"Submitting job with config:")
    for key, value in config_overrides.items():
        print(f"  {key}: {value}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        # Submit the job and capture output
        # The script will call sbatch internally and output the job ID
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract job ID from sbatch output (stdout or stderr, sbatch outputs to stderr)
        # sbatch output format: "Submitted batch job 12345"
        output = result.stdout.strip() + "\n" + result.stderr.strip()
        match = re.search(r'Submitted batch job (\d+)', output)
        
        if match:
            job_id = match.group(1)
            print(f"✓ Job submitted successfully! Job ID: {job_id}")
            return job_id
        else:
            # If no match, print output for debugging
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


def create_run_name(config_overrides: Dict) -> str:
    """Create a descriptive name for this run based on config overrides."""
    parts = []
    
    # Add scheduler type
    if 'scheduler' in config_overrides:
        parts.append(f"sched-{config_overrides['scheduler']}")
    
    # Add learning rate
    if 'lr' in config_overrides:
        lr = config_overrides['lr']
        parts.append(f"lr-{lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e+'))
    
    # Add batch size
    if 'batch_size' in config_overrides:
        parts.append(f"bs-{config_overrides['batch_size']}")
    
    # Add other important params
    if 'scheduler_max_lr_mult' in config_overrides:
        parts.append(f"maxlr-{config_overrides['scheduler_max_lr_mult']}")
    
    if 'scheduler_patience' in config_overrides:
        parts.append(f"pat-{config_overrides['scheduler_patience']}")
    
    return "_".join(parts) if parts else "default"


# ============================================================================
# SLURM Configuration
# ============================================================================
# SLURM options to pass to sbatch command
# These can be overridden by environment variables or modified here
SLURM_ACCOUNT = os.getenv("SLURM_ACCOUNT", "ebrains-0000006")
SLURM_PARTITION = os.getenv("SLURM_PARTITION", "gpus")
# Add other SLURM options as needed (e.g., --time, --mem, --gres, etc.)
# Format: list of [option, value] pairs or just [option] for flags
SLURM_OPTIONS = [
    ["--account", SLURM_ACCOUNT],
    ["--partition", SLURM_PARTITION],
    # Uncomment and modify as needed:
    # ["--time", "24:00:00"],
    # ["--mem", "64G"],
    # ["--gres", "gpu:1"],
]


# ============================================================================
# Hyperparameter Configurations
# ============================================================================
# Define your hyperparameter search space here.
# Each dictionary represents one job with specific parameter overrides.
# Parameters not specified will use values from config.yaml

HYPERPARAMETER_CONFIGS = [
    # Fast training with OneCycle (recommended in config comments)
    {
        "name": "onecycle_fast",
        "scheduler": "onecycle",
        "lr": 0.0001,
        "scheduler_max_lr_mult": 10.0,
    },
    
    # Conservative stable training (recommended in config comments)
    {
        "name": "reduce_on_plateau_conservative",
        "scheduler": "reduce_on_plateau",
        "scheduler_patience": 5,
        "scheduler_factor": 0.5,
        "lr": 0.0005,
    },
    
    # Experimental with warm restarts (recommended in config comments)
    {
        "name": "cosine_warm_restarts",
        "scheduler": "cosine",
        "scheduler_t0": 10,
        "scheduler_tmult": 2,
        "lr": 0.0005,
    },
    
    # OneCycle with different learning rates
    {
        "name": "onecycle_lr_5e4",
        "scheduler": "onecycle",
        "lr": 0.0005,
        "scheduler_max_lr_mult": 10.0,
    },
    
    {
        "name": "onecycle_lr_1e4",
        "scheduler": "onecycle",
        "lr": 0.0001,
        "scheduler_max_lr_mult": 10.0,
    },
    
    {
        "name": "onecycle_lr_1e5",
        "scheduler": "onecycle",
        "lr": 0.00001,
        "scheduler_max_lr_mult": 10.0,
    },
    
    # ReduceLROnPlateau with different patience values
    {
        "name": "reduce_on_plateau_patience_3",
        "scheduler": "reduce_on_plateau",
        "scheduler_patience": 3,
        "scheduler_factor": 0.5,
        "lr": 0.0005,
    },
    
    {
        "name": "reduce_on_plateau_patience_7",
        "scheduler": "reduce_on_plateau",
        "scheduler_patience": 7,
        "scheduler_factor": 0.5,
        "lr": 0.0005,
    },
    
    # Different batch sizes
    {
        "name": "onecycle_batch_16",
        "scheduler": "onecycle",
        "lr": 0.0001,
        "scheduler_max_lr_mult": 10.0,
    },
    
    {
        "name": "onecycle_batch_4",
        "scheduler": "onecycle",
        "lr": 0.0001,
        "batch_size": 4,
        "scheduler_max_lr_mult": 10.0,
    },
    
    # Cosine simple scheduler
    {
        "name": "cosine_simple",
        "scheduler": "cosine_simple",
        "lr": 0.0005,
    },
    
    # Step scheduler
    {
        "name": "step_lr",
        "scheduler": "step",
        "lr": 0.0005,
        "scheduler_step_size": 10,
    },
]


def main():
    """Main function to submit all hyperparameter tuning jobs."""
    print("="*80)
    print("Hyperparameter Tuning Script")
    print("="*80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Total configurations: {len(HYPERPARAMETER_CONFIGS)}")
    print(f"\nSLURM Configuration:")
    print(f"  Account: {SLURM_ACCOUNT}")
    print(f"  Partition: {SLURM_PARTITION}")
    
    # Load and display CHECKPOINT_DIR from .env
    env_vars = load_env_file()
    checkpoint_dir = env_vars.get('CHECKPOINT_DIR')
    if checkpoint_dir:
        print(f"  Checkpoint Dir: {checkpoint_dir} (from .env)")
    else:
        print(f"  Checkpoint Dir: (not set in .env, will use default)")
    print("="*80)
    
    # Load base config
    try:
        base_config = load_base_config()
        print(f"✓ Loaded base config from config.yaml")
    except Exception as e:
        print(f"✗ Error loading base config: {e}")
        return 1
    
    # Track all runs
    runs = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Submit each configuration
    for idx, config in enumerate(HYPERPARAMETER_CONFIGS, 1):
        print(f"\n[{idx}/{len(HYPERPARAMETER_CONFIGS)}] Processing: {config.get('name', 'unnamed')}")
        
        # Remove 'name' key as it's not a training parameter
        config_overrides = {k: v for k, v in config.items() if k != 'name'}
        run_name = config.get('name', create_run_name(config_overrides))
        
        # Submit the job
        job_id = submit_job(config_overrides, base_config)
        
        if job_id:
            # Store run information
            run_info = {
                "job_id": job_id,
                "run_name": run_name,
                "config": config_overrides,
                "timestamp": datetime.now().isoformat(),
                "status": "submitted"
            }
            runs.append(run_info)
        else:
            print(f"⚠ Failed to submit job for config: {run_name}")
            # Still record it with None job_id
            run_info = {
                "job_id": None,
                "run_name": run_name,
                "config": config_overrides,
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
            runs.append(run_info)
    
    # Save results to YAML file
    output_file = PROJECT_ROOT / f"hyperparameter_runs_{timestamp}.yaml"
    output_data = {
        "timestamp": timestamp,
        "total_runs": len(runs),
        "successful_submissions": sum(1 for r in runs if r["job_id"] is not None),
        "failed_submissions": sum(1 for r in runs if r["job_id"] is None),
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
    if any(r["job_id"] for r in runs):
        print("\nJob IDs:")
        for run in runs:
            if run["job_id"]:
                print(f"  {run['job_id']:>8} - {run['run_name']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
