#!/usr/bin/env python3
"""
Experiment Tracking with Aim

This module provides a simple wrapper around Aim for tracking training experiments.

Usage:
    from logger import ExperimentLogger

    # Initialize logger
    logger = ExperimentLogger(
        experiment_name="multiclass_segmentation",
        hyperparams={
            "batch_size": 8,
            "lr": 1e-5,
            "loss": "dice_bce"
        }
    )

    # Log metrics during training
    for epoch in range(num_epochs):
        # Training
        train_loss, train_dice = train_epoch(...)
        logger.log_metrics(
            {"loss": train_loss, "dice": train_dice},
            step=epoch,
            context="train"
        )

        # Validation
        val_loss, val_dice = validate_epoch(...)
        logger.log_metrics(
            {"loss": val_loss, "dice": val_dice},
            step=epoch,
            context="val"
        )

    # Finish logging
    logger.close()
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch


class ExperimentLogger:
    """
    Wrapper around Aim for experiment tracking

    Tracks:
    - Hyperparameters
    - Training/validation metrics (loss, dice, etc.)
    - Model checkpoints
    - System info
    """

    def __init__(
        self,
        experiment_name: str = "mri_segmentation",
        hyperparams: Optional[Dict[str, Any]] = None,
        repo_path: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Initialize Aim logger

        Args:
            experiment_name: Name of the experiment
            hyperparams: Dictionary of hyperparameters to track
            repo_path: Path to Aim repository (default: .aim in project root)
            run_name: Optional name for this run
        """
        try:
            from aim import Run

            self.aim_available = True
        except ImportError:
            print("\n⚠️  Aim not installed. Install with: pip install aim")
            print("   Logging will be disabled.\n")
            self.aim_available = False
            return

        # Setup Aim repository path
        if repo_path is None:
            # Default to .aim in project root
            repo_path = str(Path(__file__).parent.parent / ".aim")

        # Initialize Aim Run
        self.run = Run(
            repo=repo_path,
            experiment=experiment_name,
        )

        # Set run name if provided
        if run_name:
            self.run.name = run_name

        # Log hyperparameters
        if hyperparams:
            self.run["hparams"] = hyperparams

        print(f"\n✓ Aim logging initialized")
        print(f"  Experiment: {experiment_name}")
        print(f"  Run hash: {self.run.hash}")
        print(f"  Repository: {repo_path}")
        if run_name:
            print(f"  Run name: {run_name}")
        print()

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        context: str = "train",
        epoch: Optional[int] = None,
    ):
        """
        Log metrics to Aim

        Args:
            metrics: Dictionary of metric name -> value
            step: Step number (usually epoch)
            context: Context tag (e.g., "train", "val")
            epoch: Optional epoch number (if different from step)
        """
        if not self.aim_available:
            return

        try:
            for metric_name, value in metrics.items():
                self.run.track(
                    value,
                    name=metric_name,
                    step=step,
                    context={"subset": context},
                )
        except Exception as e:
            print(f"⚠️  Error logging metrics to Aim: {e}")
            import traceback

            traceback.print_exc()

    def log_hyperparams(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters

        Args:
            hyperparams: Dictionary of hyperparameter name -> value
        """
        if not self.aim_available:
            return

        self.run["hparams"] = hyperparams

    def log_checkpoint(
        self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]
    ):
        """
        Log checkpoint information

        Args:
            checkpoint_path: Path to saved checkpoint
            epoch: Epoch number
            metrics: Dictionary of metrics at this checkpoint
        """
        if not self.aim_available:
            return

        self.run.track(
            {
                "path": checkpoint_path,
                "epoch": epoch,
                **metrics,
            },
            name="checkpoint",
            step=epoch,
            context={"type": "model"},
        )

    def log_best_model(self, metrics: Dict[str, float], epoch: int):
        """
        Log best model metrics

        Args:
            metrics: Dictionary of best metrics
            epoch: Epoch where best model was found
        """
        if not self.aim_available:
            return

        self.run["best_metrics"] = metrics
        self.run["best_epoch"] = epoch

    def log_text(self, text: str, name: str = "note", step: Optional[int] = None):
        """
        Log text/notes

        Args:
            text: Text to log
            name: Name of the text
            step: Optional step number
        """
        if not self.aim_available:
            return

        from aim import Text

        if step is not None:
            self.run.track(Text(text), name=name, step=step)
        else:
            self.run[name] = text

    def log_images(
        self,
        images: Dict[str, Any],
        step: int,
        context: str = "predictions",
    ):
        """
        Log images to Aim

        Args:
            images: Dictionary of image name -> image (PIL Image, numpy array, or torch tensor)
            step: Step number
            context: Context tag
        """
        if not self.aim_available:
            return

        from aim import Image
        import numpy as np

        for image_name, image in images.items():
            # Convert torch tensor to numpy if needed
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()

            # Convert to uint8 if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    # Normalize to 0-255
                    image = (
                        (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
                    ).astype(np.uint8)

            self.run.track(
                Image(image),
                name=image_name,
                step=step,
                context={"subset": context},
            )

    def log_system_info(self):
        """Log system information"""
        if not self.aim_available:
            return

        import platform

        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            system_info["cuda_version"] = torch.version.cuda
            system_info["gpu_name"] = torch.cuda.get_device_name(0)
            system_info["gpu_count"] = torch.cuda.device_count()

        self.run["system_info"] = system_info

    def close(self):
        """Close the Aim run"""
        if not self.aim_available:
            return

        self.run.close()
        print("\n✓ Aim logging closed")
        print(f"  View results: aim up")
        print()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def get_run_name(args: Any) -> str:
    """
    Generate a descriptive run name from arguments

    Args:
        args: Argument namespace from argparse

    Returns:
        Descriptive run name
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%m%d_%H%M")

    # Extract key parameters
    manifest_name = (
        Path(args.manifest).parent.name if hasattr(args, "manifest") else "unknown"
    )
    batch_size = args.batch_size if hasattr(args, "batch_size") else "unknown"
    lr = args.lr if hasattr(args, "lr") else "unknown"
    loss = args.loss if hasattr(args, "loss") else "unknown"

    return f"{manifest_name}_bs{batch_size}_lr{lr}_{loss}_{timestamp}"
