#!/usr/bin/env python3
"""
Train 2.5D Multi-Class Segmentation Model

This training script trains on ALL masks simultaneously:
- Images from data/processed/
- All masks from data/processed_seg/ (prostate, target1, target2)

Output: Multi-class segmentation model
  - Class 0: Prostate (Red)
  - Class 1: Target (Green) [combined Target1 + Target2]

Features:
  - Advanced learning rate schedulers (OneCycle, Cosine, ReduceLROnPlateau, etc.)
  - Validation visualizations logged to Aim
  - Comprehensive experiment tracking with Aim and Weights & Biases (wandb)
  - Checkpoint saving and resuming

Usage:
    # Use config file (recommended)
    python service/train.py --config config.yaml

    # Override specific parameters
    python service/train.py --config config.yaml --epochs 100 --batch_size 16

    # Continue from checkpoint (use job-specific path)
    python service/train.py --config config.yaml --resume checkpoints/755384/model_epoch_25.pt

    # Pure CLI (no config file)
    python service/train.py --manifest data/processed/class2/manifest.csv --epochs 50 --scheduler onecycle

    # Enable wandb logging
    python service/train.py --config config.yaml --wandb --wandb_project your_project_name

Note:
    For wandb, ensure WANDB_API_KEY is set:
    export WANDB_API_KEY="<your_wandb_api_key>"
    Find your key at: https://wandb.ai/authorize
"""

import os
import sys
from pathlib import Path

# Add project root to path (so we can import tools package)
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from logger import ExperimentLogger, get_run_name
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.dataset.dataset_2d5_multiclass import (
    MRI25DMultiClassDataset,
    create_multiclass_dataloader,
)

# Import models and metrics from separate modules
from service.models import SimpleUNet, DiceLoss, DiceBCELoss, FocalTverskyLoss
from service.metrics import (
    compute_dice_score,
    compute_precision,
    compute_recall,
    initialize_monai_metrics,
    MONAI_AVAILABLE,
)

# Optional wandb/weave integration
try:
    import wandb
    import weave

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    weave = None


# ===========================
# Learning Rate Schedulers
# ===========================


def get_scheduler(optimizer, scheduler_type, total_epochs, steps_per_epoch, args):
    """
    Create learning rate scheduler based on type

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        total_epochs: Total number of training epochs
        steps_per_epoch: Number of batches per epoch
        args: Command line arguments

    Returns:
        scheduler: Learning rate scheduler or None
        step_on: 'epoch' or 'batch' - when to step the scheduler
    """
    if scheduler_type == "none":
        return None, None

    elif scheduler_type == "reduce_on_plateau":
        # Reduces LR when validation metric plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # For Dice score (higher is better)
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=1e-7,
        )
        return scheduler, "epoch"

    elif scheduler_type == "cosine":
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.scheduler_t0,  # Number of epochs for first restart
            T_mult=args.scheduler_tmult,  # Multiply T_0 after each restart
            eta_min=args.scheduler_min_lr,
            verbose=False,
        )
        return scheduler, "epoch"

    elif scheduler_type == "cosine_simple":
        # Simple cosine annealing (no restarts)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=args.scheduler_min_lr, verbose=False
        )
        return scheduler, "epoch"

    elif scheduler_type == "onecycle":
        # One cycle learning rate policy (very effective for fast training)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr * args.scheduler_max_lr_mult,
            total_steps=total_epochs * steps_per_epoch,
            pct_start=args.scheduler_warmup_pct,
            anneal_strategy="cos",
            div_factor=args.scheduler_div_factor,
            final_div_factor=args.scheduler_final_div_factor,
            verbose=False,
        )
        return scheduler, "batch"

    elif scheduler_type == "step":
        # Step decay
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_factor,
            verbose=False,
        )
        return scheduler, "epoch"

    elif scheduler_type == "exponential":
        # Exponential decay
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.scheduler_factor, verbose=False
        )
        return scheduler, "epoch"

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_current_lr(optimizer):
    """Get current learning rate from optimizer"""
    return optimizer.param_groups[0]["lr"]


# ===========================
# Visualization
# ===========================


def create_prediction_visualization(images, masks_gt, masks_pred, num_samples=4):
    """
    Create visualization of predictions vs ground truth

    Args:
        images: (B, C, H, W) - input images
        masks_gt: (B, 3, H, W) - ground truth masks
        masks_pred: (B, 3, H, W) - predicted masks (after sigmoid)
        num_samples: Number of samples to visualize

    Returns:
        Dictionary of matplotlib figures
    """
    batch_size = min(images.shape[0], num_samples)

    # Class colors (RGB)
    colors = {
        0: [1.0, 0.0, 0.0],  # Prostate - Red
        1: [0.0, 1.0, 0.0],  # Target - Green (combined Target1 + Target2)
    }

    class_names = ["Prostate", "Target"]

    visualizations = {}

    for idx in range(batch_size):
        # Get middle slice from stack
        img = images[idx, images.shape[1] // 2].cpu().numpy()
        gt = masks_gt[idx].cpu().numpy()
        pred = masks_pred[idx].cpu().numpy()

        # Normalize image for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Create figure with 4 columns: Image, GT, Pred, Overlay
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Ground Truth
        # Image
        axes[0, 0].imshow(img, cmap="gray")
        axes[0, 0].set_title("Input Image", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        # Ground truth overlay
        axes[0, 1].imshow(img, cmap="gray")
        gt_overlay = np.zeros((*img.shape, 3))
        for c in range(3):
            for channel in range(3):
                gt_overlay[:, :, channel] += gt[c] * colors[c][channel]
        axes[0, 1].imshow(gt_overlay, alpha=0.5)
        axes[0, 1].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        # Individual GT masks
        axes[0, 2].imshow(img, cmap="gray")
        axes[0, 2].set_title("GT Classes", fontsize=12, fontweight="bold")
        for c in range(2):
            if gt[c].max() > 0:
                axes[0, 2].contour(
                    gt[c], levels=[0.5], colors=[colors[c]], linewidths=2
                )
        axes[0, 2].axis("off")

        # Row 2: Predictions
        # Image (repeated for alignment)
        axes[1, 0].imshow(img, cmap="gray")
        axes[1, 0].set_title("Input Image", fontsize=12, fontweight="bold")
        axes[1, 0].axis("off")

        # Prediction overlay
        axes[1, 1].imshow(img, cmap="gray")
        pred_overlay = np.zeros((*img.shape, 3))
        pred_binary = (pred > 0.5).astype(float)
        for c in range(3):
            for channel in range(3):
                pred_overlay[:, :, channel] += pred_binary[c] * colors[c][channel]
        axes[1, 1].imshow(pred_overlay, alpha=0.5)
        axes[1, 1].set_title("Prediction", fontsize=12, fontweight="bold")
        axes[1, 1].axis("off")

        # Individual prediction masks
        axes[1, 2].imshow(img, cmap="gray")
        axes[1, 2].set_title("Pred Classes", fontsize=12, fontweight="bold")
        for c in range(2):
            if pred_binary[c].max() > 0:
                axes[1, 2].contour(
                    pred_binary[c], levels=[0.5], colors=[colors[c]], linewidths=2
                )
        axes[1, 2].axis("off")

        # Add legend
        legend_elements = [
            mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(2)
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=3,
            fontsize=11,
            frameon=True,
        )

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        visualizations[f"sample_{idx}"] = fig

    return visualizations


# ===========================
# Training
# ===========================


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    step_on="epoch",
    logger=None,
    epoch=0,
    wandb_run=None,
    log_batch_every=10,
):
    """Train for one epoch with batch-level logging"""
    model.train()
    total_loss = 0
    total_dice = 0
    # Per-class accumulators
    total_dice_per_class = [0.0, 0.0]  # [prostate, target]
    total_precision_per_class = [0.0, 0.0]  # [prostate, target]
    total_recall_per_class = [0.0, 0.0]  # [prostate, target]
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Step scheduler if batch-level
        if scheduler is not None and step_on == "batch":
            scheduler.step()

        # Compute metrics
        dice = compute_dice_score(outputs, masks)

        # Compute per-class metrics for training
        pred = torch.sigmoid(outputs)
        # Use configurable threshold (could be made class-specific in future)
        threshold = 0.5
        pred_binary = (pred > threshold).float()

        # Compute per-class dice
        dice_per_class = []
        for c in range(2):  # 2 classes: prostate, target
            pred_c = pred_binary[:, c].reshape(-1)
            target_c = masks[:, c].reshape(-1)
            intersection = (pred_c * target_c).sum()
            dice_c = (2.0 * intersection) / (pred_c.sum() + target_c.sum() + 1e-8)
            dice_per_class.append(dice_c.item())

        # Get per-class precision and recall
        precision_per_class = compute_precision(outputs, masks, return_per_class=True)
        recall_per_class = compute_recall(outputs, masks, return_per_class=True)

        total_loss += loss.item()
        total_dice += dice
        # Accumulate per-class metrics
        for c in range(2):
            total_dice_per_class[c] += dice_per_class[c]
            total_precision_per_class[c] += precision_per_class[c]
            total_recall_per_class[c] += recall_per_class[c]
        num_batches += 1

        current_lr = get_current_lr(optimizer)
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice:.4f}",
                "lr": f"{current_lr:.2e}",
            }
        )

        # Log batch-level metrics
        if logger is not None:
            global_step = epoch * len(dataloader) + batch_idx
            batch_metrics = {
                "batch_loss": loss.item(),
                "batch_dice": dice,
                "learning_rate": current_lr,
            }
            # Add per-class metrics
            class_names = ["prostate", "target"]
            for c, name in enumerate(class_names):
                batch_metrics[f"batch_dice_{name}"] = dice_per_class[c]
                batch_metrics[f"batch_precision_{name}"] = precision_per_class[c]
                batch_metrics[f"batch_recall_{name}"] = recall_per_class[c]

            logger.log_metrics(batch_metrics, step=global_step, context="train_batch")

        # Log batch-level metrics to wandb (every N batches to reduce overhead)
        if wandb_run is not None and batch_idx % log_batch_every == 0:
            global_step = epoch * len(dataloader) + batch_idx
            wandb_metrics = {
                "train/batch_loss": loss.item(),
                "train/batch_dice": dice,
                "learning_rate": current_lr,
            }
            # Add per-class metrics
            class_names = ["prostate", "target"]
            for c, name in enumerate(class_names):
                wandb_metrics[f"train/batch_dice_{name}"] = dice_per_class[c]
                wandb_metrics[f"train/batch_precision_{name}"] = precision_per_class[c]
                wandb_metrics[f"train/batch_recall_{name}"] = recall_per_class[c]

            wandb.log(wandb_metrics, step=global_step)

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    avg_dice_per_class = [d / max(num_batches, 1) for d in total_dice_per_class]
    avg_precision_per_class = [p / max(num_batches, 1) for p in total_precision_per_class]
    avg_recall_per_class = [r / max(num_batches, 1) for r in total_recall_per_class]

    return avg_loss, avg_dice, avg_dice_per_class, avg_precision_per_class, avg_recall_per_class


def validate_epoch(
    model,
    dataloader,
    criterion,
    device,
    dice_metric=None,
    dice_metric_per_class=None,
    post_pred=None,
    post_label=None,
    save_visualizations=False,
    logger=None,
    epoch=0,
    wandb_run=None,
    log_batch_every=10,
    thr_sweep_every=1,
):
    """
    Validate for one epoch with batch-level logging
    
    Args:
        dice_metric: MONAI DiceMetric for mean Dice (if MONAI available)
        dice_metric_per_class: MONAI DiceMetric for per-class Dice (if MONAI available)
        post_pred: MONAI AsDiscrete transform for predictions
        post_label: MONAI AsDiscrete transform for labels
    """
    model.eval()
    total_loss = 0
    total_precision = 0
    total_recall = 0
    # Per-class accumulators
    total_precision_per_class = [0.0, 0.0]  # [prostate, target]
    total_recall_per_class = [0.0, 0.0]  # [prostate, target]
    # Dice accumulators (for fallback when MONAI not available)
    total_dice = 0.0
    total_dice_per_class = [0.0, 0.0]  # [prostate, target]
    num_batches = 0

    # Store first batch for visualization
    vis_images = None
    vis_masks_gt = None
    vis_masks_pred = None

    # Accumulate all predictions and targets for threshold sweeping
    all_probs = []  # Store sigmoid probabilities for threshold sweeping
    all_targets = []  # Store ground truth masks for threshold sweeping

    # Reset MONAI metrics at start of validation
    if dice_metric is not None:
        dice_metric.reset()
    if dice_metric_per_class is not None:
        dice_metric_per_class.reset()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            # Store predictions and targets for threshold sweeping
            probs = torch.sigmoid(outputs)  # Convert logits to probabilities
            all_probs.append(probs.cpu())  # Move to CPU for accumulation
            all_targets.append(masks.cpu())  # Move to CPU for accumulation

            # Compute precision/recall (custom implementation)
            precision = compute_precision(outputs, masks)
            recall = compute_recall(outputs, masks)
            # Get per-class metrics
            precision_per_class = compute_precision(outputs, masks, return_per_class=True)
            recall_per_class = compute_recall(outputs, masks, return_per_class=True)

            # Use MONAI DiceMetric if available
            if dice_metric is not None and post_pred is not None and post_label is not None:
                # Apply sigmoid and threshold
                probs = torch.sigmoid(outputs)
                
                # MONAI expects list of tensors per batch element
                # Each tensor shape: (C, H, W)
                batch_size = probs.shape[0]
                preds_list = []
                targets_list = []
                
                for b in range(batch_size):
                    # Apply post-processing (threshold)
                    pred_b = post_pred(probs[b])  # (C, H, W)
                    target_b = post_label(masks[b])  # (C, H, W)
                    preds_list.append(pred_b)
                    targets_list.append(target_b)
                
                # Update MONAI metrics
                dice_metric(y_pred=preds_list, y=targets_list)
                dice_metric_per_class(y_pred=preds_list, y=targets_list)
                
                # For progress bar, compute batch Dice using custom function
                dice = compute_dice_score(outputs, masks)
            else:
                # Fallback to custom Dice computation
                dice = compute_dice_score(outputs, masks)
                # Also compute per-class Dice for fallback
                pred = torch.sigmoid(outputs)
                pred = (pred > 0.5).float()
                for c in range(2):
                    pred_c = pred[:, c].reshape(-1)
                    target_c = masks[:, c].reshape(-1)
                    intersection = (pred_c * target_c).sum()
                    dice_c = (2.0 * intersection) / (pred_c.sum() + target_c.sum() + 1e-8)
                    total_dice_per_class[c] += dice_c.item()
                total_dice += dice

            total_loss += loss.item()
            total_precision += precision
            total_recall += recall
            for c in range(2):
                total_precision_per_class[c] += precision_per_class[c]
                total_recall_per_class[c] += recall_per_class[c]
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice:.4f}",
                "prec": f"{precision:.4f}",
                "rec": f"{recall:.4f}",
            })

            # Log batch-level metrics
            if logger is not None:
                global_step = epoch * len(dataloader) + batch_idx
                batch_metrics = {
                    "batch_loss": loss.item(),
                    "batch_dice": dice,
                    "batch_precision": precision,
                    "batch_recall": recall,
                }
                # Add per-class metrics
                class_names = ["prostate", "target"]
                for c, name in enumerate(class_names):
                    batch_metrics[f"batch_precision_{name}"] = precision_per_class[c]
                    batch_metrics[f"batch_recall_{name}"] = recall_per_class[c]
                logger.log_metrics(
                    batch_metrics,
                    step=global_step,
                    context="val_batch",
                )
            
            # Log batch-level metrics to wandb (every N batches to reduce overhead)
            if wandb_run is not None and batch_idx % log_batch_every == 0:
                global_step = epoch * len(dataloader) + batch_idx
                wandb_metrics = {
                    "val/batch_loss": loss.item(),
                    "val/batch_dice": dice,
                    "val/batch_precision": precision,
                    "val/batch_recall": recall,
                }
                # Add per-class metrics
                class_names = ["prostate", "target"]
                for c, name in enumerate(class_names):
                    # Compute per-class dice for batch-level logging
                    pred = torch.sigmoid(outputs)
                    pred_binary = (pred > 0.5).float()
                    pred_c = pred_binary[:, c].reshape(-1)
                    target_c = masks[:, c].reshape(-1)
                    intersection = (pred_c * target_c).sum()
                    dice_c = (2.0 * intersection) / (pred_c.sum() + target_c.sum() + 1e-8)
                    wandb_metrics[f"val/batch_dice_{name}"] = dice_c.item()
                    wandb_metrics[f"val/batch_precision_{name}"] = precision_per_class[c]
                    wandb_metrics[f"val/batch_recall_{name}"] = recall_per_class[c]
                wandb.log(wandb_metrics, step=global_step)

            # Save first batch for visualization
            if save_visualizations and batch_idx == 0:
                vis_images = images
                vis_masks_gt = masks
                vis_masks_pred = torch.sigmoid(outputs)

    # Aggregate MONAI Dice metrics
    if dice_metric is not None and dice_metric_per_class is not None:
        mean_dice, not_nans = dice_metric.aggregate()
        per_class_dice, not_nans_per_class = dice_metric_per_class.aggregate()
        
        # Convert to Python scalars/lists
        if isinstance(mean_dice, torch.Tensor):
            avg_dice = mean_dice.item()
        else:
            avg_dice = float(mean_dice)
            
        if isinstance(per_class_dice, torch.Tensor):
            per_class_dice_list = per_class_dice.tolist()
        else:
            per_class_dice_list = list(per_class_dice)
    else:
        # Fallback: use accumulated batch-averaged Dice
        avg_dice = total_dice / max(num_batches, 1) if num_batches > 0 else 0.0
        per_class_dice_list = [d / max(num_batches, 1) for d in total_dice_per_class] if num_batches > 0 else [0.0, 0.0]

    avg_loss = total_loss / max(num_batches, 1)
    avg_precision = total_precision / max(num_batches, 1)
    avg_recall = total_recall / max(num_batches, 1)
    avg_precision_per_class = [p / max(num_batches, 1) for p in total_precision_per_class]
    avg_recall_per_class = [r / max(num_batches, 1) for r in total_recall_per_class]

    # Threshold sweeping for Target1 and Target2 (classes 1 and 2)
    threshold_sweep_results = {}
    if thr_sweep_every > 0 and (epoch + 1) % thr_sweep_every == 0 and all_probs:
        print("  Performing threshold sweep...")

        # Concatenate all accumulated predictions and targets
        all_probs_tensor = torch.cat(all_probs, dim=0)  # (N, 3, H, W)
        all_targets_tensor = torch.cat(all_targets, dim=0)  # (N, 3, H, W)

        # Thresholds to sweep
        thresholds = [round(t, 2) for t in torch.arange(0.1, 0.95, 0.05).tolist()]

        # Results for the target class
        for target_class in [1]:  # Target (combined Target1 + Target2)
            class_name = "target"
            print(f"    Sweeping {class_name}...")

            threshold_results = []

            for threshold in thresholds:
                # Apply threshold to get binary predictions for this class
                pred_binary = (all_probs_tensor[:, target_class] > threshold).float()

                # Flatten predictions and targets for this class
                pred_flat = pred_binary.reshape(-1)
                target_flat = all_targets_tensor[:, target_class].reshape(-1)

                # Compute TP, FP, FN
                tp = (pred_flat * target_flat).sum().item()
                fp = (pred_flat * (1 - target_flat)).sum().item()
                fn = ((1 - pred_flat) * target_flat).sum().item()

                # Compute precision, recall, dice
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

                threshold_results.append({
                    'threshold': threshold,
                    'dice': dice,
                    'precision': precision,
                    'recall': recall,
                    'tp': int(tp),
                    'fp': int(fp),
                    'fn': int(fn),
                })

            threshold_sweep_results[class_name] = threshold_results

        # Find best threshold (max Dice) for the target class
        best_thresholds = {}
        for target_class in [1]:
            class_name = "target"
            results = threshold_sweep_results[class_name]
            best_result = max(results, key=lambda x: x['dice'])
            best_thresholds[class_name] = best_result

        # Log to W&B if available
        if wandb_run is not None:
            try:
                # Log tables
                for target_class in [1]:
                    class_name = "target"
                    results = threshold_sweep_results[class_name]

                    # Only log threshold sweep table every 5 epochs to reduce cache size
                    if (epoch + 1) % 5 == 0:
                        # Create table data
                        table_data = [[epoch + 1, r['threshold'], r['dice'], r['precision'], r['recall'], r['tp'], r['fp'], r['fn']]
                                     for r in results]

                        # Create wandb table
                        table = wandb.Table(
                            data=table_data,
                            columns=["epoch", "threshold", "dice", "precision", "recall", "tp", "fp", "fn"]
                        )

                        # Log table
                        wandb.log({f"val/threshold_sweep_{class_name}": table}, step=epoch)

                # Log plots for the target class
                for target_class in [1]:
                    class_name = "target"
                    results = threshold_sweep_results[class_name]

                    thresholds = [r['threshold'] for r in results]
                    dice_scores = [r['dice'] for r in results]
                    precisions = [r['precision'] for r in results]
                    recalls = [r['recall'] for r in results]

                    # Only create plots every 10 epochs to reduce cache size
                    if (epoch + 1) % 10 == 0:
                        # Dice vs Threshold plot
                        wandb.log({
                            f"plots/{class_name}_dice_vs_thr": wandb.plot.line(
                                wandb.Table(data=[[t, d] for t, d in zip(thresholds, dice_scores)],
                                           columns=["threshold", "dice"]),
                                "threshold", "dice",
                                title=f"Target Dice vs Threshold (Epoch {epoch+1})"
                            )
                        }, step=epoch)

                        # Precision vs Threshold plot
                        wandb.log({
                            f"plots/{class_name}_precision_vs_thr": wandb.plot.line(
                                wandb.Table(data=[[t, p] for t, p in zip(thresholds, precisions)],
                                           columns=["threshold", "precision"]),
                                "threshold", "precision",
                                title=f"Target Precision vs Threshold (Epoch {epoch+1})"
                            )
                        }, step=epoch)

                        # Recall vs Threshold plot
                        wandb.log({
                            f"plots/{class_name}_recall_vs_thr": wandb.plot.line(
                                wandb.Table(data=[[t, r] for t, r in zip(thresholds, recalls)],
                                           columns=["threshold", "recall"]),
                                "threshold", "recall",
                                title=f"Target Recall vs Threshold (Epoch {epoch+1})"
                            )
                        }, step=epoch)

                        # Precision-Recall curve
                        wandb.log({
                            f"plots/{class_name}_precision_recall": wandb.plot.line(
                                wandb.Table(data=[[r, p] for r, p in zip(recalls, precisions)],
                                           columns=["recall", "precision"]),
                                "recall", "precision",
                                title=f"Target Precision-Recall (Epoch {epoch+1})"
                            )
                        }, step=epoch)

                # Log best threshold scalar metrics
                for target_class in [1]:
                    class_name = "target"
                    best = best_thresholds[class_name]
                    wandb.log({
                        f"val/best_thr_{class_name}": best['threshold'],
                        f"val/best_dice_{class_name}": best['dice'],
                        f"val/best_precision_{class_name}": best['precision'],
                        f"val/best_recall_{class_name}": best['recall'],
                    }, step=epoch)

                print("  ✓ Logged threshold sweep results to W&B")

            except Exception as e:
                print(f"  ⚠️  Error logging threshold sweep to W&B: {e}")

    return avg_loss, avg_dice, avg_precision, avg_recall, avg_precision_per_class, avg_recall_per_class, per_class_dice_list, (vis_images, vis_masks_gt, vis_masks_pred)


# ===========================
# Configuration
# ===========================


def get_git_commit_hash():
    """Get current git commit hash for code versioning"""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_config_with_args(args):
    """
    Merge YAML config with command-line args.
    CLI args take precedence over config file.
    """
    if args.config is None:
        return args

    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Get default values from parser
    parser = create_argument_parser()
    defaults = vars(parser.parse_args([]))

    # Merge: config < defaults < CLI args
    for key, value in config.items():
        # Only use config value if CLI arg is at default value
        if hasattr(args, key) and getattr(args, key) == defaults.get(key):
            setattr(args, key, value)
            print(f"  {key}: {value}")

    print()
    return args


def create_argument_parser():
    """Create argument parser (separated for config merging)"""
    parser = argparse.ArgumentParser(
        description="Train 2.5D multi-class segmentation model (prostate + target1 + target2)"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config)",
    )

    # Data arguments
    parser.add_argument(
        "--manifest", type=str, default=None, help="Path to manifest CSV"
    )
    parser.add_argument(
        "--stack_depth", type=int, default=5, help="Number of slices to stack"
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5 * 1e-5, help="Learning rate")
    parser.add_argument(
        "--loss",
        type=str,
        default="dice_bce",
        choices=["dice", "bce", "dice_bce", "focal_tversky"],
        help="Loss function",
    )

    # Focal Tversky loss arguments
    parser.add_argument(
        "--ft_gamma",
        type=float,
        default=1.33,
        help="Focal Tversky gamma parameter",
    )
    parser.add_argument(
        "--ft_alpha",
        type=float,
        nargs=2,
        default=[0.6, 0.8],
        help="Focal Tversky alpha per class [prostate, target]",
    )
    parser.add_argument(
        "--ft_beta",
        type=float,
        nargs=2,
        default=[0.4, 0.2],
        help="Focal Tversky beta per class [prostate, target]",
    )
    parser.add_argument(
        "--ft_class_weights",
        type=float,
        nargs=2,
        default=[1.0, 2.0],
        help="Focal Tversky class weights [prostate, target]",
    )

    # Learning rate scheduler arguments
    parser.add_argument(
        "--scheduler",
        type=str,
        default="reduce_on_plateau",
        choices=[
            "none",
            "reduce_on_plateau",
            "cosine",
            "cosine_simple",
            "onecycle",
            "step",
            "exponential",
        ],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.5,
        help="Factor for ReduceLROnPlateau, StepLR, ExponentialLR",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=5,
        help="Patience for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--scheduler_t0",
        type=int,
        default=10,
        help="T_0 for CosineAnnealingWarmRestarts (epochs for first restart)",
    )
    parser.add_argument(
        "--scheduler_tmult",
        type=int,
        default=2,
        help="T_mult for CosineAnnealingWarmRestarts",
    )
    parser.add_argument(
        "--scheduler_min_lr",
        type=float,
        default=1e-7,
        help="Minimum learning rate for cosine schedulers",
    )
    parser.add_argument(
        "--scheduler_max_lr_mult",
        type=float,
        default=10.0,
        help="Max LR multiplier for OneCycleLR (max_lr = lr * mult)",
    )
    parser.add_argument(
        "--scheduler_warmup_pct",
        type=float,
        default=0.3,
        help="Warmup percentage for OneCycleLR",
    )
    parser.add_argument(
        "--scheduler_div_factor",
        type=float,
        default=25.0,
        help="Initial LR divisor for OneCycleLR",
    )
    parser.add_argument(
        "--scheduler_final_div_factor",
        type=float,
        default=1e4,
        help="Final LR divisor for OneCycleLR",
    )
    parser.add_argument(
        "--scheduler_step_size", type=int, default=10, help="Step size for StepLR"
    )

    # Model arguments
    parser.add_argument(
        "--model", type=str, default="simple_unet", help="Model architecture"
    )

    # Other arguments
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints", help="Output directory"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--vis_every",
        type=int,
        default=5,
        help="Save validation visualizations every N epochs",
    )
    parser.add_argument(
        "--num_vis_samples",
        type=int,
        default=4,
        help="Number of validation samples to visualize",
    )
    parser.add_argument(
        "--thr_sweep_every",
        type=int,
        default=1,
        help="Perform threshold sweep every N epochs (0 to disable)",
    )

    # Wandb arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases (wandb) logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mri-segmentation",
        help="Wandb project name (format: entity/project or just project)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity (username or team name)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (auto-generated if not provided)",
    )

    return parser


# ===========================
# Main
# ===========================


def main():
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Merge with config file if provided
    args = merge_config_with_args(args)

    # Validate required arguments
    if args.manifest is None:
        parser.error("--manifest is required (either via CLI or config file)")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"Training 2.5D Multi-Class Segmentation Model")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Manifest: {args.manifest}")
    print(f"Multi-class: Prostate + Target1 + Target2")
    print(f"Image size: 512x512")
    print(f"Stack depth: {args.stack_depth}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss: {args.loss}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    print("Creating multi-class datasets...")

    train_dataset = MRI25DMultiClassDataset(
        manifest_csv=args.manifest,
        stack_depth=args.stack_depth,
        normalize=True,
        skip_no_masks=True,
        target_size=(512, 512),  # Resize all images and masks to 512x512
    )
    
    # Collect original input sizes from the dataset
    print("Collecting input image sizes...")
    original_sizes = set()
    df = pd.read_csv(args.manifest)
    # Sample up to 100 images to check sizes (to avoid loading all)
    sample_size = min(100, len(df))
    sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    for _, row in sample_df.iterrows():
        img_path = Path(row['image_path'])
        if img_path.exists():
            try:
                img = Image.open(img_path)
                original_sizes.add(img.size)  # PIL size is (width, height)
            except Exception as e:
                pass  # Skip if can't read
    
    # Print input size information
    if original_sizes:
        print(f"  Original input sizes found: {sorted(original_sizes)}")
        print(f"  Target size (after resize): {train_dataset.target_size[1]}x{train_dataset.target_size[0]} (width x height)")
        print(f"  All images will be resized to: {train_dataset.target_size[0]}x{train_dataset.target_size[1]} (height x width)")
    else:
        print("  Could not determine original input sizes")

    # Split train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}\n")

    # Create model
    print("Creating model...")
    model = SimpleUNet(in_channels=args.stack_depth, out_channels=2)  # 2 classes: prostate, target
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}\n")

    # Loss function
    if args.loss == "dice":
        criterion = DiceLoss()
    elif args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "focal_tversky":
        # Ensure FocalTversky parameters match the 2-class setup
        if len(args.ft_alpha) != 2:
            print(f"Warning: ft_alpha has {len(args.ft_alpha)} values, expected 2 for 2-class setup. Using first 2 values.")
            args.ft_alpha = args.ft_alpha[:2]
        if len(args.ft_beta) != 2:
            print(f"Warning: ft_beta has {len(args.ft_beta)} values, expected 2 for 2-class setup. Using first 2 values.")
            args.ft_beta = args.ft_beta[:2]
        if len(args.ft_class_weights) != 2:
            print(f"Warning: ft_class_weights has {len(args.ft_class_weights)} values, expected 2 for 2-class setup. Using first 2 values.")
            args.ft_class_weights = args.ft_class_weights[:2]

        criterion = FocalTverskyLoss(
            alpha=args.ft_alpha,
            beta=args.ft_beta,
            gamma=args.ft_gamma,
            class_weights=args.ft_class_weights,
        )
    else:  # dice_bce
        criterion = DiceBCELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    print("Creating learning rate scheduler...")
    scheduler, step_on = get_scheduler(
        optimizer, args.scheduler, args.epochs, len(train_loader), args
    )

    if scheduler is not None:
        print(f"  Scheduler: {args.scheduler}")
        print(f"  Step on: {step_on}")
        if args.scheduler == "reduce_on_plateau":
            print(
                f"  Factor: {args.scheduler_factor}, Patience: {args.scheduler_patience}"
            )
        elif args.scheduler == "onecycle":
            print(f"  Max LR: {args.lr * args.scheduler_max_lr_mult:.2e}")
            print(f"  Warmup: {args.scheduler_warmup_pct*100:.0f}%")
        elif args.scheduler in ["cosine", "cosine_simple"]:
            print(f"  Min LR: {args.scheduler_min_lr:.2e}")
    else:
        print(f"  Scheduler: None (constant learning rate)")
    print()

    # Get job_id early (used for checkpoint paths and wandb run name)
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    best_checkpoint_path = None  # Track the best checkpoint for final artifact

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            # Try to find the checkpoint in the current job's directory
            job_checkpoint_dir = Path(args.output_dir) / job_id
            alt_resume_path = job_checkpoint_dir / resume_path.name
            if alt_resume_path.exists():
                resume_path = alt_resume_path
                print(f"Found checkpoint in job directory: {resume_path}")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", 0.0)

        # Resume scheduler state if available
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print(f"  Scheduler state restored")
            except Exception as e:
                print(f"  Warning: Could not restore scheduler state: {e}")

        print(f"  Resumed from epoch {start_epoch}")
        print(f"  Best Dice so far: {best_dice:.4f}\n")

    # Prepare hyperparameters dict (shared by Aim and wandb)
    hyperparams = {
        "manifest": str(args.manifest),
        "stack_depth": args.stack_depth,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "loss": args.loss,
        "model": args.model,
        "num_workers": args.num_workers,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "model_params": num_params,
        "resumed_from": str(args.resume) if args.resume else None,
        # Scheduler params
        "scheduler": args.scheduler,
        "scheduler_factor": (
            args.scheduler_factor
            if args.scheduler in ["reduce_on_plateau", "step", "exponential"]
            else None
        ),
        "scheduler_patience": (
            args.scheduler_patience if args.scheduler == "reduce_on_plateau" else None
        ),
        "scheduler_t0": args.scheduler_t0 if args.scheduler == "cosine" else None,
        "scheduler_min_lr": (
            args.scheduler_min_lr
            if args.scheduler in ["cosine", "cosine_simple"]
            else None
        ),
        "scheduler_max_lr_mult": (
            args.scheduler_max_lr_mult if args.scheduler == "onecycle" else None
        ),
        # Focal Tversky loss params
        "ft_gamma": args.ft_gamma if args.loss == "focal_tversky" else None,
        "ft_alpha": args.ft_alpha if args.loss == "focal_tversky" else None,
        "ft_beta": args.ft_beta if args.loss == "focal_tversky" else None,
        "ft_class_weights": args.ft_class_weights if args.loss == "focal_tversky" else None,
    }

    # Initialize Aim logger
    logger = ExperimentLogger(
        experiment_name="multiclass_segmentation",
        hyperparams=hyperparams,
        run_name=get_run_name(args),
    )

    # Log system info
    logger.log_system_info()

    # Initialize wandb/weave if enabled
    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            print(
                "\n⚠️  wandb/weave not installed. Install with: pip install wandb weave"
            )
            print("   wandb logging will be disabled.\n")
        elif not os.environ.get("WANDB_API_KEY"):
            print("\n⚠️  WANDB_API_KEY not set.")
            print("   Set it with: export WANDB_API_KEY='<your_key>'")
            print("   Find your key at: https://wandb.ai/authorize")
            print("   wandb logging will be disabled.\n")
        else:
            try:
                # Check if we're in offline mode (for HPC clusters without internet)
                wandb_mode = os.environ.get("WANDB_MODE", "online")
                
                # Initialize weave for tracking (skip if offline)
                weave_project = args.wandb_project
                if args.wandb_entity:
                    weave_project = f"{args.wandb_entity}/{args.wandb_project}"
                
                if wandb_mode != "offline":
                    try:
                        weave.init(weave_project)
                    except Exception as weave_error:
                        print(f"⚠️  Weave initialization failed (non-critical): {weave_error}")

                # Initialize wandb run - use job_id as run name
                wandb_run_name = f"{job_id}_{args.loss}_lr:{args.lr}_epochs:{args.epochs}_datetime:{datetime.now().strftime('%m%d_%H%M')}"  # Use job_id as run name
                
                # Add git commit to hyperparams if available
                git_commit = get_git_commit_hash()
                if git_commit:
                    hyperparams["git_commit"] = git_commit
                
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=wandb_run_name,
                    config=hyperparams,
                    mode=wandb_mode,  # Respect WANDB_MODE environment variable
                    reinit=True,
                )
                
                # Save config.yaml as artifact if it exists (will be saved later after resolved config is created)
                # This is handled after output_dir is created

                print(f"\n✓ Wandb/Weave logging initialized")
                print(f"  Project: {weave_project}")
                print(f"  Run: {wandb_run_name}")
                print(f"  URL: {wandb_run.get_url()}")
                print()
            except Exception as e:
                print(f"\n⚠️  Failed to initialize wandb: {e}")
                print("   wandb logging will be disabled.\n")
                wandb_run = None

    # Setup output directory with job ID
    output_dir = Path(args.output_dir) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get git commit hash for code versioning
    git_commit = get_git_commit_hash()
    if git_commit:
        print(f"Code version (git commit): {git_commit}")
    else:
        print("⚠️  Could not determine git commit hash")

    # Save resolved config (after all overrides) as YAML
    resolved_config_path = output_dir / "config_resolved.yaml"
    resolved_config = vars(args)
    # Add metadata
    resolved_config["_metadata"] = {
        "job_id": job_id,
        "git_commit": git_commit,
        "timestamp": datetime.now().isoformat(),
    }
    with open(resolved_config_path, "w") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Saved resolved config: {resolved_config_path}")

    # Save original config.yaml to checkpoint directory if it exists
    if args.config and Path(args.config).exists():
        config_artifact_path = output_dir / "config.yaml"
        import shutil
        shutil.copy2(args.config, config_artifact_path)
        print(f"✓ Saved original config: {config_artifact_path}")
        
        # Also save to wandb artifacts
        if wandb_run is not None:
            try:
                artifact = wandb.Artifact(
                    name="config",
                    type="config",
                    description="Training configuration file",
                )
                artifact.add_file(str(args.config))
                wandb_run.log_artifact(artifact)
                print(f"  ✓ Saved config.yaml to wandb artifacts")
            except Exception as artifact_error:
                print(f"  ⚠️  Could not save config to wandb artifacts: {artifact_error}")

    print(f"Checkpoints will be saved to: {output_dir}")
    print("Only the best model will be saved to conserve space.\n")

    # Initialize MONAI metrics for validation
    print("Initializing metrics...")
    dice_metric, dice_metric_per_class, post_pred, post_label = initialize_monai_metrics(num_classes=2)
    
    if dice_metric is not None:
        print("  ✓ MONAI DiceMetric initialized (mean and per-class)")
    else:
        print("  ⚠️  Using custom Dice computation (MONAI not available)")

    # Training loop
    print("Starting training...\n")

    for epoch in range(start_epoch, args.epochs):
        current_lr = get_current_lr(optimizer)
        print(f"Epoch {epoch+1}/{args.epochs} - LR: {current_lr:.2e}")
        print("-" * 80)

        # Train
        train_loss, train_dice, train_dice_per_class, train_precision_per_class, train_recall_per_class = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler,
            step_on,
            logger=logger,
            epoch=epoch,
            wandb_run=wandb_run,
            log_batch_every=10,  # Log every 10 batches to wandb
        )
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")

        # Log epoch-level training metrics
        train_metrics = {
            "loss": train_loss,
            "dice": train_dice,
            "learning_rate": current_lr,
        }
        # Add per-class metrics
        class_names = ["prostate", "target"]
        for c, name in enumerate(class_names):
            train_metrics[f"dice_{name}"] = train_dice_per_class[c]
            train_metrics[f"precision_{name}"] = train_precision_per_class[c]
            train_metrics[f"recall_{name}"] = train_recall_per_class[c]
        logger.log_metrics(train_metrics, step=epoch, context="train")

        # Log to wandb
        if wandb_run is not None:
            wandb_metrics = {
                "train/loss": train_loss,
                "train/dice": train_dice,
                "learning_rate": current_lr,
                "epoch": epoch,
            }
            # Add per-class metrics
            for c, name in enumerate(class_names):
                wandb_metrics[f"train/dice_{name}"] = train_dice_per_class[c]
                wandb_metrics[f"train/precision_{name}"] = train_precision_per_class[c]
                wandb_metrics[f"train/recall_{name}"] = train_recall_per_class[c]
            wandb.log(wandb_metrics, step=epoch)

        # Validate
        should_visualize = (epoch + 1) % args.vis_every == 0 or epoch == start_epoch
        val_loss, val_dice, val_precision, val_recall, val_precision_per_class, val_recall_per_class, val_dice_per_class, vis_data = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            dice_metric=dice_metric,
            dice_metric_per_class=dice_metric_per_class,
            post_pred=post_pred,
            post_label=post_label,
            save_visualizations=should_visualize,
            logger=logger,
            epoch=epoch,
            wandb_run=wandb_run,
            log_batch_every=10,  # Log every 10 batches to wandb
            thr_sweep_every=args.thr_sweep_every,
        )
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        print(f"      - Dice per class: Prostate={val_dice_per_class[0]:.4f}, Target={val_dice_per_class[1]:.4f}")
        print(f"      - Precision per class: Prostate={val_precision_per_class[0]:.4f}, Target={val_precision_per_class[1]:.4f}")
        print(f"      - Recall per class: Prostate={val_recall_per_class[0]:.4f}, Target={val_recall_per_class[1]:.4f}")

        # Log epoch-level validation metrics
        val_metrics = {
            "loss": val_loss,
            "dice": val_dice,
            "precision": val_precision,
            "recall": val_recall,
        }
        # Add per-class metrics
        class_names = ["prostate", "target"]
        for c, name in enumerate(class_names):
            val_metrics[f"dice_{name}"] = val_dice_per_class[c]
            val_metrics[f"precision_{name}"] = val_precision_per_class[c]
            val_metrics[f"recall_{name}"] = val_recall_per_class[c]
        logger.log_metrics(val_metrics, step=epoch, context="val")

        # Log validation metrics to wandb
        if wandb_run is not None:
            wandb_metrics = {
                "val/loss": val_loss,
                "val/dice": val_dice,
                "val/precision": val_precision,
                "val/recall": val_recall,
                "epoch": epoch,
            }
            # Add per-class metrics
            for c, name in enumerate(class_names):
                wandb_metrics[f"val/dice_{name}"] = val_dice_per_class[c]
                wandb_metrics[f"val/precision_{name}"] = val_precision_per_class[c]
                wandb_metrics[f"val/recall_{name}"] = val_recall_per_class[c]
            wandb.log(wandb_metrics, step=epoch)

        # Log visualizations if generated
        if should_visualize and vis_data[0] is not None:
            print("  Generating validation visualizations...")
            try:
                vis_images, vis_masks_gt, vis_masks_pred = vis_data
                visualizations = create_prediction_visualization(
                    vis_images,
                    vis_masks_gt,
                    vis_masks_pred,
                    num_samples=args.num_vis_samples,
                )

                # Convert figures to images and log
                wandb_images = {}
                for name, fig in visualizations.items():
                    # Convert figure to numpy array
                    fig.canvas.draw()
                    # Use buffer_rgba() for newer matplotlib versions, fallback to tostring_rgb()
                    try:
                        buffer = fig.canvas.buffer_rgba()
                        img_array = np.asarray(buffer)
                    except AttributeError:
                        # Fallback for older matplotlib versions
                        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        img_array = img_array.reshape(
                            fig.canvas.get_width_height()[::-1] + (3,)
                        )
                    else:
                        # buffer_rgba() returns RGBA, convert to RGB
                        img_array = img_array[:, :, :3]

                    logger.log_images(
                        {name: img_array}, step=epoch, context="val_predictions"
                    )

                    # Store for wandb logging
                    if wandb_run is not None:
                        wandb_images[f"predictions/{name}"] = wandb.Image(
                            img_array, caption=f"Epoch {epoch+1} - {name}"
                        )

                    # Close figure to free memory
                    plt.close(fig)

                # Log images to wandb
                if wandb_run is not None and wandb_images:
                    wandb.log(wandb_images, step=epoch)

                print(f"  ✓ Logged {len(visualizations)} validation visualizations")
            except Exception as e:
                print(f"  ⚠️  Error creating visualizations: {e}")

        # Step scheduler if epoch-level
        if scheduler is not None and step_on == "epoch":
            if args.scheduler == "reduce_on_plateau":
                # ReduceLROnPlateau needs the metric
                scheduler.step(val_dice)
            else:
                scheduler.step()

            # Log new learning rate after scheduler step
            new_lr = get_current_lr(optimizer)
            if new_lr != current_lr:
                print(f"Learning rate changed: {current_lr:.2e} -> {new_lr:.2e}")

        # Save checkpoint (only best model to save space)
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice

            # Delete previous best model(s) for this job to keep only one best model
            previous_best_pattern = f"model_best_{job_id}_*.pt"
            previous_best_files = list(output_dir.glob(previous_best_pattern))
            for prev_file in previous_best_files:
                try:
                    prev_file.unlink()
                    print(f"  Deleted previous best model: {prev_file.name}")
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not delete previous best model {prev_file.name}: {e}")

            # Get git commit for checkpoint
            git_commit = get_git_commit_hash()
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_dice_per_class": val_dice_per_class,  # [prostate, target]
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_precision_per_class": val_precision_per_class,
                "val_recall_per_class": val_recall_per_class,
                "best_dice": best_dice,
                "args": vars(args),
                "job_id": job_id,
                "git_commit": git_commit,
                "timestamp": datetime.now().isoformat(),
            }

            # Save scheduler state if available
            if scheduler is not None:
                try:
                    scheduler_state = scheduler.state_dict()
                    checkpoint["scheduler_state_dict"] = scheduler_state
                except Exception as e:
                    print(f"⚠️  Warning: Could not save scheduler state: {e}")
                    # Continue without scheduler state

            checkpoint_path = output_dir / f"model_best_{job_id}_{epoch+1}.pt"
            try:
                torch.save(checkpoint, checkpoint_path)
                print(f"✓ Saved best model: {checkpoint_path} (Dice: {val_dice:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f})")
            except Exception as e:
                print(f"⚠️  Failed to save checkpoint: {e}")
                # Clean up any partially written file
                if checkpoint_path.exists():
                    try:
                        checkpoint_path.unlink()
                        print(f"  Removed corrupted checkpoint file: {checkpoint_path}")
                    except Exception as cleanup_e:
                        print(f"  Warning: Could not remove corrupted file: {cleanup_e}")
                raise  # Re-raise the exception to stop training

            # Log best model
            best_model_metrics = {
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_precision": val_precision,
                "val_recall": val_recall,
            }
            # Add per-class metrics
            class_names = ["prostate", "target"]
            for c, name in enumerate(class_names):
                best_model_metrics[f"val_dice_{name}"] = val_dice_per_class[c]
                best_model_metrics[f"val_precision_{name}"] = val_precision_per_class[c]
                best_model_metrics[f"val_recall_{name}"] = val_recall_per_class[c]
            logger.log_best_model(best_model_metrics, epoch=epoch)

            # Log best metrics to wandb
            if wandb_run is not None:
                wandb.run.summary["best_val_dice"] = val_dice
                wandb.run.summary["best_val_loss"] = val_loss
                wandb.run.summary["best_val_precision"] = val_precision
                wandb.run.summary["best_val_recall"] = val_recall
                # Add per-class metrics
                for c, name in enumerate(class_names):  # class_names is ["prostate", "target"]
                    wandb.run.summary[f"best_val_dice_{name}"] = val_dice_per_class[c]
                    wandb.run.summary[f"best_val_precision_{name}"] = val_precision_per_class[c]
                    wandb.run.summary[f"best_val_recall_{name}"] = val_recall_per_class[c]
                wandb.run.summary["best_epoch"] = epoch + 1
                # Store the best checkpoint path for final artifact logging
                best_checkpoint_path = checkpoint_path

        print()

    # Save final best model artifact to wandb (only once at the end)
    if wandb_run is not None and best_checkpoint_path is not None:
        try:
            artifact = wandb.Artifact(
                name=f"model-best-{job_id}",
                type="model",
                description=f"Best model from job {job_id} with final val_dice={best_dice:.4f}",
            )
            artifact.add_file(str(best_checkpoint_path))
            wandb_run.log_artifact(artifact)
            print(f"✓ Saved final best model artifact: {best_checkpoint_path}")
        except Exception as artifact_error:
            print(f"⚠️  Could not log final model artifact (non-critical): {artifact_error}")

    # Close logger
    logger.close()

    # Close wandb
    if wandb_run is not None:
        wandb.finish()
        print("✓ Wandb logging closed")

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
