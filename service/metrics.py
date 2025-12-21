"""
Metrics for 2.5D multi-class segmentation evaluation

Includes:
- Custom Dice, Precision, Recall computation
- MONAI metrics integration (with fallback)
"""

import numpy as np
import torch

# MONAI metrics
try:
    from monai.metrics import DiceMetric
    from monai.transforms import AsDiscrete
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    DiceMetric = None
    AsDiscrete = None


def compute_dice_score(pred, target, threshold=0.5):
    """
    Compute Dice score for multi-class segmentation (fallback if MONAI not available)

    Args:
        pred: (B, C, H, W) - raw logits
        target: (B, C, H, W) - binary masks

    Returns:
        Mean Dice score across all classes
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    # Compute Dice per class
    dice_scores = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c].reshape(-1)
        target_c = target[:, c].reshape(-1)

        intersection = (pred_c * target_c).sum()
        dice = (2.0 * intersection) / (pred_c.sum() + target_c.sum() + 1e-8)
        dice_scores.append(dice.item())

    # Return mean Dice across classes
    return np.mean(dice_scores)


def compute_precision(pred, target, threshold=0.5, return_per_class=False):
    """
    Compute Precision for multi-class segmentation

    Args:
        pred: (B, C, H, W) - raw logits
        target: (B, C, H, W) - binary masks
        return_per_class: If True, return per-class metrics; if False, return mean

    Returns:
        If return_per_class=False: Mean Precision across all classes
        If return_per_class=True: List of Precision scores per class [prostate, target1, target2]
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    # Compute Precision per class
    precision_scores = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c].reshape(-1)
        target_c = target[:, c].reshape(-1)

        intersection = (pred_c * target_c).sum()
        pred_sum = pred_c.sum()
        precision = intersection / (pred_sum + 1e-8)  # TP / (TP + FP)
        precision_scores.append(precision.item())

    if return_per_class:
        return precision_scores
    else:
        # Return mean Precision across classes
        return np.mean(precision_scores)


def compute_recall(pred, target, threshold=0.5, return_per_class=False):
    """
    Compute Recall (Sensitivity) for multi-class segmentation

    Args:
        pred: (B, C, H, W) - raw logits
        target: (B, C, H, W) - binary masks
        return_per_class: If True, return per-class metrics; if False, return mean

    Returns:
        If return_per_class=False: Mean Recall across all classes
        If return_per_class=True: List of Recall scores per class [prostate, target1, target2]
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    # Compute Recall per class
    recall_scores = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c].reshape(-1)
        target_c = target[:, c].reshape(-1)

        intersection = (pred_c * target_c).sum()
        target_sum = target_c.sum()
        recall = intersection / (target_sum + 1e-8)  # TP / (TP + FN)
        recall_scores.append(recall.item())

    if return_per_class:
        return recall_scores
    else:
        # Return mean Recall across classes
        return np.mean(recall_scores)


def initialize_monai_metrics(num_classes=3):
    """
    Initialize MONAI metrics for validation
    
    Args:
        num_classes: Number of segmentation classes (default: 3)
        
    Returns:
        tuple: (dice_metric, dice_metric_per_class, post_pred, post_label) or None if MONAI unavailable
    """
    if not MONAI_AVAILABLE:
        return None, None, None, None
    
    # Mean Dice metric (macro average over classes)
    dice_metric = DiceMetric(
        include_background=False,  # No background channel in multilabel
        reduction="mean",           # Macro average over classes
        get_not_nans=True,         # Tracks empty-target cases
    )
    
    # Per-class Dice metric
    dice_metric_per_class = DiceMetric(
        include_background=False,
        reduction="none",           # Per-class Dice
        get_not_nans=True,
    )
    
    # Post-processing: sigmoid + threshold
    post_pred = AsDiscrete(threshold=0.5)
    post_label = AsDiscrete(threshold=0.5)
    
    return dice_metric, dice_metric_per_class, post_pred, post_label

