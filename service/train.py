#!/usr/bin/env python3
"""
Train 2.5D Multi-Class Segmentation Model

This training script trains on ALL masks simultaneously:
- Images from data/processed/
- All masks from data/processed_seg/ (prostate, target1, target2)

Output: Multi-class segmentation model
  - Class 0: Prostate
  - Class 1: Target1
  - Class 2: Target2

Usage:
    # Train on all masks (prostate + target1 + target2)
    python service/train.py --manifest data/processed/class2/manifest.csv

    # Continue from checkpoint
    python service/train.py --manifest data/processed/class2/manifest.csv --resume checkpoints/model_epoch_10.pt

    # Advanced options with learning rate scheduler
    python service/train.py --manifest data/processed/class2/manifest.csv --batch_size 16 --epochs 100 --scheduler onecycle

    # Use cosine annealing with warm restarts
    python service/train.py --manifest data/processed/class2/manifest.csv --scheduler cosine --scheduler_t0 10

    # Use ReduceLROnPlateau (default)
    python service/train.py --manifest data/processed/class2/manifest.csv --scheduler reduce_on_plateau --scheduler_patience 5
"""

import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from tqdm import tqdm

from dataset_2d5_multiclass import MRI25DMultiClassDataset, create_multiclass_dataloader
from logger import ExperimentLogger, get_run_name


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
            verbose=True,
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
# Loss Functions
# ===========================


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - raw logits
            target: (B, 1, H, W) - binary mask
        """
        pred = torch.sigmoid(pred)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()

        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE Loss"""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


# ===========================
# Simple U-Net Model
# ===========================


class SimpleUNet(nn.Module):
    """Simple 2.5D U-Net for multi-class segmentation"""

    def __init__(
        self, in_channels=5, out_channels=3
    ):  # 3 classes: prostate, target1, target2
        super().__init__()

        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        # Bottleneck
        self.bottleneck = self._block(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._block(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, 1)

        self.pool = nn.MaxPool2d(2)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        out = self.out(dec1)
        return out


# ===========================
# Metrics
# ===========================


def compute_dice_score(pred, target, threshold=0.5):
    """
    Compute Dice score for multi-class segmentation

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


# ===========================
# Training
# ===========================


def train_epoch(
    model, dataloader, criterion, optimizer, device, scheduler=None, step_on="epoch"
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
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

        total_loss += loss.item()
        total_dice += dice
        num_batches += 1

        current_lr = get_current_lr(optimizer)
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice:.4f}",
                "lr": f"{current_lr:.2e}",
            }
        )

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)

    return avg_loss, avg_dice


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = compute_dice_score(outputs, masks)

            total_loss += loss.item()
            total_dice += dice
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)

    return avg_loss, avg_dice


# ===========================
# Main
# ===========================


def main():
    parser = argparse.ArgumentParser(
        description="Train 2.5D multi-class segmentation model (prostate + target1 + target2)"
    )

    # Data arguments
    parser.add_argument(
        "--manifest", type=str, required=True, help="Path to manifest CSV"
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
        choices=["dice", "bce", "dice_bce"],
        help="Loss function",
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
        "--save_every", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    args = parser.parse_args()

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
    model = SimpleUNet(in_channels=args.stack_depth, out_channels=3)  # 3 classes
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}\n")

    # Loss function
    if args.loss == "dice":
        criterion = DiceLoss()
    elif args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
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

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
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

    # Initialize Aim logger
    logger = ExperimentLogger(
        experiment_name="multiclass_segmentation",
        hyperparams={
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
                args.scheduler_patience
                if args.scheduler == "reduce_on_plateau"
                else None
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
        },
        run_name=get_run_name(args),
    )

    # Log system info
    logger.log_system_info()

    # Training loop
    print("Starting training...\n")

    for epoch in range(start_epoch, args.epochs):
        current_lr = get_current_lr(optimizer)
        print(f"Epoch {epoch+1}/{args.epochs} - LR: {current_lr:.2e}")
        print("-" * 80)

        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler, step_on
        )
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")

        # Log training metrics
        logger.log_metrics(
            {"loss": train_loss, "dice": train_dice, "learning_rate": current_lr},
            step=epoch,
            context="train",
        )

        # Validate
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        # Log validation metrics
        logger.log_metrics(
            {"loss": val_loss, "dice": val_dice},
            step=epoch,
            context="val",
        )

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

        # Save checkpoint
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice

        if (epoch + 1) % args.save_every == 0 or is_best:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "best_dice": best_dice,
                "args": vars(args),
            }

            # Save scheduler state if available
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            checkpoint_path = output_dir / f"model_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")

            # Log checkpoint
            logger.log_checkpoint(
                str(checkpoint_path),
                epoch=epoch,
                metrics={
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                },
            )

            if is_best:
                best_path = output_dir / "model_best.pt"
                torch.save(checkpoint, best_path)
                print(f"✓ Saved best model: {best_path}")

                # Log best model
                logger.log_best_model(
                    {"val_loss": val_loss, "val_dice": val_dice}, epoch=epoch
                )

        print()

    # Close logger
    logger.close()

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
