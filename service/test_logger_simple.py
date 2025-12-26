#!/usr/bin/env python3
"""
Simple test of logger exactly as train.py uses it
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import ExperimentLogger, get_run_name


# Simulate args
class Args:
    manifest = "data/processed/class4/manifest.csv"
    batch_size = 32
    lr = 5e-5
    loss = "dice_bce"
    model = "simple_unet"
    num_workers = 4
    stack_depth = 5
    epochs = 50
    resume = None


args = Args()

print("=" * 80)
print("Testing logger exactly as train.py uses it")
print("=" * 80)

# Initialize logger exactly as train.py does
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
        "train_samples": 1000,
        "val_samples": 250,
        "model_params": 31044803,
        "resumed_from": str(args.resume) if args.resume else None,
    },
    run_name=get_run_name(args),
)

# Log system info
logger.log_system_info()

print("\n" + "=" * 80)
print("Simulating training loop...")
print("=" * 80 + "\n")

# Simulate 3 epochs
for epoch in range(3):
    train_loss = 1.0 - epoch * 0.1
    train_dice = epoch * 0.2
    val_loss = 1.1 - epoch * 0.12
    val_dice = epoch * 0.18

    print(f"Epoch {epoch+1}/3")
    print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")

    # Log training metrics EXACTLY as train.py does
    logger.log_metrics(
        {"loss": train_loss, "dice": train_dice},
        step=epoch,
        context="train",
    )

    print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

    # Log validation metrics EXACTLY as train.py does
    logger.log_metrics(
        {"loss": val_loss, "dice": val_dice},
        step=epoch,
        context="val",
    )

# Close logger
logger.close()

print("\n" + "=" * 80)
print("✓ Test complete! Now check metrics:")
print("  1. Run: /opt/miniconda3/envs/mri/bin/python service/debug_aim.py")
print("  2. Look for the newest run and check if loss/dice are logged")
print("=" * 80 + "\n")

