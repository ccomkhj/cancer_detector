#!/usr/bin/env python3
"""
Test script for 2.5D segmentation models.

Tests both SMP ResUNet and MONAI SegResNet with the MRI25DDataset
to verify that the data pipeline works correctly with both models.

Usage:
    python tools/test_2d5_models.py
    
    # Test specific model
    python tools/test_2d5_models.py --model smp
    python tools/test_2d5_models.py --model monai
    
    # Test with specific data
    python tools/test_2d5_models.py --manifest data/processed/class1/manifest.csv
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dataset_2d5 import MRI25DDataset, create_dataloader
from tools.transforms_2d5 import get_train_transforms, get_val_transforms


def test_smp_resunet(input_tensor: torch.Tensor, stack_depth: int = 5):
    """
    Test SMP ResUNet model.
    
    Args:
        input_tensor: Input tensor [batch_size, stack_depth, H, W]
        stack_depth: Number of input channels
    
    Returns:
        Output tensor and model
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("Error: segmentation_models_pytorch not installed.")
        print("Install with: pip install segmentation-models-pytorch")
        return None, None
    
    print("\n" + "="*80)
    print("Testing SMP ResUNet")
    print("="*80)
    
    # Create model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # Don't load ImageNet weights for testing
        in_channels=stack_depth,
        classes=1,  # Binary segmentation
        activation=None,  # We'll apply sigmoid separately
    )
    
    print(f"Model created: SMP Unet with ResNet34 encoder")
    print(f"  Input channels: {stack_depth}")
    print(f"  Output classes: 1")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Set to eval mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"\nForward pass successful!")
    print(f"  Input shape:  {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Apply sigmoid for binary segmentation
    output_prob = torch.sigmoid(output)
    print(f"  After sigmoid: [{output_prob.min():.3f}, {output_prob.max():.3f}]")
    
    return output, model


def test_monai_segresnet(input_tensor: torch.Tensor, stack_depth: int = 5):
    """
    Test MONAI SegResNet model.
    
    Args:
        input_tensor: Input tensor [batch_size, stack_depth, H, W]
        stack_depth: Number of input channels
    
    Returns:
        Output tensor and model
    """
    try:
        from monai.networks.nets import SegResNet
    except ImportError:
        print("Error: MONAI not installed.")
        print("Install with: pip install monai")
        return None, None
    
    print("\n" + "="*80)
    print("Testing MONAI SegResNet")
    print("="*80)
    
    # Create model
    model = SegResNet(
        spatial_dims=2,  # 2D mode for 2.5D approach
        in_channels=stack_depth,
        out_channels=1,  # Binary segmentation
        init_filters=32,
        dropout_prob=0.2,
    )
    
    print(f"Model created: MONAI SegResNet")
    print(f"  Spatial dims: 2")
    print(f"  Input channels: {stack_depth}")
    print(f"  Output channels: 1")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Set to eval mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"\nForward pass successful!")
    print(f"  Input shape:  {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Apply sigmoid for binary segmentation
    output_prob = torch.sigmoid(output)
    print(f"  After sigmoid: [{output_prob.min():.3f}, {output_prob.max():.3f}]")
    
    return output, model


def test_dataset_loading(manifest_csv: str, batch_size: int = 4, stack_depth: int = 5):
    """
    Test dataset loading.
    
    Args:
        manifest_csv: Path to manifest CSV
        batch_size: Batch size
        stack_depth: Stack depth
    
    Returns:
        Sample batch (images, masks)
    """
    print("\n" + "="*80)
    print("Testing Dataset Loading")
    print("="*80)
    
    print(f"Loading dataset from: {manifest_csv}")
    
    # Create dataset
    dataset = MRI25DDataset(
        manifest_csv=manifest_csv,
        stack_depth=stack_depth,
        image_size=(256, 256),
        normalize_method="scale",
        has_masks=True,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return None, None
    
    # Create dataloader
    dataloader = create_dataloader(
        manifest_csv=manifest_csv,
        batch_size=batch_size,
        stack_depth=stack_depth,
        image_size=(256, 256),
        shuffle=False,
        num_workers=0,  # Use 0 for testing
    )
    
    print(f"DataLoader created: {len(dataloader)} batches")
    
    # Load one batch
    images, masks = next(iter(dataloader))
    
    print(f"\nBatch loaded:")
    print(f"  Images shape: {images.shape}")
    print(f"  Images dtype: {images.dtype}")
    print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
    
    if masks is not None:
        print(f"  Masks shape: {masks.shape}")
        print(f"  Masks dtype: {masks.dtype}")
        print(f"  Masks range: [{masks.min():.3f}, {masks.max():.3f}]")
        
        # Count samples with masks
        num_with_masks = (masks.sum(dim=[1,2,3]) > 0).sum().item()
        print(f"  Samples with masks: {num_with_masks}/{masks.shape[0]}")
    else:
        print(f"  Masks: None (no masks available in dataset)")
    
    return images, masks


def test_training_step(model, images, masks, loss_fn):
    """
    Test a single training step.
    
    Args:
        model: Model to test
        images: Input images
        masks: Ground truth masks
        loss_fn: Loss function
    
    Returns:
        Loss value
    """
    print("\nTesting training step...")
    
    # Forward pass
    model.train()
    outputs = model(images)
    
    # Compute loss (only for samples with masks)
    if masks is not None:
        # Find samples with masks
        has_mask = masks.sum(dim=[1,2,3]) > 0
        
        if has_mask.any():
            outputs_masked = outputs[has_mask]
            masks_masked = masks[has_mask]
            
            loss = loss_fn(outputs_masked, masks_masked)
            print(f"  Loss: {loss.item():.4f}")
            
            # Backward pass
            loss.backward()
            print(f"  Backward pass successful!")
            
            return loss.item()
        else:
            print("  No samples with masks in batch, skipping loss computation")
            return None
    else:
        print("  No masks available")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test 2.5D segmentation models")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/manifest_all.csv",
        help="Path to manifest CSV file",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["smp", "monai", "both"],
        default="both",
        help="Which model to test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--stack-depth",
        type=int,
        default=5,
        help="Number of slices to stack",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("2.5D Segmentation Model Testing")
    print("="*80)
    
    # Check manifest file
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"\nError: Manifest file not found: {args.manifest}")
        print("Please run dicom_converter.py first to generate the manifest.")
        print("\nExample:")
        print("  python tools/dicom_converter.py --all")
        return 1
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Test dataset loading
    images, masks = test_dataset_loading(
        args.manifest,
        batch_size=args.batch_size,
        stack_depth=args.stack_depth,
    )
    
    if images is None:
        return 1
    
    # Move to device
    images = images.to(device)
    if masks is not None:
        masks = masks.to(device)
    
    # Define loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Test models
    results = {}
    
    if args.model in ["smp", "both"]:
        output_smp, model_smp = test_smp_resunet(images, stack_depth=args.stack_depth)
        if model_smp is not None:
            model_smp = model_smp.to(device)
            loss_smp = test_training_step(model_smp, images, masks, loss_fn)
            results["smp"] = {"success": True, "loss": loss_smp}
        else:
            results["smp"] = {"success": False}
    
    if args.model in ["monai", "both"]:
        output_monai, model_monai = test_monai_segresnet(images, stack_depth=args.stack_depth)
        if model_monai is not None:
            model_monai = model_monai.to(device)
            loss_monai = test_training_step(model_monai, images, masks, loss_fn)
            results["monai"] = {"success": True, "loss": loss_monai}
        else:
            results["monai"] = {"success": False}
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    all_success = True
    for model_name, result in results.items():
        status = "✓ PASSED" if result["success"] else "✗ FAILED"
        print(f"{model_name.upper()}: {status}")
        if result["success"] and result.get("loss") is not None:
            print(f"  Training loss: {result['loss']:.4f}")
        all_success = all_success and result["success"]
    
    if all_success:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("1. The dataset and models are working correctly")
        print("2. You can now implement a full training script")
        print("3. Both SMP ResUNet and MONAI SegResNet accept the same input format")
        print("4. Use tools/dataset_2d5.py for your training pipeline")
        return 0
    else:
        print("\n✗ Some tests failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

