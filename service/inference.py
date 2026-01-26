#!/usr/bin/env python3
"""
Inference Script for 2.5D MRI Segmentation

Runs segmentation on new MRI data using a trained model.

Usage:
    # Run inference on test data
    python service/inference.py --checkpoint checkpoints/best_model.pth \
                                --manifest data/processed/class3/manifest.csv \
                                --output predictions/
    
    # Run inference on specific cases
    python service/inference.py --checkpoint checkpoints/best_model.pth \
                                --manifest data/processed/class2/manifest.csv \
                                --case-ids 1 13 22 \
                                --output predictions/
    
    # Run with visualization
    python service/inference.py --checkpoint checkpoints/best_model.pth \
                                --manifest data/processed/class2/manifest.csv \
                                --output predictions/ \
                                --visualize
    
    # Batch inference
    python service/inference.py --checkpoint checkpoints/best_model.pth \
                                --input-dir data/processed/ \
                                --output predictions/batch/
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dataset.dataset_2d5 import MRI25DDataset, collate_fn_with_none


def _is_mri_config(config_path: str) -> bool:
    try:
        import yaml

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return isinstance(cfg.get("task"), dict) and isinstance(cfg.get("data"), dict)
    except Exception:
        return False


class SegmentationInference:
    """Inference engine for segmentation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        
        self.model.eval()
    
    def predict_batch(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """Predict on a batch of images."""
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            predictions = torch.sigmoid(outputs)
        
        return predictions
    
    def predict_dataset(
        self,
        dataloader: DataLoader,
        return_metadata: bool = True
    ) -> Dict:
        """Run inference on entire dataset."""
        all_predictions = []
        all_images = []
        all_metadata = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Running inference")
            for batch_idx, (images, _) in enumerate(pbar):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_images.append(images.cpu())
                
                if return_metadata:
                    for i in range(images.shape[0]):
                        all_metadata.append({
                            'batch_idx': batch_idx,
                            'sample_idx': batch_idx * dataloader.batch_size + i,
                        })
        
        # Concatenate
        all_predictions = torch.cat(all_predictions, dim=0)
        all_images = torch.cat(all_images, dim=0)
        
        return {
            'predictions': all_predictions,
            'images': all_images,
            'metadata': all_metadata if return_metadata else None,
        }
    
    def save_predictions(
        self,
        predictions: torch.Tensor,
        output_dir: Path,
        dataset: Optional[MRI25DDataset] = None,
        prefix: str = "pred",
    ):
        """Save predictions as PNG images."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving predictions to {output_dir}")
        
        for i in tqdm(range(predictions.shape[0]), desc="Saving"):
            pred = predictions[i, 0].numpy()
            
            # Save probability map
            pred_prob = (pred * 255).astype(np.uint8)
            Image.fromarray(pred_prob).save(
                output_dir / f"{prefix}_{i:04d}_prob.png"
            )
            
            # Save binary mask
            pred_binary = (pred > self.threshold).astype(np.uint8) * 255
            Image.fromarray(pred_binary).save(
                output_dir / f"{prefix}_{i:04d}_mask.png"
            )
            
            # Save metadata
            if dataset is not None:
                info = dataset.get_sample_info(i)
                info_path = output_dir / f"{prefix}_{i:04d}_info.json"
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2)
        
        print(f"✓ Saved {predictions.shape[0]} predictions")
    
    def create_visualization(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        output_dir: Path,
        num_samples: int = 20,
    ):
        """Create visualization overlays."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating visualizations...")
        
        # Select samples
        indices = np.linspace(
            0,
            min(predictions.shape[0], images.shape[0]) - 1,
            min(num_samples, predictions.shape[0]),
            dtype=int
        )
        
        for idx in tqdm(indices, desc="Creating visualizations"):
            # Get central slice from stack
            image = images[idx]
            central_slice = image[image.shape[0] // 2].numpy()
            
            # Get prediction
            pred = predictions[idx, 0].numpy()
            pred_binary = (pred > self.threshold)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Original
            axes[0].imshow(central_slice, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Prediction probability
            axes[1].imshow(central_slice, cmap='gray')
            axes[1].imshow(pred, cmap='hot', alpha=0.5, vmin=0, vmax=1)
            axes[1].set_title('Prediction Probability')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(central_slice, cmap='gray')
            axes[2].contour(pred_binary, colors='red', linewidths=2, levels=[0.5])
            axes[2].set_title(f'Segmentation (threshold={self.threshold})')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(
                output_dir / f"vis_{idx:04d}.png",
                dpi=150,
                bbox_inches='tight'
            )
            plt.close()
        
        print(f"✓ Created {len(indices)} visualizations")
    
    def save_summary_report(
        self,
        predictions: torch.Tensor,
        output_path: Path,
        dataset: Optional[MRI25DDataset] = None,
    ):
        """Save summary report of predictions."""
        print("\nGenerating summary report...")
        
        results = []
        
        for i in range(predictions.shape[0]):
            pred = predictions[i, 0].numpy()
            pred_binary = (pred > self.threshold)
            
            # Calculate statistics
            stats = {
                'sample_idx': i,
                'mean_probability': float(pred.mean()),
                'max_probability': float(pred.max()),
                'min_probability': float(pred.min()),
                'positive_pixels': int(pred_binary.sum()),
                'positive_ratio': float(pred_binary.mean()),
            }
            
            # Add metadata if available
            if dataset is not None:
                info = dataset.get_sample_info(i)
                stats['case_id'] = info['case_id']
                stats['series_uid'] = info['series_uid']
                stats['central_slice_idx'] = info['central_slice_idx']
            
            results.append(stats)
        
        # Save as CSV
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Saved summary report: {output_path}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Total samples: {len(results)}")
        print(f"  Samples with positive predictions: {(df['positive_pixels'] > 0).sum()}")
        print(f"  Mean positive ratio: {df['positive_ratio'].mean():.4f}")
        print(f"  Mean probability: {df['mean_probability'].mean():.4f}")


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> tuple:
    """Load model and config from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    config_path = Path(checkpoint_path).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        print("Warning: config.json not found, using defaults")
        config = {
            'model': 'smp',
            'encoder': 'resnet34',
            'stack_depth': 5,
        }
    
    # Create model
    model_type = config.get('model', 'smp')
    encoder = config.get('encoder', 'resnet34')
    stack_depth = config.get('stack_depth', 5)
    
    if model_type == 'smp':
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=stack_depth,
            classes=1,
            activation=None,
        )
    elif model_type == 'monai':
        from monai.networks.nets import SegResNet
        model = SegResNet(
            spatial_dims=2,
            in_channels=stack_depth,
            out_channels=1,
            init_filters=32,
            dropout_prob=0.2,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Model: {model_type}")
    print(f"  Epoch: {checkpoint['epoch']}")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Run Inference with 2.5D Segmentation Model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to nested MRI YAML config (dispatches to mri/cli/infer.py)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split key to use with nested MRI config (e.g., train/val/test)",
    )
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    
    # Input
    parser.add_argument("--manifest", type=str, help="Path to manifest CSV")
    parser.add_argument("--input-dir", type=str, help="Input directory with manifests")
    parser.add_argument("--case-ids", type=str, nargs="+", help="Specific case IDs to process")
    parser.add_argument("--classes", type=int, nargs="+", help="Filter by PIRADS classes")
    
    # Output
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--num-vis", type=int, default=20, help="Number of visualizations")
    parser.add_argument("--save-summary", action="store_true", default=True, help="Save summary report")
    
    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    
    args = parser.parse_args()

    if args.config and _is_mri_config(args.config):
        from mri.cli.infer import main as mri_infer_main

        print("Detected nested MRI config. Dispatching to mri/cli/infer.py ...")
        cli_args = ["--config", args.config]
        if args.split:
            cli_args += ["--split", args.split]
        mri_infer_main(cli_args)
        return
    
    print("="*80)
    print("2.5D MRI Segmentation Inference")
    print("="*80)
    
    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Determine input
    if args.manifest:
        manifests = [Path(args.manifest)]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        manifests = list(input_dir.glob("class*/manifest.csv"))
        if not manifests:
            print(f"Error: No manifests found in {input_dir}")
            return 1
    else:
        print("Error: Must provide --manifest or --input-dir")
        return 1
    
    print(f"\nFound {len(manifests)} manifest(s) to process")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each manifest
    for manifest_path in manifests:
        print(f"\n{'='*80}")
        print(f"Processing: {manifest_path}")
        print(f"{'='*80}")
        
        # Create dataset
        dataset = MRI25DDataset(
            manifest_csv=str(manifest_path),
            stack_depth=config['stack_depth'],
            image_size=(256, 256),
            normalize_method="scale",
            has_masks=False,  # Don't require masks for inference
            filter_by_class=args.classes,
        )
        
        # Filter by case IDs if provided
        if args.case_ids:
            # This would require filtering the dataset
            # For now, just print a warning
            print(f"Warning: Case ID filtering not yet implemented")
        
        print(f"Dataset size: {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("Skipping empty dataset")
            continue
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn_with_none,
        )
        
        # Create inference engine
        inference = SegmentationInference(
            model=model,
            device=args.device,
            threshold=args.threshold,
        )
        
        # Run inference
        print("\nRunning inference...")
        results = inference.predict_dataset(dataloader)
        
        # Determine output subdirectory
        manifest_name = manifest_path.parent.name
        output_subdir = output_dir / manifest_name
        
        # Save predictions
        inference.save_predictions(
            results['predictions'],
            output_subdir / "masks",
            dataset=dataset,
        )
        
        # Create visualizations
        if args.visualize:
            inference.create_visualization(
                results['images'],
                results['predictions'],
                output_subdir / "visualizations",
                num_samples=args.num_vis,
            )
        
        # Save summary report
        if args.save_summary:
            inference.save_summary_report(
                results['predictions'],
                output_subdir / "summary.csv",
                dataset=dataset,
            )
    
    print("\n" + "="*80)
    print("✓ Inference complete!")
    print(f"✓ Results saved to: {output_dir}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
