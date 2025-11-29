#!/usr/bin/env python3
"""
Testing and Evaluation Script for 2.5D MRI Segmentation

Evaluates trained models on test data with comprehensive metrics.

Usage:
    # Evaluate on test set
    python service/test.py --checkpoint checkpoints/best_model.pth
    
    # Evaluate on specific manifest
    python service/test.py --checkpoint checkpoints/best_model.pth \
                          --manifest data/processed/class3/manifest.csv
    
    # Save predictions
    python service/test.py --checkpoint checkpoints/best_model.pth \
                          --save-predictions results/predictions
    
    # Visualize results
    python service/test.py --checkpoint checkpoints/best_model.pth \
                          --visualize --num-vis 10
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dataset.dataset_2d5 import MRI25DDataset, collate_fn_with_none


class Evaluator:
    """Evaluator for segmentation models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.test_loader = test_loader
        
        self.model.eval()
    
    def calculate_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate segmentation metrics."""
        pred_binary = (pred > threshold).float()
        
        # True Positive, False Positive, False Negative, True Negative
        tp = (pred_binary * target).sum().item()
        fp = (pred_binary * (1 - target)).sum().item()
        fn = ((1 - pred_binary) * target).sum().item()
        tn = ((1 - pred_binary) * (1 - target)).sum().item()
        
        # Dice Score
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        
        # IoU (Jaccard)
        iou = tp / (tp + fp + fn + 1e-8)
        
        # Precision and Recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # F1 Score (same as Dice for binary)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        all_metrics = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")
            for images, masks in pbar:
                images = images.to(self.device)
                
                # Skip if no masks
                if masks is None:
                    continue
                
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                pred = torch.sigmoid(outputs)
                
                # Calculate metrics for each sample
                for i in range(pred.shape[0]):
                    metrics = self.calculate_metrics(pred[i], masks[i])
                    all_metrics.append(metrics)
                    
                    # Store for later analysis
                    all_predictions.append(pred[i].cpu())
                    all_targets.append(masks[i].cpu())
                
                # Update progress
                if all_metrics:
                    avg_dice = np.mean([m['dice'] for m in all_metrics])
                    pbar.set_postfix({"dice": f"{avg_dice:.4f}"})
        
        # Aggregate metrics
        if not all_metrics:
            print("Warning: No samples with masks found!")
            return {}
        
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_median'] = np.median(values)
        
        aggregated['num_samples'] = len(all_metrics)
        
        return aggregated, all_predictions, all_targets
    
    def save_predictions(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        output_dir: Path,
    ):
        """Save predictions as images."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving predictions to {output_dir}")
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            # Convert to numpy
            pred_np = (pred[0].numpy() * 255).astype(np.uint8)
            target_np = (target[0].numpy() * 255).astype(np.uint8)
            
            # Save
            Image.fromarray(pred_np).save(output_dir / f"pred_{i:04d}.png")
            Image.fromarray(target_np).save(output_dir / f"target_{i:04d}.png")
        
        print(f"✓ Saved {len(predictions)} predictions")
    
    def visualize_results(
        self,
        dataset: MRI25DDataset,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        output_dir: Path,
        num_samples: int = 10,
    ):
        """Create visualization of results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating visualizations...")
        
        # Select samples to visualize
        indices = np.linspace(0, len(predictions)-1, num_samples, dtype=int)
        
        for idx in indices:
            # Get original image (central slice)
            sample_info = dataset.get_sample_info(idx)
            central_slice_idx = sample_info['central_slice_idx']
            
            # Load central slice from dataset
            image_tensor, _ = dataset[idx]
            central_image = image_tensor[image_tensor.shape[0]//2]  # Middle slice
            
            # Get prediction and target
            pred = predictions[idx][0].numpy()
            target = targets[idx][0].numpy()
            
            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Original image
            axes[0].imshow(central_image.numpy(), cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(central_image.numpy(), cmap='gray')
            axes[1].imshow(target, cmap='Reds', alpha=0.5 * (target > 0))
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(central_image.numpy(), cmap='gray')
            axes[2].imshow(pred, cmap='Greens', alpha=0.5 * (pred > 0.5))
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            # Overlay
            axes[3].imshow(central_image.numpy(), cmap='gray')
            axes[3].imshow(target, cmap='Reds', alpha=0.3 * (target > 0))
            axes[3].imshow(pred, cmap='Greens', alpha=0.3 * (pred > 0.5))
            axes[3].set_title('Overlay (GT=Red, Pred=Green)')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"vis_{idx:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Saved {len(indices)} visualizations")


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load config
    config_path = Path(checkpoint_path).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        model_type = config.get('model', 'smp')
        encoder = config.get('encoder', 'resnet34')
        stack_depth = config.get('stack_depth', 5)
    else:
        print("Warning: config.json not found, using defaults")
        model_type = 'smp'
        encoder = 'resnet34'
        stack_depth = 5
    
    # Create model
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
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Test 2.5D Segmentation Model")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    
    # Data
    parser.add_argument("--manifest", type=str, default="data/processed/class2/manifest.csv")
    parser.add_argument("--classes", type=int, nargs="+", help="Filter by PIRADS classes")
    parser.add_argument("--stack-depth", type=int, default=5)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256])
    
    # Evaluation
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num-vis", type=int, default=10, help="Number of visualizations")
    
    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    
    args = parser.parse_args()
    
    print("="*80)
    print("2.5D MRI Segmentation Testing")
    print("="*80)
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Create dataset
    test_dataset = MRI25DDataset(
        manifest_csv=args.manifest,
        stack_depth=args.stack_depth,
        image_size=tuple(args.image_size),
        normalize_method="scale",
        has_masks=True,
        filter_by_class=args.classes,
    )
    
    print(f"\nTest dataset size: {len(test_dataset)} samples")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_with_none,
    )
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=args.device,
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics, predictions, targets = evaluator.evaluate()
    
    if not metrics:
        print("Error: No samples with masks to evaluate!")
        return 1
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nNumber of samples: {metrics['num_samples']}")
    print("\nMetrics:")
    for key in ['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy']:
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        if mean_key in metrics:
            print(f"  {key.upper():12s}: {metrics[mean_key]:.4f} ± {metrics[std_key]:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved results: {results_path}")
    
    # Save predictions
    if args.save_predictions:
        pred_dir = output_dir / "predictions"
        evaluator.save_predictions(predictions, targets, pred_dir)
    
    # Create visualizations
    if args.visualize:
        vis_dir = output_dir / "visualizations"
        evaluator.visualize_results(
            test_dataset,
            predictions,
            targets,
            vis_dir,
            num_samples=min(args.num_vis, len(predictions))
        )
    
    print("\n" + "="*80)
    print("✓ Evaluation complete!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

