#!/usr/bin/env python3
"""
Inference Script for 2.5D MRI Segmentation

Run segmentation on processed MRI manifests using a trained checkpoint.

How to run:
    1) Provide model checkpoint: --checkpoint <path_to_.pt/.pth>
    2) Provide input source:
       - --manifest <path_to_manifest.csv> for one class/folder, or
       - --input-dir <dir_with_class*/manifest.csv> for batch mode
    3) Provide output directory: --output <predictions_dir>

Preprocessing behavior:
    - For multimodal checkpoints (T2 + ADC + Calc), preprocessing settings are
      loaded from checkpoint/config (metadata path, stack depth, filters).
    - You can override metadata path with: --metadata <aligned_v2/metadata.json>

Outputs per manifest:
    - masks/pred_XXXX_prob.png    (probability map)
    - masks/pred_XXXX_mask.png    (thresholded mask)
    - masks/pred_XXXX_info.json   (sample metadata)
    - summary.csv                 (per-sample stats)
    - visualizations/             (only with --visualize)

Examples:
    # Single manifest
    python service/inference.py --checkpoint checkpoints/766564/model_best_766564_196.pt \
        --manifest data/processed/class2/manifest.csv \
        --output predictions/

    # Filter by case IDs
    python service/inference.py --checkpoint checkpoints/766564/model_best_766564_196.pt \
        --manifest data/processed/class2/manifest.csv \
        --case-ids 1 13 22 \
        --output predictions/

    # Filter by class label(s)
    python service/inference.py --checkpoint checkpoints/766564/model_best_766564_196.pt \
        --manifest data/processed/class4/manifest.csv \
        --classes 4 \
        --output predictions/

    # Batch mode (all class*/manifest.csv)
    python service/inference.py --checkpoint checkpoints/766564/model_best_766564_196.pt \
        --input-dir data/processed/ \
        --output predictions/batch/

    # Save overlays
    python service/inference.py --checkpoint checkpoints/766564/model_best_766564_196.pt \
        --manifest data/processed/class2/manifest.csv \
        --output predictions/ \
        --visualize --num-vis 20
"""

import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dataset.dataset_2d5 import MRI25DDataset, collate_fn_with_none
from tools.dataset.dataset_multimodal import MultiModalDataset
from service.models import SimpleUNet
from service.preprocessing import (
    DEFAULT_MULTIMODAL_METADATA,
    MultiModalPreprocessingConfig,
    build_multimodal_preprocessing_config,
    create_multimodal_dataset,
    infer_class_from_manifest_path,
    select_multimodal_sample_indices,
)


def _is_mri_config(config_path: str) -> bool:
    try:
        import yaml

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return isinstance(cfg.get("task"), dict) and isinstance(cfg.get("data"), dict)
    except Exception:
        return False


def _load_checkpoint_compat(checkpoint_path: str, device: str) -> Dict:
    """Load checkpoint across PyTorch versions (incl. 2.6 weights_only change)."""
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions without weights_only argument
        return torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError as e:
        if "Weights only load failed" not in str(e):
            raise
        print(
            "Warning: weights-only checkpoint loading failed. "
            "Retrying with full pickle deserialization."
        )
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _infer_io_channels_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> tuple[int, int]:
    """Infer model input/output channels from common checkpoint keys."""
    in_channels = None
    out_channels = None

    if "enc1.0.weight" in state_dict:
        weight = state_dict["enc1.0.weight"]
        if isinstance(weight, torch.Tensor) and weight.ndim == 4:
            in_channels = int(weight.shape[1])
    if "out.weight" in state_dict:
        weight = state_dict["out.weight"]
        if isinstance(weight, torch.Tensor) and weight.ndim == 4:
            out_channels = int(weight.shape[0])

    if in_channels is None:
        for tensor in state_dict.values():
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                in_channels = int(tensor.shape[1])
                break

    if out_channels is None:
        for tensor in reversed(list(state_dict.values())):
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                out_channels = int(tensor.shape[0])
                break

    return in_channels or 5, out_channels or 1


def _load_inference_config(checkpoint_path: str, checkpoint: Dict) -> Dict:
    """Load config from nearby files and checkpoint args metadata."""
    base_dir = Path(checkpoint_path).parent
    config: Dict = {}

    config_json = base_dir / "config.json"
    if config_json.exists():
        with open(config_json) as f:
            config.update(json.load(f) or {})
    else:
        for name in ("config_resolved.yaml", "config.yaml"):
            cfg_path = base_dir / name
            if not cfg_path.exists():
                continue
            try:
                import yaml

                with open(cfg_path, "r") as f:
                    yaml_cfg = yaml.safe_load(f) or {}
                if isinstance(yaml_cfg, dict):
                    config.update(yaml_cfg)
                    break
            except Exception as e:
                print(f"Warning: failed to parse {cfg_path.name}: {e}")

    ckpt_args = checkpoint.get("args", {})
    if isinstance(ckpt_args, dict):
        for key in (
            "model",
            "encoder",
            "stack_depth",
            "metadata",
            "require_complete",
            "require_positive",
            "normalize",
        ):
            if key in ckpt_args and key not in config:
                config[key] = ckpt_args[key]

    config.setdefault("model", "smp")
    config.setdefault("encoder", "resnet34")
    config.setdefault("stack_depth", 5)
    config.setdefault("metadata", DEFAULT_MULTIMODAL_METADATA)
    config.setdefault("normalize", True)
    config.setdefault("require_complete", False)
    config.setdefault("require_positive", False)
    return config


def _filter_dataset_by_case_ids(dataset: MRI25DDataset, case_ids: List[str]) -> int:
    """Filter dataset samples to only include requested case IDs."""
    case_ids_set = {str(case_id) for case_id in case_ids}
    before = len(dataset.valid_samples)
    dataset.valid_samples = [
        sample
        for sample in dataset.valid_samples
        if str(sample.get("case_id")) in case_ids_set
    ]
    return before - len(dataset.valid_samples)


def _to_json_serializable(value):
    """Recursively convert common non-JSON-native objects to Python primitives."""
    if isinstance(value, dict):
        return {str(k): _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


class MultiModalInferenceSubset(Dataset):
    """Inference-only view over MultiModalDataset, keeping sample metadata."""

    def __init__(self, base_dataset: MultiModalDataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, _mask = self.base_dataset[self.indices[idx]]
        return image, None

    def get_sample_info(self, idx: int) -> Dict:
        sample = self.base_dataset.samples[self.indices[idx]]
        return {
            "sample_id": sample.get("sample_id"),
            "case_id": sample.get("case_id"),
            "class": sample.get("class"),
            "slice_idx": sample.get("slice_idx"),
        }


def _build_multimodal_inference_dataset(
    manifest_path: Path,
    preprocessing: MultiModalPreprocessingConfig,
    case_ids: Optional[List[str]],
    classes: Optional[List[int]],
) -> MultiModalInferenceSubset:
    """Build inference dataset aligned with training preprocessing."""
    base_dataset = create_multimodal_dataset(preprocessing)

    selected_indices = select_multimodal_sample_indices(
        base_dataset.samples,
        manifest_path=manifest_path,
        case_ids=case_ids,
        classes=classes,
    )

    return MultiModalInferenceSubset(base_dataset=base_dataset, indices=selected_indices)


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
        dataset: Optional[Dataset] = None,
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
                    json.dump(_to_json_serializable(info), f, indent=2)
        
        print(f"✓ Saved {predictions.shape[0]} predictions")
    
    def create_visualization(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        output_dir: Path,
        num_samples: int = 20,
    ):
        """Create visualization overlays.

        For 7-channel inputs (5x T2 + ADC + CALC), renders:
        - Row 1: T2 channels (5 columns)
        - Row 2: ADC at column 3
        - Row 3: CALC at column 3
        With mask overlays on the third-column panels of all rows.
        """
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
            image = images[idx].numpy()

            # Use first output channel for consistency with mask/prob export.
            pred = predictions[idx, 0].numpy()
            pred_binary = (pred > self.threshold)

            def _draw_overlay(ax, base_img: np.ndarray, title: str, with_overlay: bool = False):
                ax.imshow(base_img, cmap="gray")
                if with_overlay:
                    mask = np.ma.masked_where(~pred_binary, pred_binary.astype(np.float32))
                    ax.imshow(mask, cmap="autumn", alpha=0.35, vmin=0, vmax=1)
                    ax.contour(pred_binary, colors="lime", linewidths=1.0, levels=[0.5])
                ax.set_title(title)
                ax.axis("off")

            if image.shape[0] >= 7:
                # 3x5 layout for multi-modal inputs: 5xT2 + ADC + CALC.
                fig, axes = plt.subplots(3, 5, figsize=(18, 10))

                # First row: T2 stack (5 channels). Overlay on third column (center T2).
                for col in range(5):
                    _draw_overlay(
                        axes[0, col],
                        image[col],
                        f"T2-{col + 1}" + (" (+Mask)" if col == 2 else ""),
                        with_overlay=(col == 2),
                    )

                # Hide all slots in row 2 and row 3 first.
                for row in [1, 2]:
                    for col in range(5):
                        axes[row, col].axis("off")

                # Second row, third column: ADC with overlay.
                _draw_overlay(axes[1, 2], image[5], "ADC (+Mask)", with_overlay=True)

                # Third row, third column: CALC with overlay.
                _draw_overlay(axes[2, 2], image[6], "CALC (+Mask)", with_overlay=True)

                fig.suptitle(f"Sample {idx} (threshold={self.threshold})", fontsize=12)
            else:
                # Fallback visualization for non 7-channel inputs.
                central_slice = image[image.shape[0] // 2]
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                _draw_overlay(axes[0], central_slice, "Original Image")

                axes[1].imshow(central_slice, cmap="gray")
                axes[1].imshow(pred, cmap="hot", alpha=0.5, vmin=0, vmax=1)
                axes[1].set_title("Prediction Probability")
                axes[1].axis("off")

                _draw_overlay(
                    axes[2],
                    central_slice,
                    f"Segmentation (threshold={self.threshold})",
                    with_overlay=True,
                )
            
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
        dataset: Optional[Dataset] = None,
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
                for key in (
                    "case_id",
                    "series_uid",
                    "central_slice_idx",
                    "slice_idx",
                    "class",
                    "sample_id",
                ):
                    if key in info:
                        stats[key] = info[key]
            
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
    checkpoint = _load_checkpoint_compat(checkpoint_path, device)
    config = _load_inference_config(checkpoint_path, checkpoint)

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain 'model_state_dict' or 'model'")

    inferred_in_channels, inferred_out_channels = _infer_io_channels_from_state_dict(
        state_dict
    )

    # Create model
    model_type = str(config.get("model", "smp")).lower()
    encoder = config.get("encoder", "resnet34")
    configured_stack_depth = int(config.get("stack_depth", 5))

    if model_type == "simple_unet":
        model = SimpleUNet(
            in_channels=inferred_in_channels,
            out_channels=inferred_out_channels,
        )
    elif model_type == 'smp':
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=inferred_in_channels,
            classes=inferred_out_channels,
            activation=None,
        )
    elif model_type == 'monai':
        from monai.networks.nets import SegResNet
        model = SegResNet(
            spatial_dims=2,
            in_channels=inferred_in_channels,
            out_channels=inferred_out_channels,
            init_filters=32,
            dropout_prob=0.2,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(state_dict)

    # Derive how inference inputs should be built.
    expects_multimodal = inferred_in_channels == configured_stack_depth + 2
    t2_stack_depth = configured_stack_depth
    if not expects_multimodal and configured_stack_depth != inferred_in_channels:
        print(
            f"Warning: config stack_depth={configured_stack_depth}, but checkpoint expects "
            f"in_channels={inferred_in_channels}. Using stack_depth={inferred_in_channels} for inference."
        )
        t2_stack_depth = inferred_in_channels
    if expects_multimodal:
        print(
            "Detected multi-modal checkpoint input shape "
            f"(in_channels={inferred_in_channels} = stack_depth({configured_stack_depth}) + 2). "
            "Inference will use MultiModalDataset preprocessing."
        )

    config["t2_stack_depth"] = t2_stack_depth
    config["model_in_channels"] = inferred_in_channels
    config["model_out_channels"] = inferred_out_channels
    config["expects_multimodal"] = expects_multimodal
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Model: {model_type}")
    print(f"  In/Out channels: {inferred_in_channels}/{inferred_out_channels}")
    print(f"  T2 stack depth: {config['t2_stack_depth']}")
    if "epoch" in checkpoint:
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
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help=(
            "Path to aligned_v2 metadata.json for multimodal inference. "
            "If omitted, uses checkpoint training config metadata (or default)."
        ),
    )
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

    preprocessing_config = None
    if config.get("expects_multimodal", False):
        preprocessing_source = dict(config)
        preprocessing_source["stack_depth"] = int(config["t2_stack_depth"])
        preprocessing_config = build_multimodal_preprocessing_config(
            preprocessing_source,
            metadata_override=args.metadata,
        )

        metadata_path = Path(preprocessing_config.metadata_path)
        if not metadata_path.exists():
            print(
                f"Error: expected multimodal metadata not found: {metadata_path}. "
                "Provide --metadata with aligned_v2 metadata.json."
            )
            return 1

        print("Using multimodal preprocessing:")
        print(f"  metadata: {metadata_path}")
        print(f"  stack_depth: {preprocessing_config.stack_depth}")
        print(f"  normalize: {preprocessing_config.normalize}")
        print(f"  require_complete: {preprocessing_config.require_complete}")
        print(f"  require_positive: {preprocessing_config.require_positive}")
    
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
        
        if config.get("expects_multimodal", False):
            if preprocessing_config is None:
                raise RuntimeError("Multimodal preprocessing config was not initialized")
            dataset = _build_multimodal_inference_dataset(
                manifest_path=manifest_path,
                preprocessing=preprocessing_config,
                case_ids=args.case_ids,
                classes=args.classes,
            )
            if args.case_ids:
                print(f"Applied case ID filter: {args.case_ids}")
            if args.classes:
                print(f"Applied class filter: {args.classes}")
            manifest_class = infer_class_from_manifest_path(manifest_path)
            if manifest_class is not None:
                print(f"Applied manifest class filter: class{manifest_class}")
        else:
            # Legacy manifest-only dataset path (single-modality stacks)
            dataset = MRI25DDataset(
                manifest_csv=str(manifest_path),
                stack_depth=int(config["t2_stack_depth"]),
                image_size=(256, 256),
                normalize_method="scale",
                has_masks=False,  # Don't require masks for inference
                filter_by_class=args.classes,
            )

            # Filter by case IDs if provided
            if args.case_ids:
                removed = _filter_dataset_by_case_ids(dataset, args.case_ids)
                print(
                    f"Applied case ID filter: {args.case_ids} "
                    f"(removed {removed} samples)"
                )
        
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
