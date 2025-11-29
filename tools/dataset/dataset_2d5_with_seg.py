"""
2.5D MRI Dataset with Segmentation Masks from processed_seg/

This dataset loads:
- Images from data/processed/ (as in manifest.csv)
- Masks from data/processed_seg/ (prostate, target1, target2)

Usage:
    from dataset_2d5_with_seg import MRI25DSegDataset, create_seg_dataloader
    
    dataset = MRI25DSegDataset(
        manifest_csv="data/processed/class2/manifest.csv",
        stack_depth=5,
        mask_type="prostate"  # or "target1", "target2"
    )
    
    dataloader = create_seg_dataloader(
        manifest_csv="data/processed/class2/manifest.csv",
        stack_depth=5,
        mask_type="prostate",
        batch_size=8
    )
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class MRI25DSegDataset(Dataset):
    """
    2.5D MRI Dataset with segmentation masks from processed_seg/
    
    Loads:
    - Images from data/processed/{classN}/{case_id}/{series_uid}/images/
    - Masks from data/processed_seg/{classN}/case_{case_id}/{series_uid}/{mask_type}/
    """
    
    def __init__(
        self,
        manifest_csv: Union[str, Path],
        stack_depth: int = 5,
        mask_type: str = "prostate",
        transform=None,
        normalize: bool = True,
        skip_missing_masks: bool = False,
    ):
        """
        Args:
            manifest_csv: Path to manifest CSV file
            stack_depth: Number of slices to stack (must be odd)
            mask_type: Which mask to load ("prostate", "target1", "target2")
            transform: Optional transform to apply to (image, mask) pair
            normalize: Whether to normalize images to [0, 1]
            skip_missing_masks: If True, skip samples without masks. If False, return None for missing masks.
        """
        self.manifest_csv = Path(manifest_csv)
        self.stack_depth = stack_depth
        self.mask_type = mask_type.lower()
        self.transform = transform
        self.normalize = normalize
        self.skip_missing_masks = skip_missing_masks
        
        if stack_depth % 2 == 0:
            raise ValueError(f"stack_depth must be odd, got {stack_depth}")
        
        # Load manifest
        self.df = pd.read_csv(manifest_csv)
        
        # Determine class from manifest path
        self.class_name = None
        for i in range(1, 5):
            if f'/class{i}/' in str(manifest_csv):
                self.class_name = f'class{i}'
                break
        
        if not self.class_name:
            raise ValueError(f"Could not determine class from manifest path: {manifest_csv}")
        
        # Build processed_seg base path
        self.processed_seg_base = Path("data/processed_seg") / self.class_name
        
        # Group by series to ensure proper stacking
        self.series_groups = self.df.groupby(['case_id', 'series_uid'])
        
        # Build valid indices (slices that can form complete stacks and have masks)
        self.valid_indices = self._build_valid_indices()
        
        print(f"Loaded {len(self.valid_indices)} valid samples from {len(self.df)} total slices")
        print(f"  Class: {self.class_name}")
        print(f"  Mask type: {self.mask_type}")
        print(f"  Stack depth: {self.stack_depth}")
    
    def _build_valid_indices(self) -> List[Tuple[int, int, str]]:
        """Build list of valid (case_id, series_uid, slice_idx) tuples."""
        valid = []
        half_depth = self.stack_depth // 2
        
        for (case_id, series_uid), group in self.series_groups:
            sorted_group = group.sort_values('slice_idx')
            slice_indices = sorted_group['slice_idx'].values
            
            # Find mask directory
            case_dir = self.processed_seg_base / f"case_{int(case_id):04d}"
            series_dir = case_dir / series_uid
            mask_dir = series_dir / self.mask_type
            
            if not mask_dir.exists():
                if not self.skip_missing_masks:
                    # Add all stackable slices even without masks
                    for slice_idx in slice_indices:
                        if (slice_idx >= half_depth and 
                            slice_idx <= slice_indices[-1] - half_depth):
                            valid.append((case_id, series_uid, slice_idx))
                continue
            
            # Get available mask slice numbers
            mask_files = list(mask_dir.glob("*.png"))
            available_mask_slices = set(int(f.stem) for f in mask_files)
            
            # Only include slices that can form complete stacks AND have masks
            for slice_idx in slice_indices:
                # Check if we can form a complete stack
                if slice_idx < half_depth or slice_idx > slice_indices[-1] - half_depth:
                    continue
                
                # Check if mask exists (or we don't care)
                if self.skip_missing_masks and slice_idx not in available_mask_slices:
                    continue
                
                valid.append((case_id, series_uid, slice_idx))
        
        return valid
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and normalize a single image."""
        img = Image.open(path).convert('L')
        img_array = np.array(img, dtype=np.float32)
        
        if self.normalize:
            img_array = img_array / 255.0
        
        return img_array
    
    def _load_mask(self, path: Path) -> Optional[np.ndarray]:
        """Load a mask (binary: 0 or 1)."""
        if not path.exists():
            return None
        
        mask = Image.open(path).convert('L')
        mask_array = np.array(mask, dtype=np.float32)
        
        # Binarize: 0 = background, 1 = foreground
        mask_array = (mask_array > 127).astype(np.float32)
        
        return mask_array
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            image: Tensor of shape (stack_depth, H, W)
            mask: Tensor of shape (1, H, W) or None if mask doesn't exist
        """
        case_id, series_uid, center_slice = self.valid_indices[idx]
        
        # Get all slices in this series
        series_df = self.df[
            (self.df['case_id'] == case_id) & 
            (self.df['series_uid'] == series_uid)
        ].sort_values('slice_idx')
        
        # Calculate stack range
        half_depth = self.stack_depth // 2
        stack_slices = range(center_slice - half_depth, center_slice + half_depth + 1)
        
        # Load image stack
        image_stack = []
        for slice_idx in stack_slices:
            slice_row = series_df[series_df['slice_idx'] == slice_idx].iloc[0]
            img_path = Path(slice_row['image_path'])
            img = self._load_image(img_path)
            image_stack.append(img)
        
        # Stack images: (stack_depth, H, W)
        image_stack = np.stack(image_stack, axis=0)
        
        # Load mask for center slice
        case_dir = self.processed_seg_base / f"case_{int(case_id):04d}"
        series_dir = case_dir / series_uid
        mask_dir = series_dir / self.mask_type
        mask_path = mask_dir / f"{center_slice:04d}.png"
        
        mask = self._load_mask(mask_path)
        
        if mask is not None:
            # Add channel dimension: (1, H, W)
            mask = mask[np.newaxis, ...]
        
        # Apply transforms
        if self.transform:
            image_stack, mask = self.transform(image_stack, mask)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_stack).float()
        
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float()
        else:
            mask_tensor = None
        
        return image_tensor, mask_tensor


def collate_fn_with_none(batch):
    """
    Custom collate function that handles None values in masks.
    
    Filters out samples without masks if all masks are None.
    """
    images = []
    masks = []
    
    for img, mask in batch:
        images.append(img)
        masks.append(mask)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Handle masks
    if all(m is None for m in masks):
        masks = None
    else:
        # Keep valid masks, skip None
        valid_samples = [(img, mask) for img, mask in zip(images, masks) if mask is not None]
        
        if not valid_samples:
            masks = None
        else:
            images = torch.stack([img for img, _ in valid_samples], dim=0)
            masks = torch.stack([mask for _, mask in valid_samples], dim=0)
    
    return images, masks


def create_seg_dataloader(
    manifest_csv: Union[str, Path],
    stack_depth: int = 5,
    mask_type: str = "prostate",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    normalize: bool = True,
    skip_missing_masks: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for 2.5D MRI segmentation with masks from processed_seg/
    
    Args:
        manifest_csv: Path to manifest CSV
        stack_depth: Number of slices to stack (must be odd)
        mask_type: Which mask to load ("prostate", "target1", "target2")
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        transform: Optional transform
        normalize: Normalize images to [0, 1]
        skip_missing_masks: Skip samples without masks (recommended for training)
    
    Returns:
        DataLoader instance
    """
    dataset = MRI25DSegDataset(
        manifest_csv=manifest_csv,
        stack_depth=stack_depth,
        mask_type=mask_type,
        transform=transform,
        normalize=normalize,
        skip_missing_masks=skip_missing_masks,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_with_none,
    )
    
    return dataloader


if __name__ == "__main__":
    """Quick test of the dataset"""
    import matplotlib.pyplot as plt
    
    manifest_csv = "data/processed/class2/manifest.csv"
    
    if not Path(manifest_csv).exists():
        print(f"Manifest not found: {manifest_csv}")
        exit(1)
    
    # Test dataset
    print("Creating dataset...")
    dataset = MRI25DSegDataset(
        manifest_csv=manifest_csv,
        stack_depth=5,
        mask_type="prostate",
        skip_missing_masks=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        print("\nLoading sample...")
        image, mask = dataset[0]
        
        print(f"  Image shape: {image.shape}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        
        if mask is not None:
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask unique values: {torch.unique(mask).numpy()}")
            print(f"  Mask coverage: {(mask > 0).float().mean() * 100:.1f}%")
        else:
            print(f"  Mask: None")
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Show image stack
        for i in range(5):
            axes[0, i % 3].imshow(image[i], cmap='gray')
            axes[0, i % 3].set_title(f"Slice {i}")
            axes[0, i % 3].axis('off')
            
            if i >= 3:
                break
        
        # Show center slice with mask overlay
        center_idx = 2
        axes[1, 0].imshow(image[center_idx], cmap='gray')
        axes[1, 0].set_title("Center Slice")
        axes[1, 0].axis('off')
        
        if mask is not None:
            axes[1, 1].imshow(mask[0], cmap='Reds')
            axes[1, 1].set_title("Mask")
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(image[center_idx], cmap='gray')
            axes[1, 2].imshow(mask[0], cmap='Reds', alpha=0.5 * (mask[0] > 0))
            axes[1, 2].set_title("Overlay")
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig("test_seg_dataset.png", dpi=150)
        print("\n✓ Saved visualization: test_seg_dataset.png")
    
    # Test dataloader
    print("\nCreating dataloader...")
    dataloader = create_seg_dataloader(
        manifest_csv=manifest_csv,
        stack_depth=5,
        mask_type="prostate",
        batch_size=4,
        shuffle=False,
        num_workers=0,
        skip_missing_masks=True,
    )
    
    print(f"Dataloader batches: {len(dataloader)}")
    
    if len(dataloader) > 0:
        print("\nLoading batch...")
        images, masks = next(iter(dataloader))
        
        print(f"  Batch images shape: {images.shape}")
        if masks is not None:
            print(f"  Batch masks shape: {masks.shape}")
        else:
            print(f"  Batch masks: None")
        
        print("\n✓ Dataset test complete!")

