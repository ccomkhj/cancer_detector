"""
2.5D MRI Dataset with Multi-Class Segmentation (Prostate + Target1 + Target2)

This dataset loads ALL masks together:
- Images from data/processed/ (as in manifest.csv)
- All masks from data/processed_seg/ (prostate, target1, target2)

Output format:
- Image: (stack_depth, H, W)
- Mask: (num_classes, H, W) where:
  - Channel 0: Prostate
  - Channel 1: Target1
  - Channel 2: Target2

Usage:
    from dataset_2d5_multiclass import MRI25DMultiClassDataset, create_multiclass_dataloader
    
    dataset = MRI25DMultiClassDataset(
        manifest_csv="data/processed/class2/manifest.csv",
        stack_depth=5
    )
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class MRI25DMultiClassDataset(Dataset):
    """
    2.5D MRI Dataset with multi-class segmentation masks
    
    Loads ALL available masks (prostate, target1, target2) simultaneously
    """
    
    def __init__(
        self,
        manifest_csv: Union[str, Path],
        stack_depth: int = 5,
        transform=None,
        normalize: bool = True,
        skip_no_masks: bool = True,
        target_size: Tuple[int, int] = (256, 256),
    ):
        """
        Args:
            manifest_csv: Path to manifest CSV file
            stack_depth: Number of slices to stack (must be odd)
            transform: Optional transform to apply to (image, mask) pair
            normalize: Whether to normalize images to [0, 1]
            skip_no_masks: If True, skip samples with no masks at all
            target_size: Resize all images and masks to this size (height, width)
        """
        self.manifest_csv = Path(manifest_csv)
        self.stack_depth = stack_depth
        self.transform = transform
        self.normalize = normalize
        self.skip_no_masks = skip_no_masks
        self.target_size = target_size
        
        # Mask types to load
        self.mask_types = ['prostate', 'target1', 'target2']
        
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
        
        # Build valid indices (slices that can form complete stacks)
        self.valid_indices = self._build_valid_indices()
        
        print(f"Loaded {len(self.valid_indices)} valid samples from {len(self.df)} total slices")
        print(f"  Class: {self.class_name}")
        print(f"  Stack depth: {self.stack_depth}")
        print(f"  Multi-class: {', '.join(self.mask_types)}")
    
    def _build_valid_indices(self) -> List[Tuple[int, int, str]]:
        """Build list of valid (case_id, series_uid, slice_idx) tuples."""
        valid = []
        half_depth = self.stack_depth // 2
        
        for (case_id, series_uid), group in self.series_groups:
            sorted_group = group.sort_values('slice_idx')
            slice_indices = sorted_group['slice_idx'].values
            
            # Find mask directories
            case_dir = self.processed_seg_base / f"case_{int(case_id):04d}"
            series_dir = case_dir / series_uid
            
            # Check if any masks exist for this series
            has_any_masks = False
            for mask_type in self.mask_types:
                mask_dir = series_dir / mask_type
                if mask_dir.exists() and list(mask_dir.glob("*.png")):
                    has_any_masks = True
                    break
            
            if self.skip_no_masks and not has_any_masks:
                continue
            
            # Get available mask slice numbers across all types
            all_mask_slices = set()
            for mask_type in self.mask_types:
                mask_dir = series_dir / mask_type
                if mask_dir.exists():
                    mask_files = list(mask_dir.glob("*.png"))
                    all_mask_slices.update(int(f.stem) for f in mask_files)
            
            # Only include slices that can form complete stacks
            for slice_idx in slice_indices:
                # Check if we can form a complete stack
                if slice_idx < half_depth or slice_idx > slice_indices[-1] - half_depth:
                    continue
                
                # If skip_no_masks, only include slices with at least one mask
                if self.skip_no_masks and slice_idx not in all_mask_slices:
                    continue
                
                valid.append((case_id, series_uid, slice_idx))
        
        return valid
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load, resize, and normalize a single image."""
        img = Image.open(path).convert('L')
        
        # Resize to target size
        if img.size != (self.target_size[1], self.target_size[0]):  # PIL uses (width, height)
            img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        
        img_array = np.array(img, dtype=np.float32)
        
        if self.normalize:
            img_array = img_array / 255.0
        
        return img_array
    
    def _load_mask(self, path: Path) -> Optional[np.ndarray]:
        """Load, resize, and binarize a mask."""
        if not path.exists():
            return None
        
        mask = Image.open(path).convert('L')
        
        # Resize to target size using nearest neighbor to preserve binary values
        if mask.size != (self.target_size[1], self.target_size[0]):  # PIL uses (width, height)
            mask = mask.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        
        mask_array = np.array(mask, dtype=np.float32)
        
        # Binarize: 0 = background, 1 = foreground
        mask_array = (mask_array > 127).astype(np.float32)
        
        return mask_array
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor of shape (stack_depth, H, W)
            mask: Tensor of shape (num_classes, H, W) where each channel is a binary mask
                  Channel 0: Prostate
                  Channel 1: Target1
                  Channel 2: Target2
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
        
        # Load all masks for center slice
        case_dir = self.processed_seg_base / f"case_{int(case_id):04d}"
        series_dir = case_dir / series_uid
        
        # Create multi-class mask: (num_classes, H, W)
        mask_channels = []
        for mask_type in self.mask_types:
            mask_dir = series_dir / mask_type
            mask_path = mask_dir / f"{center_slice:04d}.png"
            
            mask = self._load_mask(mask_path)
            
            if mask is None:
                # Create empty mask if this type doesn't exist
                if len(mask_channels) > 0:
                    mask = np.zeros_like(mask_channels[0])
                else:
                    # Use image shape
                    mask = np.zeros((image_stack.shape[1], image_stack.shape[2]), dtype=np.float32)
            
            mask_channels.append(mask)
        
        # Stack masks: (num_classes, H, W)
        mask_stack = np.stack(mask_channels, axis=0)
        
        # Apply transforms
        if self.transform:
            image_stack, mask_stack = self.transform(image_stack, mask_stack)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_stack).float()
        mask_tensor = torch.from_numpy(mask_stack).float()
        
        return image_tensor, mask_tensor


def create_multiclass_dataloader(
    manifest_csv: Union[str, Path],
    stack_depth: int = 5,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    normalize: bool = True,
    skip_no_masks: bool = True,
    target_size: Tuple[int, int] = (256, 256),
) -> DataLoader:
    """
    Create a DataLoader for multi-class 2.5D MRI segmentation
    
    Args:
        manifest_csv: Path to manifest CSV
        stack_depth: Number of slices to stack (must be odd)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        transform: Optional transform
        normalize: Normalize images to [0, 1]
        skip_no_masks: Skip samples with no masks
    
    Returns:
        DataLoader instance
    """
    dataset = MRI25DMultiClassDataset(
        manifest_csv=manifest_csv,
        stack_depth=stack_depth,
        transform=transform,
        normalize=normalize,
        skip_no_masks=skip_no_masks,
        target_size=target_size,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
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
    print("Creating multi-class dataset...")
    dataset = MRI25DMultiClassDataset(
        manifest_csv=manifest_csv,
        stack_depth=5,
        skip_no_masks=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        print("\nLoading sample...")
        image, mask = dataset[0]
        
        print(f"  Image shape: {image.shape}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Mask shape: {mask.shape}")
        
        # Check each mask channel
        for i, mask_type in enumerate(['Prostate', 'Target1', 'Target2']):
            mask_ch = mask[i]
            coverage = (mask_ch > 0).float().mean() * 100
            print(f"  {mask_type}: {coverage:.1f}% coverage")
        
        # Visualize
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Show image stack (center slices)
        axes[0, 0].imshow(image[1], cmap='gray')
        axes[0, 0].set_title("Slice -1")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(image[2], cmap='gray')
        axes[0, 1].set_title("Center Slice")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(image[3], cmap='gray')
        axes[0, 2].set_title("Slice +1")
        axes[0, 2].axis('off')
        
        # Show combined overlay
        axes[0, 3].imshow(image[2], cmap='gray')
        # Overlay all masks with different colors
        colors = [(1, 1, 0), (1, 0, 0), (1, 0.5, 0)]  # Yellow, Red, Orange
        for i, color in enumerate(colors):
            if mask[i].max() > 0:
                mask_rgba = np.zeros((*mask[i].shape, 4))
                mask_rgba[..., 0] = color[0]
                mask_rgba[..., 1] = color[1]
                mask_rgba[..., 2] = color[2]
                mask_rgba[..., 3] = 0.5 * mask[i].numpy()
                axes[0, 3].imshow(mask_rgba)
        axes[0, 3].set_title("All Masks")
        axes[0, 3].axis('off')
        
        # Show individual masks
        mask_names = ['Prostate', 'Target1', 'Target2']
        for i in range(3):
            axes[1, i].imshow(mask[i], cmap='gray')
            axes[1, i].set_title(f"{mask_names[i]} Mask")
            axes[1, i].axis('off')
        
        # Show overlay again
        axes[1, 3].imshow(image[2], cmap='gray')
        for i, color in enumerate(colors):
            if mask[i].max() > 0:
                mask_rgba = np.zeros((*mask[i].shape, 4))
                mask_rgba[..., 0] = color[0]
                mask_rgba[..., 1] = color[1]
                mask_rgba[..., 2] = color[2]
                mask_rgba[..., 3] = 0.5 * mask[i].numpy()
                axes[1, 3].imshow(mask_rgba)
        axes[1, 3].set_title("Overlay")
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig("test_multiclass_dataset.png", dpi=150)
        print("\n✓ Saved visualization: test_multiclass_dataset.png")
    
    # Test dataloader
    print("\nCreating dataloader...")
    dataloader = create_multiclass_dataloader(
        manifest_csv=manifest_csv,
        stack_depth=5,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        skip_no_masks=True,
    )
    
    print(f"Dataloader batches: {len(dataloader)}")
    
    if len(dataloader) > 0:
        print("\nLoading batch...")
        images, masks = next(iter(dataloader))
        
        print(f"  Batch images shape: {images.shape}")
        print(f"  Batch masks shape: {masks.shape}")
        
        print("\n✓ Multi-class dataset test complete!")

