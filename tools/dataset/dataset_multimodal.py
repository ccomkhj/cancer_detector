"""
Multi-Modal 2.5D MRI Dataset (T2 + ADC + Calc)

This dataset implementation follows the structure defined in data/aligned_v2/train.md.
It loads 7 channels of input and 2 channels of output (Prostate, Target).

Input Channels:
  0-4: T2 context slices (depth 5)
  5:   ADC slice (center)
  6:   Calc slice (center)

Output Channels:
  0: Prostate
  1: Target (Union of Target1 and Target2)

Usage:
    dataset = MultiModalDataset(
        metadata_path="data/aligned_v2/metadata.json",
        stack_depth=5
    )
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional, List, Dict, Tuple, Union

class MultiModalDataset(Dataset):
    """
    PyTorch Dataset for 2.5D multi-modal prostate segmentation.
    
    Args:
        metadata_path: Path to metadata.json
        stack_depth: Number of T2 context slices (default: 5). 
                     Note: The metadata.json might have fixed indices for 5 slices.
                     If stack_depth is different, we might need to adjust logic or warn.
                     For now, we assume stack_depth=5 fits the metadata.
        transform: Optional augmentation transform
        require_complete: If True, only include samples with ADC+Calc (has_adc=True, has_calc=True)
        require_positive: If True, only include samples with prostate mask (has_prostate=True)
        normalize: If True, apply normalization using global stats from metadata
    """
    
    def __init__(
        self,
        metadata_path: Union[str, Path],
        stack_depth: int = 5,
        transform=None,
        require_complete: bool = False,
        require_positive: bool = False,
        normalize: bool = True,
    ):
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(self.metadata_path) as f:
            self.metadata = json.load(f)
        
        self.base_dir = self.metadata_path.parent
        self.transform = transform
        self.normalize_enabled = normalize
        self.global_stats = self.metadata.get("global_stats", {})
        self.stack_depth = stack_depth
        
        # Validate stack_depth against metadata if possible
        # metadata["config"]["t2_context_window"] usually stores this
        meta_stack_depth = self.metadata.get("config", {}).get("t2_context_window", 5)
        if stack_depth != meta_stack_depth:
            print(f"Warning: Requested stack_depth={stack_depth} but metadata was generated with {meta_stack_depth}. "
                  f"This might lead to runtime errors if indices are missing.")

        # Filter samples based on requirements
        self.samples = []
        for sample in self.metadata["samples"]:
            if require_complete and not (sample.get("has_adc", False) and sample.get("has_calc", False)):
                continue
            if require_positive and not sample.get("has_prostate", False):
                continue
            self.samples.append(sample)
            
        print(f"Initialized MultiModalDataset from {metadata_path}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Stack depth: {stack_depth}")
        print(f"  Require complete (ADC+Calc): {require_complete}")
        print(f"  Require positive (Prostate): {require_positive}")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, path: Path) -> np.ndarray:
        if not path.exists():
            # Return zeros if file missing (should be handled by has_adc/has_calc check usually)
            return np.zeros((256, 256), dtype=np.float32)
        
        try:
            img = Image.open(path).convert('L') # Ensure grayscale
            # Resize if necessary (though metadata implies 256x256)
            if img.size != (256, 256):
                img = img.resize((256, 256), Image.BILINEAR)
            return np.array(img, dtype=np.float32)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return np.zeros((256, 256), dtype=np.float32)

    def _load_mask(self, path: Path) -> np.ndarray:
        if not path.exists():
            return np.zeros((256, 256), dtype=np.float32)
            
        try:
            img = Image.open(path).convert('L')
            if img.size != (256, 256):
                img = img.resize((256, 256), Image.NEAREST)
            return (np.array(img, dtype=np.float32) > 127).astype(np.float32)
        except Exception as e:
            print(f"Error loading mask {path}: {e}")
            return np.zeros((256, 256), dtype=np.float32)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize each channel using global statistics."""
        if not self.global_stats:
            return image / 255.0 # Fallback
            
        # T2 channels (0 to stack_depth-1)
        t2_mean = self.global_stats["t2"]["mean"]
        t2_std = self.global_stats["t2"]["std"]
        # Avoid div by zero
        t2_std = t2_std if t2_std > 1e-6 else 1.0
        
        image[:self.stack_depth] = (image[:self.stack_depth] - t2_mean) / t2_std
        
        # ADC channel
        adc_mean = self.global_stats["adc"]["mean"]
        adc_std = self.global_stats["adc"]["std"]
        adc_std = adc_std if adc_std > 1e-6 else 1.0
        
        image[self.stack_depth] = (image[self.stack_depth] - adc_mean) / adc_std
        
        # Calc channel
        calc_mean = self.global_stats["calc"]["mean"]
        calc_std = self.global_stats["calc"]["std"]
        calc_std = calc_std if calc_std > 1e-6 else 1.0
        
        image[self.stack_depth+1] = (image[self.stack_depth+1] - calc_mean) / calc_std
        
        return image

    def __getitem__(self, idx):
        sample = self.samples[idx]
        case_id = sample["case_id"] # e.g. "class1/case_0144"
        case_dir = self.base_dir / case_id
        
        # 1. Load T2 Context Slices
        t2_slices = []
        # Use t2_context_indices from metadata
        # We need to handle if stack_depth != len(t2_context_indices)
        # For now, we assume they match or we take the middle 'stack_depth' ones
        context_indices = sample["t2_context_indices"]
        
        # If stack_depth is smaller than available context, center crop
        if self.stack_depth < len(context_indices):
            start = (len(context_indices) - self.stack_depth) // 2
            context_indices = context_indices[start : start + self.stack_depth]
        elif self.stack_depth > len(context_indices):
            # Pad or duplicate? For now, just duplicate edges
            diff = self.stack_depth - len(context_indices)
            # This is complex, but typically we expect them to match.
            # Fallback: repeat first/last
            context_indices = [context_indices[0]] * (diff//2) + context_indices + [context_indices[-1]] * (diff - diff//2)

        for slice_idx in context_indices:
            t2_file = f"{slice_idx:04d}.png"
            t2_path = case_dir / "t2" / t2_file
            t2_slices.append(self._load_image(t2_path))
            
        # 2. Load ADC and Calc
        # Filename is usually same as slice index for that modality
        # Metadata sample["files"] gives specific filename
        
        # ADC
        if sample.get("has_adc", False):
            adc_file = sample["files"].get("adc", f"{sample['slice_idx']:04d}.png")
            adc_path = case_dir / "adc" / adc_file
            adc_img = self._load_image(adc_path)
        else:
            adc_img = np.zeros((256, 256), dtype=np.float32)
            
        # Calc
        if sample.get("has_calc", False):
            calc_file = sample["files"].get("calc", f"{sample['slice_idx']:04d}.png")
            calc_path = case_dir / "calc" / calc_file
            calc_img = self._load_image(calc_path)
        else:
            calc_img = np.zeros((256, 256), dtype=np.float32)
            
        # Stack inputs: T2s + ADC + Calc
        # Shape: (stack_depth + 2, H, W)
        image_stack = np.stack(t2_slices + [adc_img, calc_img], axis=0)
        
        # 3. Load Masks
        # Prostate
        prostate_file = sample["files"].get("mask_prostate", f"{sample['slice_idx']:04d}.png")
        prostate_path = case_dir / "mask_prostate" / prostate_file
        mask_prostate = self._load_mask(prostate_path)
        
        # Target 1
        target1_file = sample["files"].get("mask_target1", f"{sample['slice_idx']:04d}.png")
        target1_path = case_dir / "mask_target1" / target1_file
        mask_target1 = self._load_mask(target1_path)
        
        # Target 2 (Infer path, assume same filename)
        target2_path = case_dir / "mask_target2" / target1_file # Use same filename
        mask_target2 = self._load_mask(target2_path)
        
        # Combine Targets (Union)
        mask_target = np.maximum(mask_target1, mask_target2)
        
        # Stack masks: (Prostate, Target)
        # Shape: (2, H, W)
        mask_stack = np.stack([mask_prostate, mask_target], axis=0)
        
        # 4. Augmentation
        if self.transform:
            # Albumentations expects (H, W, C) usually, or we implement custom
            # Here assuming transform takes (image, mask) as (C, H, W) or consistent
            # If standard torchvision/custom transform:
            image_stack, mask_stack = self.transform(image_stack, mask_stack)

        # 5. Normalize
        if self.normalize_enabled:
            image_stack = self._normalize(image_stack)
            
        # Convert to torch tensor
        # Ensure float32
        return torch.from_numpy(image_stack).float(), torch.from_numpy(mask_stack).float()

def create_multimodal_dataloader(
    metadata_path: Union[str, Path],
    stack_depth: int = 5,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    require_complete: bool = False,
    require_positive: bool = False,
    normalize: bool = True,
) -> DataLoader:
    dataset = MultiModalDataset(
        metadata_path=metadata_path,
        stack_depth=stack_depth,
        transform=transform,
        require_complete=require_complete,
        require_positive=require_positive,
        normalize=normalize,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    # Test the dataset
    import sys
    
    metadata_path = "data/aligned_v2/metadata.json"
    if not Path(metadata_path).exists():
        print(f"Metadata not found: {metadata_path}")
        sys.exit(1)
        
    print("Testing MultiModalDataset...")
    dataset = MultiModalDataset(
        metadata_path=metadata_path,
        stack_depth=5,
        normalize=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"Sample 0 Image Shape: {image.shape} (Expect 7, 256, 256)")
        print(f"Sample 0 Mask Shape: {mask.shape} (Expect 2, 256, 256)")
        print(f"Image Range: {image.min():.2f} to {image.max():.2f}")
        print(f"Mask Unique Values: {torch.unique(mask)}")
        
        # Check channels
        print("Channels:")
        print("  0-4: T2 Context")
        print("  5:   ADC")
        print("  6:   Calc")
