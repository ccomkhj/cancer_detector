# 2.5D MRI Segmentation Pipeline

This document explains how to use the 2.5D data pipeline for training segmentation models on the MRI dataset.

## Overview

The 2.5D approach combines the efficiency of 2D models with limited 3D context by stacking consecutive MRI slices along the channel dimension. This provides spatial context from neighboring slices while keeping computational requirements manageable.

## Components

### 1. Dataset (`dataset_2d5.py`)

**Main Class:** `MRI25DDataset`

Loads PNG slices from the processed data directory and stacks them to create 2.5D input tensors.

**Key Features:**
- Stacks k consecutive slices (e.g., k=5) along channel dimension
- Handles boundary cases with padding (reflect/replicate/edge)
- Supports both scale (0-1) and z-score normalization
- Compatible with masks from segmentation pipeline
- Efficient loading with caching support

**Usage:**

```python
from tools.dataset_2d5 import MRI25DDataset, create_dataloader

# Create dataset
dataset = MRI25DDataset(
    manifest_csv="data/processed/manifest_all.csv",
    stack_depth=5,              # Number of slices to stack
    image_size=(256, 256),      # Target size (H, W)
    normalize_method="scale",   # "scale" or "zscore"
    has_masks=True,             # Load masks if available
    filter_by_class=[1, 2],     # Optional: filter by PIRADS class
    padding_mode="reflect",     # How to handle edges
)

# Or use helper function
dataloader = create_dataloader(
    manifest_csv="data/processed/manifest_all.csv",
    batch_size=8,
    stack_depth=5,
    image_size=(256, 256),
    shuffle=True,
    num_workers=4,
)

# Iterate
for images, masks in dataloader:
    # images: [batch_size, stack_depth, H, W], torch.float32
    # masks: [batch_size, 1, H, W], torch.float32
    outputs = model(images)
```

### 2. Transforms (`transforms_2d5.py`)

**Data Augmentation:**

Provides both custom PyTorch transforms and MONAI-based transforms for medical image augmentation.

**Available Transforms:**
- `RandomHorizontalFlip` - Random horizontal flipping
- `RandomVerticalFlip` - Random vertical flipping
- `RandomRotate90` - Random 90° rotations
- `RandomIntensityScale` - Scale intensity values
- `RandomIntensityShift` - Shift intensity values
- `RandomGaussianNoise` - Add Gaussian noise

**Usage:**

```python
from tools.transforms_2d5 import get_train_transforms, get_val_transforms

# Training transforms with augmentation
train_transforms = get_train_transforms(
    horizontal_flip_prob=0.5,
    vertical_flip_prob=0.5,
    rotate90_prob=0.5,
    intensity_scale=True,
    intensity_shift=True,
    gaussian_noise=True,
)

# Validation transforms (no augmentation)
val_transforms = get_val_transforms()

# Use in dataset
dataset = MRI25DDataset(
    manifest_csv="...",
    transform=train_transforms,
)
```

### 3. Model Testing (`test_2d5_models.py`)

Tests both SMP ResUNet and MONAI SegResNet to verify compatibility.

**Usage:**

```bash
# Test both models
python tools/test_2d5_models.py

# Test specific model
python tools/test_2d5_models.py --model smp
python tools/test_2d5_models.py --model monai

# Test with specific manifest
python tools/test_2d5_models.py --manifest data/processed/class1/manifest.csv

# Custom batch size and stack depth
python tools/test_2d5_models.py --batch-size 8 --stack-depth 7
```

## Data Format

### Input Format (Current Pipeline)

Your existing pipeline produces:
- **Storage:** PNG files (per-slice, 2D)
- **Data type:** uint8 (0-255 intensity values)
- **Organization:** Manifest CSV with paths to individual slice files
- **Location:** `data/processed/class{N}/case_XXXX/{series_uid}/images/NNNN.png`
- **Masks:** Binary PNG (0/255) in `labels/` directories

### Model Input Format (2.5D)

The dataset converts to:
- **Tensor shape:** `[batch_size, stack_depth, height, width]`
- **Data type:** `torch.float32`
- **Value range:** [0, 1] (scale) or normalized (zscore)
- **Example:** `[8, 5, 256, 256]` = 8 samples, 5 stacked slices, 256×256 pixels

## Supported Models

### 1. SMP ResUNet (Segmentation Models PyTorch)

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",      # Or: resnet50, efficientnet-b0, etc.
    encoder_weights="imagenet",   # Or None for random init
    in_channels=5,                # Stack depth
    classes=1,                    # Binary segmentation
    activation=None,              # Apply sigmoid/softmax later
)

# Installation
pip install segmentation-models-pytorch
```

**Pros:**
- Pre-trained encoders from ImageNet
- Many encoder choices (ResNet, EfficientNet, etc.)
- Easy to use
- Well-documented

### 2. MONAI SegResNet

```python
from monai.networks.nets import SegResNet

model = SegResNet(
    spatial_dims=2,        # 2D mode for 2.5D
    in_channels=5,         # Stack depth
    out_channels=1,        # Binary segmentation
    init_filters=32,       # Base number of filters
    dropout_prob=0.2,      # Dropout for regularization
)

# Installation
pip install monai
```

**Pros:**
- Designed for medical imaging
- Robust architecture
- Good performance on small datasets
- MONAI ecosystem integration

### Both Models Accept Same Input!

```python
# Both models work with the same input format
images, masks = next(iter(dataloader))  # [B, 5, 256, 256]

# SMP
output_smp = model_smp(images)  # [B, 1, 256, 256]

# MONAI
output_monai = model_monai(images)  # [B, 1, 256, 256]
```

## Training Configuration

### Recommended Settings

```python
# Data
STACK_DEPTH = 5          # Odd number for symmetric context
IMAGE_SIZE = (256, 256)  # Or (320, 320) for larger models
BATCH_SIZE = 8           # Adjust based on GPU memory

# Training
LEARNING_RATE = 1e-4
EPOCHS = 100
OPTIMIZER = "AdamW"
SCHEDULER = "CosineAnnealing"

# Loss
LOSS = "Dice + BCE"      # Dice loss + Binary Cross Entropy

# Augmentation
HORIZONTAL_FLIP = 0.5
VERTICAL_FLIP = 0.5
ROTATE_90 = 0.5
INTENSITY_AUGMENT = True
```

### Loss Functions

For binary segmentation:

```python
import torch.nn as nn
from monai.losses import DiceLoss

# BCE with Logits (built-in)
bce_loss = nn.BCEWithLogitsLoss()

# Dice Loss (MONAI)
dice_loss = DiceLoss(sigmoid=True)

# Combined Loss
class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        return (self.dice_weight * self.dice(pred, target) + 
                self.bce_weight * self.bce(pred, target))
```

## Example Training Script

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tools.dataset_2d5 import MRI25DDataset
from tools.transforms_2d5 import get_train_transforms, get_val_transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create datasets
train_dataset = MRI25DDataset(
    manifest_csv="data/processed/manifest_all.csv",
    stack_depth=5,
    image_size=(256, 256),
    normalize_method="scale",
    transform=get_train_transforms(),
    filter_by_class=[2, 3, 4],  # Exclude low-risk cases
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# Create model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=5,
    classes=1,
).to(device)

# Optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
```

## Performance Tips

### GPU Memory

If you run out of GPU memory:
1. Reduce `BATCH_SIZE` (try 4 or 2)
2. Reduce `IMAGE_SIZE` (try 128×128)
3. Reduce `STACK_DEPTH` (try 3)
4. Use mixed precision training (`torch.cuda.amp`)

### Data Loading

For faster training:
1. Use `num_workers=4` or more
2. Set `pin_memory=True` if using GPU
3. Consider caching frequently used data

### Augmentation

Start with minimal augmentation and gradually add:
1. Basic: Flips and rotations
2. Intermediate: Add intensity augmentation
3. Advanced: Add elastic deformations (if using MONAI)

## Comparison with Current Pipeline

| Aspect | Current Pipeline | 2.5D Pipeline |
|--------|------------------|---------------|
| **Storage** | PNG files on disk | Same (PNG files) |
| **Loading** | Load single slices | Stack multiple slices |
| **Data type** | uint8 (0-255) | float32 (normalized) |
| **Model input** | Not defined | [B, k, H, W] tensors |
| **Context** | Single slice | k neighboring slices |
| **Compatibility** | Generic | SMP + MONAI ready |

**Key advantage:** No changes needed to your existing data pipeline! The 2.5D dataset simply reads your PNG files and stacks them appropriately.

## Troubleshooting

### "Manifest file not found"

Run the DICOM converter first:
```bash
python tools/dicom_converter.py --all
```

### "Dataset is empty"

Check that:
1. Manifest CSV exists and has rows
2. Image paths in manifest are valid
3. Filter settings aren't too restrictive

### "Model input size mismatch"

Ensure:
1. `in_channels` matches `stack_depth`
2. Image size is consistent
3. Batch dimension is present

### "Out of memory"

Try:
1. Smaller batch size
2. Smaller image size
3. Fewer workers
4. Gradient accumulation

## Next Steps

1. **Install dependencies:**
   ```bash
   pip install torch torchvision
   pip install segmentation-models-pytorch
   pip install monai
   ```

2. **Test the pipeline:**
   ```bash
   python tools/test_2d5_models.py
   ```

3. **Implement training script** based on the example above

4. **Experiment with configurations:**
   - Different stack depths (3, 5, 7)
   - Different models (ResNet34, ResNet50, EfficientNet)
   - Different loss functions (Dice, BCE, Focal)

5. **Evaluate and iterate:**
   - Monitor validation metrics (Dice, IoU)
   - Visualize predictions
   - Tune hyperparameters

## References

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [MONAI Documentation](https://docs.monai.io/)
- [PyTorch Data Loading](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

