# Training Enhancements

The training script has been significantly enhanced with advanced features for better model development and monitoring.

## ⚡ Quick Start with Config File

All training parameters can now be managed via YAML configuration:

```bash
# Edit config.yaml to your needs, then run:
python service/train.py --config config.yaml

# Override specific parameters:
python service/train.py --config config.yaml --epochs 100 --batch_size 16
```

## 🚀 New Features

### 1. Advanced Learning Rate Schedulers

Multiple scheduler options to optimize training:

#### **ReduceLROnPlateau** (Default)
- Reduces LR when validation Dice plateaus
- Best for: Conservative, stable training
- Key params: `--scheduler_patience`, `--scheduler_factor`

```bash
python service/train.py --manifest data/processed/class2/manifest.csv \
    --scheduler reduce_on_plateau \
    --scheduler_patience 5 \
    --scheduler_factor 0.5
```

#### **OneCycleLR** (Recommended for Fast Training)
- Super-convergence with one-cycle policy
- Best for: Fast training, excellent results
- Key params: `--scheduler_max_lr_mult`, `--scheduler_warmup_pct`

```bash
python service/train.py --manifest data/processed/class2/manifest.csv \
    --scheduler onecycle \
    --lr 1e-4 \
    --scheduler_max_lr_mult 10
```

#### **Cosine Annealing with Warm Restarts**
- Periodic learning rate restarts
- Best for: Escaping local minima
- Key params: `--scheduler_t0`, `--scheduler_tmult`

```bash
python service/train.py --manifest data/processed/class2/manifest.csv \
    --scheduler cosine \
    --scheduler_t0 10 \
    --scheduler_tmult 2
```

#### **Other Schedulers**
- `cosine_simple`: Cosine annealing without restarts
- `step`: Step decay at fixed intervals
- `exponential`: Exponential decay
- `none`: No scheduler (constant LR)

### 2. Validation Visualizations

Automatic generation and logging of prediction visualizations during training.

#### Features:
- **Side-by-side comparison**: Input, Ground Truth, Predictions
- **Multi-class visualization**: All 3 classes shown with color coding
  - 🔴 **Red**: Prostate
  - 🟢 **Green**: Target1  
  - 🟠 **Blue**: Target2
- **Both overlay and contour views**
- **Logged to Aim** for easy tracking

#### Control Options:
```bash
# Save visualizations every 10 epochs (default: 5)
python service/train.py --manifest data/processed/class2/manifest.csv \
    --vis_every 10

# Save 8 samples per visualization (default: 4)
python service/train.py --manifest data/processed/class2/manifest.csv \
    --num_vis_samples 8

# Disable visualizations (set to large number)
python service/train.py --manifest data/processed/class2/manifest.csv \
    --vis_every 99999
```

### 3. Enhanced Logging

All metrics and visualizations are automatically logged to Aim:

- **Metrics**: Loss, Dice score, Learning rate (per epoch)
- **Visualizations**: Prediction samples with ground truth comparison
- **Hyperparameters**: All training parameters including scheduler config
- **System info**: GPU, CUDA version, platform details

#### View Results:
```bash
# Start Aim UI
aim up

# Navigate to: http://localhost:43800
```

## 📊 Complete Training Example

### Option 1: Using Config File (Recommended)

1. Edit `config.yaml`:
```yaml
manifest: data/processed/class2/manifest.csv
batch_size: 16
epochs: 100
lr: 0.0001
scheduler: onecycle
scheduler_max_lr_mult: 10
loss: dice_bce
vis_every: 5
num_vis_samples: 4
```

2. Run training:
```bash
python service/train.py --config config.yaml
```

### Option 2: Pure CLI

```bash
python service/train.py \
    --manifest data/processed/class2/manifest.csv \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --scheduler onecycle \
    --scheduler_max_lr_mult 10 \
    --loss dice_bce \
    --save_every 5 \
    --vis_every 5 \
    --num_vis_samples 4 \
    --num_workers 4
```

## 🔍 What Gets Visualized

Each visualization shows:

**Top Row - Ground Truth:**
1. Input grayscale image
2. GT overlay with colored masks
3. GT contours for each class

**Bottom Row - Predictions:**
1. Input grayscale image (same)
2. Prediction overlay with colored masks
3. Prediction contours for each class

**Legend:** Shows class names and colors

## 📈 Monitoring Training

### In Terminal:
- Progress bar shows: loss, dice, learning rate
- Learning rate changes are announced
- Visualization generation status

### In Aim UI:
- Compare different runs
- View metrics over time
- Browse validation predictions
- Track learning rate schedules

## 🎯 Benefits

1. **Better Training**: Advanced schedulers can significantly improve convergence
2. **Visual Feedback**: See model predictions evolve during training
3. **Easy Comparison**: Compare different runs in Aim UI
4. **Reproducibility**: All hyperparameters logged automatically
5. **Debugging**: Spot issues early with visualizations

## 🔧 Scheduler Comparison

| Scheduler | Speed | Stability | Best For |
|-----------|-------|-----------|----------|
| OneCycleLR | ⚡⚡⚡ | ⭐⭐ | Fast training, aggressive |
| Cosine | ⚡⚡ | ⭐⭐⭐ | Escaping local minima |
| ReduceLROnPlateau | ⚡ | ⭐⭐⭐⭐ | Conservative, stable |
| Step | ⚡⚡ | ⭐⭐⭐ | Traditional, predictable |

## 📝 Notes

- Visualizations are saved from the **first validation batch**
- The middle slice of the 2.5D stack is used for visualization
- Figures are automatically closed after logging to save memory
- Scheduler state is saved in checkpoints for proper resuming
- Learning rate is logged as a metric for tracking in Aim

## 🚨 Tips

1. **Use config.yaml** for easier parameter management
2. **Start with OneCycleLR** for fastest results
3. **Use ReduceLROnPlateau** if training is unstable
4. **Check visualizations early** to spot data issues
5. **Monitor learning rate** to ensure scheduler is working
6. **Save vis_every 5-10 epochs** to balance overhead vs feedback

## 📋 Config File Benefits

- **Version control**: Track configuration changes with git
- **Reproducibility**: Share exact training settings
- **Convenience**: No need for long CLI commands
- **Flexibility**: CLI args override config values
- **Organization**: All parameters in one place

Example workflow:
```bash
# Create config for experiment
cp config.yaml experiments/onecycle_fast.yaml

# Edit and run
python service/train.py --config experiments/onecycle_fast.yaml

# Override specific params
python service/train.py --config experiments/onecycle_fast.yaml --epochs 200
```

---

View training progress in real-time:
```bash
aim up
# Open: http://localhost:43800
```

