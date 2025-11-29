# Experiment Tracking with Aim

This project uses [Aim](https://github.com/aimhubio/aim) for tracking training experiments and visualizing results.

## Installation

```bash
pip install aim
```

## Usage

### 1. Training with Aim Logging

The training script automatically logs all metrics to Aim:

```bash
python service/train.py --manifest data/processed/class2/manifest.csv
```

What gets tracked:
- **Hyperparameters**: batch size, learning rate, loss function, model architecture, etc.
- **Training metrics**: loss and Dice score per epoch
- **Validation metrics**: loss and Dice score per epoch  
- **Checkpoints**: saved model checkpoints with metrics
- **Best model**: best validation Dice score and epoch
- **System info**: GPU, CUDA version, Python version, etc.

### 2. Visualizing Results

After training starts (or completes), launch the Aim UI:

```bash
aim up
```

This will open a web interface (usually at `http://localhost:43800`) where you can:

- **Compare runs**: View metrics side-by-side across different experiments
- **Plot metrics**: Interactive charts for loss, Dice score, etc.
- **Filter runs**: Query runs by hyperparameters
- **Analyze trends**: See how different hyperparameters affect performance

### 3. Viewing Specific Experiments

List all experiments:

```bash
aim runs list
```

View details of a specific run:

```bash
aim runs info <RUN_HASH>
```

### 4. Comparing Multiple Runs

The Aim UI makes it easy to compare runs:

1. Launch Aim: `aim up`
2. Go to the **Metrics Explorer**
3. Select multiple runs
4. Group/color by hyperparameters (e.g., learning rate, batch size)
5. Compare training curves

### 5. Repository Location

Aim data is stored in `.aim/` directory in the project root. This is automatically excluded from git.

## Example Workflow

```bash
# Train with different hyperparameters
python service/train.py --manifest data/processed/class2/manifest.csv --lr 1e-4 --batch_size 8
python service/train.py --manifest data/processed/class2/manifest.csv --lr 1e-5 --batch_size 16
python service/train.py --manifest data/processed/class2/manifest.csv --lr 1e-3 --loss dice

# View and compare results
aim up
```

## Key Features

### Run Naming
Each run gets an automatic descriptive name like:
```
class2_bs8_lr0.0001_dice_bce_1129_1430
```

Format: `{dataset}_{batch_size}_{learning_rate}_{loss}_{timestamp}`

### Logged Metrics

**Per Epoch:**
- `train/loss` - Training loss
- `train/dice` - Training Dice score
- `val/loss` - Validation loss
- `val/dice` - Validation Dice score

**Best Model:**
- `best_metrics/val_dice` - Best validation Dice
- `best_epoch` - Epoch where best model was found

**Checkpoints:**
- Checkpoint paths and associated metrics

## Tips

1. **Keep Aim UI running** during training to see live updates
2. **Use tags** to organize related experiments
3. **Add notes** to runs in the UI to remember important details
4. **Export charts** from the UI for presentations/papers
5. **Query runs** by hyperparameters to find best configurations

## Troubleshooting

### Aim not found
```bash
pip install aim
```

### Port already in use
```bash
aim up --port 8080  # Use different port
```

### Clear old runs
```bash
# Be careful - this deletes all data!
rm -rf .aim/
```

## Learn More

- [Aim Documentation](https://aimstack.readthedocs.io/)
- [Aim GitHub](https://github.com/aimhubio/aim)
- [Aim Examples](https://github.com/aimhubio/aim/tree/main/examples)

