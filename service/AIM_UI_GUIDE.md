# How to View Metrics in Aim UI

## Quick Test

First, let's verify Aim logging is working:

```bash
cd /Users/huijokim/personal/mri
python service/test_aim_logging.py
```

This will create a test run with fake metrics.

## Viewing Metrics in Aim UI

### Step 1: Make sure Aim is running

```bash
aim up
```

You should see: `Open http://127.0.0.1:43800`

### Step 2: Open in Browser

Open: **http://127.0.0.1:43800**

### Step 3: Navigate to Metrics Explorer

In the Aim UI:
1. Look at the **left sidebar**
2. Click on **"Metrics"** (should have a chart icon 📊)

### Step 4: Select Metrics to Visualize

You should now see:

**Left Panel: Search & Filter**
- List of all available metrics (loss, dice, etc.)
- Checkboxes to select which metrics to display

**Main Panel: Charts**
- Interactive line charts
- Each selected metric shows as a plot

### Step 5: Configure Visualization

**To see Train vs Val separately:**
1. Click the **"Group"** button (top of chart area)
2. Select **"context.subset"** from dropdown
3. Now you'll see train and val as different colored lines!

**To compare multiple runs:**
1. In left panel, you'll see all your runs
2. Check/uncheck runs to show/hide them
3. Different runs show as different line styles

**Other useful options:**
- **Smooth**: Adjust smoothing slider to reduce noise
- **X-axis**: Change to "step", "epoch", or "time"
- **Y-axis**: Toggle between linear/log scale
- **Zoom**: Click and drag on chart to zoom in

## Troubleshooting

### "No metrics found"

**Possible reasons:**

1. **No training runs yet**
   - Start training: `python service/train.py --manifest data/processed/class2/manifest.csv`
   - Metrics appear in real-time as training progresses

2. **Wrong repository**
   - Check that Aim is using the right repo path: `.aim/` in your project root
   - Run: `aim up --repo /Users/huijokim/personal/mri/.aim`

3. **Aim not logging**
   - Check console output during training for: `✓ Aim logging initialized`
   - If you see "Aim not installed" warning, run: `pip install aim`

### "Charts not updating"

- **Refresh the browser** (F5 or Cmd+R)
- Aim shows **live updates** - metrics appear as they're logged
- Sometimes you need to click "Refresh" button in the UI

### "Only seeing some metrics"

- Make sure you've **selected the metrics** you want to see (checkboxes in left panel)
- Check that your training script is actually logging those metrics

## Example: What You Should See

After running the test script or starting training, you should see:

### Metrics Tab

**Left Panel:**
```
📊 Metrics
  ☑ loss
  ☑ dice

🏷️ Context
  ☑ train
  ☑ val

🔍 Runs
  ☑ test_run (or your training run name)
```

**Main Panel:**
- **Two charts** (one for loss, one for dice)
- Each chart shows **two lines** (train in one color, val in another)
- Lines should show **downward trend for loss**, **upward trend for dice**

## Tips for Better Visualization

### 1. Color by context
```
Group by: context.subset
Color by: context.subset
```
This makes train/val visually distinct.

### 2. Compare different hyperparameters
```
Group by: run.hparams.lr
```
This groups runs by learning rate for easy comparison.

### 3. Focus on best runs
Use the search bar:
```
run.val_dice > 0.8
```
Only shows runs with validation dice > 0.8

### 4. Export charts
- Click the **camera icon** (top right of chart)
- Downloads PNG of the current visualization

## Next Steps

Once you see metrics working:

1. **Start real training:**
   ```bash
   python service/train.py --manifest data/processed/class2/manifest.csv
   ```

2. **Watch metrics in real-time:**
   - Keep Aim UI open in browser
   - Refresh to see latest updates
   - Metrics appear every epoch

3. **Run multiple experiments:**
   ```bash
   # Different learning rates
   python service/train.py --manifest data/processed/class2/manifest.csv --lr 1e-4
   python service/train.py --manifest data/processed/class2/manifest.csv --lr 1e-5
   
   # Different losses
   python service/train.py --manifest data/processed/class2/manifest.csv --loss dice
   python service/train.py --manifest data/processed/class2/manifest.csv --loss dice_bce
   ```

4. **Compare all runs in Aim:**
   - All runs appear in the same UI
   - Easy to see which hyperparameters work best!

## Still Having Issues?

Run the diagnostic:
```bash
# Check Aim is installed
python -c "import aim; print(f'Aim version: {aim.__version__}')"

# Check .aim directory exists
ls -la .aim/

# List all runs
aim runs list
```

Show me the output and I can help debug! 🔧

