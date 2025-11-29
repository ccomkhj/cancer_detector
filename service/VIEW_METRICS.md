# 📊 How to View Your Metrics in Aim

## ✅ Your Aim Setup is Working!

You have **2 runs** logged in Aim. Now let's visualize them!

## 🎯 Step-by-Step Guide

### Step 1: Open Aim UI in Browser

Aim is already running at: **http://127.0.0.1:43800**

👉 **Open this URL in your browser** (Chrome, Safari, Firefox, etc.)

### Step 2: Navigate to Metrics Page

Once the Aim UI loads:

1. Look at the **LEFT SIDEBAR** (dark/gray panel on the left)
2. You'll see several menu items:
   - 🏠 Home
   - 📊 **Metrics** ← **CLICK HERE**
   - 🖼️ Images
   - 📝 Text
   - 🎵 Audio
   - etc.

3. **Click on "Metrics"** (📊)

### Step 3: You Should See Your Metrics!

After clicking "Metrics", you should see:

**LEFT PANEL (Metrics Search):**
```
🔍 Search

📊 Metrics
  □ loss      ← Check this box!
  □ dice      ← Check this box!
```

**MAIN PANEL (Charts):**
- Once you check the boxes, **charts will appear**!
- You'll see line plots showing how loss and dice change over epochs

### Step 4: Separate Train vs Validation

To see train and validation as different lines:

1. Look for **"Group by"** dropdown/button near the top of the charts
2. Click it and select: **`context.subset`**
3. Now you'll see:
   - 🔵 One color for **train** metrics
   - 🟢 Another color for **val** metrics

### Step 5: Interact with Charts

You can now:
- **Zoom**: Click and drag on chart to zoom in
- **Hover**: Hover over lines to see exact values
- **Smooth**: Adjust smoothing slider to reduce noise
- **Compare**: Check/uncheck runs to compare them

---

## 🐛 If You Don't See Metrics

### Problem 1: "No metrics available"

**Solution:**
1. Make sure you're on the **Metrics** page (left sidebar)
2. Check that runs are selected in the left panel
3. Try **refreshing the browser** (F5 or Cmd+R)

### Problem 2: "Metrics list is empty"

**Check if runs exist:**
```bash
cd /Users/huijokim/personal/mri
/opt/miniconda3/envs/mri/bin/aim runs ls
```

You should see: `Total 2 runs.`

If you see 0 runs, re-run the test:
```bash
/opt/miniconda3/envs/mri/bin/python service/test_aim_logging.py
```

### Problem 3: "Charts not appearing"

1. Make sure you've **checked the boxes** next to metric names (loss, dice)
2. **Scroll down** - charts might be below the fold
3. Try clicking **"Refresh"** button in the UI
4. Check browser console for errors (F12)

---

## 🎨 Making Nice Visualizations

### Color by Context (Train vs Val)

```
Top toolbar:
  Group by: context.subset
  Color by: context.subset
```

This gives you:
- Train metrics in one color
- Val metrics in another color

### Compare Hyperparameters

```
Group by: run.hparams.lr
```

This groups runs by learning rate so you can see which LR works best!

### Focus on Best Runs

Use the search box:
```
run.hparams.batch_size == 32
```

Only shows runs with batch size 32.

---

## 🏃 Ready to Train for Real?

Once you can see the test metrics, start your actual training:

```bash
cd /Users/huijokim/personal/mri

# Activate environment (in a terminal)
conda activate mri

# Start training
python service/train.py --manifest data/processed/class2/manifest.csv

# Keep Aim UI open in browser to watch live!
```

Metrics will appear **in real-time** as your model trains! 🚀

---

## 📸 What the UI Should Look Like

### Metrics Page Structure:

```
┌─────────────────────────────────────────────────────────┐
│  🔹 Aim                                                  │
├─────────────┬───────────────────────────────────────────┤
│             │  METRICS EXPLORER                          │
│  LEFT MENU  │  ┌─────────────────────────────────────┐  │
│  ┌───────┐  │  │ loss                                │  │
│  │ Home  │  │  │ [Line chart showing decreasing]    │  │
│  │Metric*│  │  │                                     │  │
│  │ Image │  │  └─────────────────────────────────────┘  │
│  │ Text  │  │  ┌─────────────────────────────────────┐  │
│  └───────┘  │  │ dice                                │  │
│             │  │ [Line chart showing increasing]    │  │
│  SEARCH     │  │                                     │  │
│  ┌───────┐  │  └─────────────────────────────────────┘  │
│  │☑ loss │  │                                            │
│  │☑ dice │  │                                            │
│  └───────┘  │                                            │
└─────────────┴───────────────────────────────────────────┘
```

---

## 🆘 Still Not Seeing Metrics?

**Tell me:**
1. Are you on the http://127.0.0.1:43800 page?
2. Do you see "Metrics" in the left sidebar?
3. When you click "Metrics", what do you see?
4. Is there a message or error?

**Quick diagnostic:**
```bash
# Check Aim is installed
/opt/miniconda3/envs/mri/bin/python -c "import aim; print('Aim installed ✓')"

# Check runs exist  
cd /Users/huijokim/personal/mri
/opt/miniconda3/envs/mri/bin/aim runs ls

# Check .aim directory
ls -la .aim/
```

Let me know what you see! 🔍

