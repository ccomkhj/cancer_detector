# Validation Tools

Scripts for testing, validation, and diagnostics of the MRI segmentation pipeline.

## 📋 Scripts

### `validate_2d5_setup.py` - Validate Training Setup

Validates that the 2.5D training setup is correctly configured.

```bash
python tools/validation/validate_2d5_setup.py
```

Checks:
- Dataset loading
- Image dimensions
- Mask alignment
- Stack depth configuration

---

### `validate_all_masks.py` - Validate Mask Integrity

Validates segmentation masks across all classes.

```bash
python tools/validation/validate_all_masks.py
```

Checks:
- Mask file existence
- Proper binary format
- Alignment with images
- Coverage statistics

---

### `test_dataset_basic.py` - Test Dataset Loaders

Tests basic dataset loading functionality.

```bash
python tools/validation/test_dataset_basic.py
```

Tests:
- PyTorch DataLoader creation
- Batch generation
- Transform application
- Data shapes and types

---

### `test_2d5_models.py` - Test Model Architectures

Tests model architectures with sample data.

```bash
python tools/validation/test_2d5_models.py
```

Tests:
- Model instantiation
- Forward pass
- Output shapes
- GPU compatibility

---

### `analyze_data.py` - Data Distribution Analysis

Analyzes data distribution and statistics.

```bash
python tools/validation/analyze_data.py
```

Provides:
- Class distribution
- Image statistics (mean, std)
- Mask coverage statistics
- Data quality metrics

---

### `diagnose_alignment.py` - Diagnose Alignment Issues

Diagnoses and visualizes mask-to-image alignment issues.

```bash
python tools/validation/diagnose_alignment.py
```

Helps identify:
- Misaligned masks
- Transform errors
- Coordinate system issues
- Geometry problems

---

## 🚀 Quick Validation Workflow

### Before Training

```bash
# 1. Validate complete setup
python tools/validation/validate_2d5_setup.py

# 2. Check all masks
python tools/validation/validate_all_masks.py

# 3. Analyze data distribution
python tools/validation/analyze_data.py

# 4. Test dataset loading
python tools/validation/test_dataset_basic.py
```

### After Changes

```bash
# If you modified masks or alignment
python tools/validation/diagnose_alignment.py

# If you modified models
python tools/validation/test_2d5_models.py
```

---

## ✅ Recommended Validation Steps

### 1. After Data Preprocessing
```bash
python service/validate_data.py
# or
python tools/validation/validate_all_masks.py
```

### 2. Before First Training
```bash
python tools/validation/validate_2d5_setup.py
python tools/validation/test_dataset_basic.py
```

### 3. After Model Changes
```bash
python tools/validation/test_2d5_models.py
```

### 4. Troubleshooting Alignment
```bash
python tools/validation/diagnose_alignment.py
```

---

## 🔍 Understanding Validation Outputs

### Successful Validation
```
✓ Dataset loaded: 1845 samples
✓ Image shape: (512, 512)
✓ Stack depth: 5
✓ Masks aligned: 100%
✓ All checks passed!
```

### Common Issues

**Missing Masks:**
```
⚠️  Warning: 50 images have no masks
```
→ Check if masks were generated for all cases

**Alignment Issues:**
```
❌ Error: Mask shape mismatch
```
→ Run `diagnose_alignment.py` for details

**Data Loading Errors:**
```
❌ Error: Cannot load dataset
```
→ Check manifest.csv and file paths

---

## 💡 Tips

1. **Run validation after each preprocessing step**
2. **Check `validation_results/` directory** for visual outputs
3. **Use `--help` flag** on any script for detailed options
4. **Validate on a subset first** to save time
5. **Check alignment visually** before full training

---

## 📚 Related Tools

- **Data Validation**: `service/validate_data.py` (comprehensive)
- **Visualization**: `tools/preprocessing/visualize_overlay_masks.py`
- **Analysis**: `tools/validation/analyze_data.py`

---

## 🐛 Troubleshooting

### "Dataset is empty"
- Check manifest.csv exists
- Verify file paths in manifest
- Run `validate_all_masks.py`

### "Mask alignment error"
- Run `diagnose_alignment.py`
- Check if DICOM geometry was used
- Verify transform pipeline

### "Model test fails"
- Check PyTorch installation
- Verify CUDA availability
- Test with smaller batch size

---

For comprehensive data validation, use:
```bash
python service/validate_data.py
```

This runs multiple validation checks and creates visual outputs in `validation_results/`.

