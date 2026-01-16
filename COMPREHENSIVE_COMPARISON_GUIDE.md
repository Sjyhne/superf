# Comprehensive Comparison Guide

This guide explains how to create comprehensive comparison images that gather results from multiple method folders into one unified view.

## Overview

The scripts create comparison images that show:
1. **LR Sample** - Original low-resolution input
2. **Bilinear** - Bilinear upsampling baseline with PSNR
3. **HR Reference** - Ground truth high-resolution image
4. **Method Results** - Outputs from different methods with metrics

## Scripts Available

### 1. `create_comprehensive_comparison.py`
Main script for creating comparisons. You specify:
- Sample IDs and their datasets
- Method folders to include
- Display names for methods

### 2. `example_comprehensive_comparison.py`
Example showing how to use the main script with specific samples and methods.

### 3. `auto_discover_comparisons.py`
Automatically discovers available methods and creates comparisons.

## Usage Examples

### Example 1: Specific Samples and Methods

```bash
python create_comprehensive_comparison.py \
    --samples "UNHCR-GNBs001116" "0" "1" \
    --datasets "worldstrat_test" "burst_synth" "burst_synth" \
    --method_folders "burst_synth_df2_inr_gnll_2" "burst_synth_df4_inr_gnll" "single_samples/worldstrat_test" \
    --method_names "Burst DF2 INR GNLL" "Burst DF4 INR GNLL" "Single Sample" \
    --output_dir "my_comparisons"
```

### Example 2: Auto-Discover Methods

```bash
python auto_discover_comparisons.py \
    --samples "UNHCR-GNBs001116" "0" \
    --datasets "worldstrat_test" "burst_synth" \
    --max_samples_per_method 3
```

### Example 3: Run Example Script

```bash
python example_comprehensive_comparison.py
```

## Method Folder Structure

The scripts expect method folders to contain sample directories with:
- `comparison.png` (preferred) OR
- `prediction_aligned.png` / `model_output_aligned.png`
- `ground_truth.png`
- `metrics.json` (for PSNR display)

## Sample Naming Patterns

The scripts look for samples using these patterns:
- `sample_000`, `sample_001`, etc.
- `0`, `1`, `2`, etc.
- `UNHCR-GNBs001116`
- `Amnesty POI-17-2-3`

## Output

Each comparison image shows:
- **Grid layout** with LR, Bilinear, HR, and method results
- **PSNR metrics** displayed in titles
- **High resolution** (300 DPI) for publication quality
- **Organized by sample** with dataset information

## Troubleshooting

### Common Issues

1. **"No method results found"**
   - Check that sample IDs match the folder names
   - Verify method folders exist and contain sample directories

2. **"Failed to extract images"**
   - Ensure dataset paths are correct
   - Check that sample exists in the dataset

3. **"Not enough methods found"**
   - Some methods might not have results for that specific sample
   - Try different sample IDs or check method folders

### Debug Mode

Add `--verbose` flag to see detailed processing information.

## Customization

You can modify the scripts to:
- Add more metrics (SSIM, LPIPS)
- Change image layout
- Add method-specific styling
- Include additional baseline methods

## File Structure

```
comprehensive_comparisons/
├── comprehensive_comparison_UNHCR-GNBs001116_worldstrat_test.png
├── comprehensive_comparison_0_burst_synth.png
└── comprehensive_comparison_1_burst_synth.png
```
