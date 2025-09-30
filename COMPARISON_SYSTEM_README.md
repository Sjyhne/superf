# Comparison System - Clean & Working

## 🎯 **What This System Does**

Creates comprehensive comparison images with the exact layout you requested:
- **LR Sample** | **Bilinear** | **Lafanetre** | **NIR** | **Ours (MSE)** | **HR Reference**

## 📁 **Working Files**

### **Main Scripts:**
1. **`create_combined_comparison.py`** - Main script for single dataset comparisons
2. **`create_data_mixed_comparison.py`** - Mixed dataset comparisons (SyntheticBurstVal + Data folder)
3. **`create_simple_data_mixed.py`** - Simple interface for mixed datasets

### **Other Files:**
- `create_comprehensive_comparison.py` - Original comprehensive script
- `create_data_from_single_image.py` - Data generation script
- `create_separate_metric_plots.py` - Metric plotting script

## 🚀 **How to Use**

### **1. Single Dataset Comparison (SyntheticBurstVal only):**
```bash
python create_combined_comparison.py --sample_indices 0 1 2 3 --scale_factor 4 --output_path my_comparison.png
```

### **2. Mixed Dataset Comparison (SyntheticBurstVal + Data folder):**
```bash
# List available samples
python create_data_mixed_comparison.py --list_samples

# Create mixed comparison
python create_simple_data_mixed.py '0006,0013' 'Amnesty POI-7-1-3_rgb,UNHCR-NERs009690_rgb' mixed_comparison.png
```

### **3. Available Sample Types:**
- **SyntheticBurstVal**: `0006`, `0013`, `0014`, `0017`, `0022`, `0024`, `0062`, `0065`, `0084`, `0089`, etc.
- **Data Folder**: `ASMSpotter-28-1-2_rgb`, `Amnesty POI-7-1-3_rgb`, `UNHCR-NERs009690_rgb`, `Landcover-1055780_rgb`, etc.

## ✅ **Features**

- **Real LR Samples**: Uses actual burst images (48x48) and data folder samples
- **Clean Layout**: Method names above images, no overlapping labels
- **Flexible Mixing**: Any combination of synthetic and data samples
- **Index-Based Matching**: Automatically matches samples across all folders
- **Easy to Use**: Simple command-line interface

## 📊 **Output**

Creates comparison images with:
- 6 columns: LR Sample | Bilinear | Lafanetre | NIR | Ours (MSE) | HR Reference
- Multiple rows (one per sample)
- Clean, professional layout
- High resolution (300 DPI)

## 🎉 **Ready to Use!**

The system is now clean and working perfectly. All obsolete files have been removed, and you have a streamlined comparison system that does exactly what you need!
