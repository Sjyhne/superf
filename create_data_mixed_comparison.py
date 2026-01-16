#!/usr/bin/env python3
"""
Create mixed dataset comparisons using SyntheticBurstVal and data folder samples.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_image_safely(image_path):
    """Safely load an image and return as numpy array."""
    if image_path is None or not image_path.exists():
        return None
    
    try:
        img = plt.imread(str(image_path))
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        return np.clip(img, 0, 1)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_available_samples(base_path):
    """Get available samples from each dataset."""
    base_path = Path(base_path)
    
    # Get SyntheticBurstVal samples
    burst_dir = base_path / "SyntheticBurstVal" / "bursts"
    synthetic_samples = sorted([d.name for d in burst_dir.iterdir() if d.is_dir()])
    
    # Get data folder samples
    data_dir = base_path / "data"
    data_samples = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    return synthetic_samples, data_samples

def load_synthetic_burst_lr(sample_name, base_path):
    """Load LR sample from SyntheticBurstVal dataset."""
    base_path = Path(base_path)
    burst_path = base_path / "SyntheticBurstVal" / "bursts" / sample_name / "im_raw_00.png"
    
    if not burst_path.exists():
        print(f"❌ SyntheticBurstVal LR not found: {burst_path}")
        return None
    
    try:
        # Load and process the raw burst image like in data.py
        im = cv2.imread(str(burst_path), cv2.IMREAD_UNCHANGED)
        im_t = im.astype(np.float32) / (2**14)

        # Extract RGGB channels
        R = im_t[..., 0]
        G1 = im_t[..., 1]
        G2 = im_t[..., 2]
        B = im_t[..., 3]
        
        # Average the two green channels
        G = (G1 + G2) / 2
        
        # Create RGB image
        rgb = np.stack([R, G, B], axis=-1)
        
        # Apply white balance and gamma correction
        wb_gains = np.array([2.0, 1.0, 1.5])
        rgb = rgb * wb_gains
        gamma = 2.2
        rgb = np.power(np.maximum(rgb, 0), 1.0/gamma)
        rgb = np.clip(rgb, 0, 1)
        
        print(f"✅ SyntheticBurstVal LR loaded: {rgb.shape}")
        return rgb
        
    except Exception as e:
        print(f"❌ Error loading SyntheticBurstVal LR: {e}")
        return None

def load_data_folder_lr(sample_name, base_path):
    """Load LR sample from data folder."""
    base_path = Path(base_path)
    
    # Try different scale/shift combinations for the data folder
    possible_paths = [
        base_path / "data" / sample_name / "scale_1_shift_4.0px_aug_light" / "sample_00.png",
        base_path / "data" / sample_name / "scale_1_shift_2.0px_aug_light" / "sample_00.png",
        base_path / "data" / sample_name / "scale_1_shift_1.0px_aug_light" / "sample_00.png",
        base_path / "data" / sample_name / "scale_2_shift_4.0px_aug_light" / "sample_00.png",
        base_path / "data" / sample_name / "scale_4_shift_4.0px_aug_light" / "sample_00.png"
    ]
    
    for lr_path in possible_paths:
        if lr_path.exists():
            print(f"✅ Data folder LR found: {lr_path}")
            return load_image_safely(lr_path)
    
    print(f"❌ Data folder LR not found for {sample_name}")
    print(f"Tried paths:")
    for path in possible_paths:
        print(f"  - {path}")
    return None

def load_mixed_sample_data(sample_config, base_path="/raid/home/sandej17/satellite_sr"):
    """Load data for a sample from mixed datasets."""
    base_path = Path(base_path)
    dataset_type = sample_config['dataset_type']
    
    print(f"Loading {dataset_type} sample: {sample_config['sample_name']}")
    
    # Load all images
    data = {}
    
    # 1. Bilinear
    handheld_dir = base_path / f"handheld/results_synthetic_scale_4_shift_1/handheld/SyntheticBurst"
    bilinear_path = handheld_dir / sample_config['handheld_sample'] / "aligned_baseline.png"
    data['bilinear'] = load_image_safely(bilinear_path)
    
    # 2. Lafanetre
    lafanetre_path = handheld_dir / sample_config['handheld_sample'] / "aligned_output.png"
    data['lafanetre'] = load_image_safely(lafanetre_path)
    
    # 3. NIR
    nir_dir = base_path / "burst_synth_df4_nir"
    nir_path = nir_dir / sample_config['nir_sample'] / "prediction_aligned.png"
    data['nir'] = load_image_safely(nir_path)
    
    # 4. Ours (MSE)
    ours_dir = base_path / "burst_synth_df4_inr"
    ours_path = ours_dir / sample_config['ours_sample'] / "prediction_aligned.png"
    data['ours'] = load_image_safely(ours_path)
    
    # 5. HR Reference
    hr_path = ours_dir / sample_config['ours_sample'] / "ground_truth.png"
    data['hr'] = load_image_safely(hr_path)
    
    # 6. LR Sample - load based on dataset type
    if dataset_type == 'synthetic_burst':
        data['lr'] = load_synthetic_burst_lr(sample_config['sample_name'], base_path)
    elif dataset_type == 'data_folder':
        data['lr'] = load_data_folder_lr(sample_config['sample_name'], base_path)
    else:
        print(f"❌ Unknown dataset type: {dataset_type}")
        data['lr'] = None
    
    return data

def create_mixed_dataset_comparison(sample_configs, output_path, base_path="/raid/home/sandej17/satellite_sr"):
    """Create a comparison with mixed datasets."""
    print(f"Creating mixed dataset comparison with {len(sample_configs)} samples")
    
    n_samples = len(sample_configs)
    n_cols = 6  # LR, Bilinear, Lafanetre, NIR, Ours (MSE), HR Reference
    n_rows = n_samples
    
    # Create the figure with extra row for headers
    fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=(4*n_cols, 4*(n_rows + 1)))
    
    if n_rows == 1:
        axes = axes.reshape(2, -1)
    
    # Column headers
    column_names = ['LR Sample', 'Bilinear', 'Lafanetre', 'NIR', 'Ours (MSE)', 'HR Reference']
    
    # Add column headers in the first row
    for col, name in enumerate(column_names):
        axes[0, col].text(0.5, 0.5, name, ha='center', va='center', 
                         transform=axes[0, col].transAxes, fontsize=16, fontweight='bold')
        axes[0, col].axis('off')
    
    # Process each sample
    for row, sample_config in enumerate(sample_configs):
        print(f"\nProcessing {sample_config['dataset_type']} sample: {sample_config['sample_name']} (row {row + 1})...")
        
        # Load data for this sample
        data = load_mixed_sample_data(sample_config, base_path)
        if data is None:
            print(f"Failed to load data for sample: {sample_config}")
            continue
        
        # Display each column (use row + 1 because row 0 is for headers)
        actual_row = row + 1
        columns = ['lr', 'bilinear', 'lafanetre', 'nir', 'ours', 'hr']
        
        for col, key in enumerate(columns):
            if key in data and data[key] is not None:
                axes[actual_row, col].imshow(data[key])
            else:
                axes[actual_row, col].text(0.5, 0.5, f'{key}\n(Not Found)', ha='center', va='center', 
                                  transform=axes[actual_row, col].transAxes, fontsize=10)
            axes[actual_row, col].axis('off')
    
    # Clean layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Mixed dataset comparison saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create mixed dataset comparison with data folder samples")
    parser.add_argument("--output_path", help="Output image path")
    parser.add_argument("--base_path", default="/raid/home/sandej17/satellite_sr", help="Base data path")
    parser.add_argument("--list_samples", action="store_true", help="List available samples")
    
    args = parser.parse_args()
    
    if args.list_samples:
        synthetic_samples, data_samples = get_available_samples(args.base_path)
        print("Available SyntheticBurstVal samples:")
        for i, sample in enumerate(synthetic_samples[:10]):  # Show first 10
            print(f"  {i}: {sample}")
        print(f"  ... and {len(synthetic_samples) - 10} more" if len(synthetic_samples) > 10 else "")
        
        print("\nAvailable data folder samples:")
        for i, sample in enumerate(data_samples[:10]):  # Show first 10
            print(f"  {i}: {sample}")
        print(f"  ... and {len(data_samples) - 10} more" if len(data_samples) > 10 else "")
        return
    
    if not args.output_path:
        print("Error: --output_path is required when not using --list_samples")
        return
    
    # Example configuration with proper sample names
    synthetic_samples, data_samples = get_available_samples(args.base_path)
    
    sample_configs = [
        {
            'dataset_type': 'synthetic_burst',
            'sample_name': synthetic_samples[0] if synthetic_samples else '0006',
            'handheld_sample': synthetic_samples[0] if synthetic_samples else '0006',
            'nir_sample': 'sample_000',
            'ours_sample': 'sample_000'
        },
        {
            'dataset_type': 'data_folder',
            'sample_name': data_samples[0] if data_samples else 'Amnesty POI-7-1-3_rgb',
            'handheld_sample': synthetic_samples[0] if synthetic_samples else '0006',
            'nir_sample': 'sample_001',
            'ours_sample': 'sample_001'
        },
        {
            'dataset_type': 'synthetic_burst',
            'sample_name': synthetic_samples[1] if len(synthetic_samples) > 1 else synthetic_samples[0],
            'handheld_sample': synthetic_samples[1] if len(synthetic_samples) > 1 else synthetic_samples[0],
            'nir_sample': 'sample_001',
            'ours_sample': 'sample_001'
        },
        {
            'dataset_type': 'data_folder',
            'sample_name': data_samples[1] if len(data_samples) > 1 else data_samples[0],
            'handheld_sample': synthetic_samples[0] if synthetic_samples else '0006',
            'nir_sample': 'sample_002',
            'ours_sample': 'sample_002'
        }
    ]
    
    print("Running with mixed datasets:")
    for i, config in enumerate(sample_configs):
        print(f"  {i+1}. {config['dataset_type']}: {config['sample_name']}")
    
    create_mixed_dataset_comparison(sample_configs, args.output_path, args.base_path)

if __name__ == "__main__":
    main()
