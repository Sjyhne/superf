#!/usr/bin/env python3
"""
Create combined comparison with multiple samples in rows and columns.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import argparse

def load_image_safely(image_path):
    """Load image safely, handling different formats."""
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return None
    
    try:
        img = Image.open(image_path)
        img = np.array(img)
        
        # Convert to float and normalize if needed
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:  # RGBA
            img = img[:, :, :3]
        
        return np.clip(img, 0, 1)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_actual_lr_sample(sample_index, base_path, scale_factor):
    """Load actual LR sample from SyntheticBurstVal dataset."""
    base_path = Path(base_path)
    
    # Map sample index to SyntheticBurstVal sample ID
    # The handheld samples are 4-digit like 0006, 0013, etc.
    handheld_dir = base_path / f"handheld/results_synthetic_scale_{scale_factor}_shift_1/handheld/SyntheticBurst"
    handheld_samples = sorted([d.name for d in handheld_dir.iterdir() if d.is_dir()])
    
    if sample_index >= len(handheld_samples):
        print(f"Sample index {sample_index} is out of range for handheld samples")
        return None
    
    handheld_sample = handheld_samples[sample_index]
    print(f"Loading LR from SyntheticBurstVal sample: {handheld_sample}")
    
    # Load the first frame (im_raw_00.png) from the burst
    burst_path = base_path / "SyntheticBurstVal" / "bursts" / handheld_sample / "im_raw_00.png"
    print(f"  📁 LR Burst path: {burst_path}")
    
    if not burst_path.exists():
        print(f"❌ LR burst image not found: {burst_path}")
        return None
    
    try:
        # Load and process the raw burst image like in data.py
        import cv2
        im = cv2.imread(str(burst_path), cv2.IMREAD_UNCHANGED)
        # Convert from 16-bit to float and normalize
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
        
        # Apply white balance (example values, actual values might differ)
        wb_gains = np.array([2.0, 1.0, 1.5])  # R, G, B gains
        rgb = rgb * wb_gains
        
        # Apply gamma correction
        gamma = 2.2
        rgb = np.power(np.maximum(rgb, 0), 1.0/gamma)
        
        # Clip values to [0, 1]
        rgb = np.clip(rgb, 0, 1)
        
        print(f"✅ LR loaded from burst: {rgb.shape}")
        return rgb
        
    except Exception as e:
        print(f"❌ Error loading LR burst image: {e}")
        return None

def load_burstsr_lr_sample(sample_index, base_path):
    """Load actual LR sample from BurstSR dataset."""
    base_path = Path(base_path)
    burstsr_path = base_path / "BurstSR" / "BurstSR"
    
    # Get all train directories
    train_dirs = sorted([d for d in burstsr_path.iterdir() if d.is_dir() and d.name.startswith("train_")])
    
    if not train_dirs:
        print("❌ No BurstSR train directories found")
        return None
    
    # Get all sample directories from all train directories
    all_samples = []
    for train_dir in train_dirs:
        samples = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        all_samples.extend(samples)
    
    if sample_index >= len(all_samples):
        print(f"Sample index {sample_index} is out of range for BurstSR samples (total: {len(all_samples)})")
        return None
    
    sample_dir = all_samples[sample_index]
    print(f"Loading LR from BurstSR sample: {sample_dir.name}")
    
    # Try to load from canon directory first, then samsung_00
    canon_path = sample_dir / "canon" / "im_raw.png"
    samsung_path = sample_dir / "samsung_00" / "im_raw.png"
    print(f"  📁 Canon path: {canon_path}")
    print(f"  📁 Samsung path: {samsung_path}")
    
    lr_path = None
    if canon_path.exists():
        lr_path = canon_path
    elif samsung_path.exists():
        lr_path = samsung_path
    else:
        print(f"❌ No LR image found in {sample_dir}")
        return None
    
    try:
        # Load the raw image
        import cv2
        im = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED)
        
        if im is None:
            print(f"❌ Failed to load image: {lr_path}")
            return None
        
        # Convert to float and normalize
        if im.dtype == np.uint8:
            im = im.astype(np.float32) / 255.0
        elif im.dtype == np.uint16:
            im = im.astype(np.float32) / 65535.0
        else:
            im = im.astype(np.float32)
        
        # Ensure 3 channels
        if len(im.shape) == 2:
            im = np.stack([im] * 3, axis=-1)
        elif im.shape[-1] == 4:  # RGBA
            im = im[:, :, :3]
        
        # Clip values to [0, 1]
        im = np.clip(im, 0, 1)
        
        print(f"✅ LR loaded from BurstSR: {im.shape}")
        return im
        
    except Exception as e:
        print(f"❌ Error loading BurstSR LR image: {e}")
        return None

def load_satburst_lr_sample(sample_index, base_path, sample_name, scale_factor):
    """Load actual LR sample from satburst_synth dataset (using dataset class)."""
    base_path = Path(base_path)
    
    try:
        import sys
        sys.path.append(str(base_path))
        from data import SRData
        
        # Map sample names to actual sample directories
        # The sample_name is like "sample_000", we need to find which actual sample it corresponds to
        sample_dirs = sorted(list((base_path / "data").glob("*_rgb")))
        
        if not sample_dirs:
            print("❌ No satburst_synth sample directories found")
            return None
        
        # Map sample_index to the corresponding sample directory
        if sample_index >= len(sample_dirs):
            print(f"❌ Sample index {sample_index} is out of range for available samples ({len(sample_dirs)})")
            return None
        
        sample_dir = sample_dirs[sample_index] / f"scale_{scale_factor}_shift_1.0px_aug_none"
        
        if not sample_dir.exists():
            print(f"❌ Sample directory not found: {sample_dir}")
            return None
        
        print(f"Loading LR from satburst_synth sample: {sample_dirs[sample_index].name}")
        print(f"  📁 Sample directory: {sample_dir}")
        
        # Create a minimal args object for SRData
        class Args:
            def __init__(self):
                self.root_satburst_synth = str(sample_dir)
                self.num_samples = 16
                self.scale_factor = 4
        
        args = Args()
        dataset = SRData(data_dir=args.root_satburst_synth, num_samples=args.num_samples, keep_in_memory=True, scale_factor=args.scale_factor)
        
        # Get the LR sample (already unstandardized by the dataset class)
        lr_sample = dataset.get_lr_sample(sample_index)  # Returns HWC format, already unstandardized
        
        # Convert to numpy (already in HWC format)
        lr_sample = lr_sample.cpu().numpy()
        
        print(f"✅ LR loaded from satburst_synth: {lr_sample.shape}")
        return lr_sample
        
    except Exception as e:
        print(f"❌ Error loading satburst_synth LR sample: {e}")
        return None

def get_lr_image_from_hr(hr_path, scale_factor=4):
    """Generate LR image by downsampling the HR image."""
    hr_img = load_image_safely(hr_path)
    if hr_img is None:
        print(f"Failed to load HR image from {hr_path}")
        return None
    
    # Downsample by the scale factor using proper resizing
    h, w = hr_img.shape[:2]
    new_h, new_w = h // scale_factor, w // scale_factor
    
    print(f"Downsampling HR {hr_img.shape} to LR {(new_h, new_w)}")
    
    # Use proper downsampling instead of simple slicing
    from PIL import Image
    hr_pil = Image.fromarray((hr_img * 255).astype(np.uint8))
    lr_pil = hr_pil.resize((new_w, new_h), Image.LANCZOS)
    lr_img = np.array(lr_pil).astype(np.float32) / 255.0
    
    print(f"Generated LR image shape: {lr_img.shape}")
    return lr_img

def load_sample_data(sample_index, scale_factor, base_path="/raid/home/sandej17/satellite_sr", dataset_type="burst_synth"):
    """Load all data for a single sample."""
    base_path = Path(base_path)
    
    if dataset_type == "burst_synth":
        # Original burst_synth logic
        handheld_dir = base_path / f"handheld/results_synthetic_scale_{scale_factor}_shift_1/handheld/SyntheticBurst"
        nir_dir = base_path / f"burst_df{scale_factor}_nir_5k"
        ours_dir = base_path / f"burst_df{scale_factor}_inr"
        
        # Get sorted lists of available samples
        handheld_samples = sorted([d.name for d in handheld_dir.iterdir() if d.is_dir()])
        nir_samples = sorted([d.name for d in nir_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
        ours_samples = sorted([d.name for d in ours_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
        
        # Check if we have enough samples
        if sample_index >= len(handheld_samples) or sample_index >= len(nir_samples) or sample_index >= len(ours_samples):
            print(f"Warning: Sample index {sample_index} is out of range")
            return None
        
        # Get the sample names by index
        handheld_sample = handheld_samples[sample_index]
        nir_sample = nir_samples[sample_index]
        ours_sample = ours_samples[sample_index]
        
        print(f"Loading sample index {sample_index}:")
        print(f"  Handheld: {handheld_sample}")
        print(f"  NIR: {nir_sample}")
        print(f"  Ours: {ours_sample}")
        
        # Load all images
        data = {}
        
        # 1. Bilinear
        bilinear_path = handheld_dir / handheld_sample / "aligned_baseline.png"
        print(f"  📁 Bilinear path: {bilinear_path}")
        data['bilinear'] = load_image_safely(bilinear_path)
        
        # 2. Lafanetre
        lafanetre_path = handheld_dir / handheld_sample / "aligned_output.png"
        print(f"  📁 Lafanetre path: {lafanetre_path}")
        data['lafanetre'] = load_image_safely(lafanetre_path)
        
        # 3. NIR
        nir_path = nir_dir / nir_sample / "prediction_aligned.png"
        print(f"  📁 NIR path: {nir_path}")
        data['nir'] = load_image_safely(nir_path)
        
        # 4. Ours (MSE)
        ours_path = ours_dir / ours_sample / "prediction_aligned.png"
        print(f"  📁 Ours path: {ours_path}")
        data['ours'] = load_image_safely(ours_path)
        
        # 5. HR Reference
        hr_path = ours_dir / ours_sample / "ground_truth.png"
        print(f"  📁 HR Reference path: {hr_path}")
        data['hr'] = load_image_safely(hr_path)
        
        # 6. LR Sample - load actual LR from SyntheticBurstVal dataset
        print(f"Loading actual LR sample from SyntheticBurstVal dataset...")
        data['lr'] = load_actual_lr_sample(sample_index, base_path, scale_factor)
        if data['lr'] is not None:
            print(f"✅ LR loaded: {data['lr'].shape}")
        else:
            print("❌ Failed to load LR")
    
    elif dataset_type == "satburst_synth":
        # satburst_synth logic
        handheld_dir = base_path / f"handheld/results_burstsr_scale_{scale_factor}_shift_1/handheld/BurstSR"
        nir_dir = base_path / f"satburst_df{scale_factor}_nir_5k_cs"
        ours_dir = base_path / f"satburst_df{scale_factor}_inr"
        
        # Get sorted lists of available samples
        handheld_samples = sorted([d.name for d in handheld_dir.iterdir() if d.is_dir()])
        nir_samples = sorted([d.name for d in nir_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
        ours_samples = sorted([d.name for d in ours_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
        
        # Check if we have enough samples
        if sample_index >= len(handheld_samples) or sample_index >= len(nir_samples) or sample_index >= len(ours_samples):
            print(f"Warning: Sample index {sample_index} is out of range")
            return None
        
        # Get the sample names by index
        handheld_sample = handheld_samples[sample_index]
        nir_sample = nir_samples[sample_index]
        ours_sample = ours_samples[sample_index]
        
        print(f"Loading satburst_synth sample index {sample_index}:")
        print(f"  Handheld: {handheld_sample}")
        print(f"  NIR: {nir_sample}")
        print(f"  Ours: {ours_sample}")
        
        # Load all images
        data = {}
        
        # 1. Bilinear - from handheld BurstSR results
        bilinear_path = handheld_dir / handheld_sample / "aligned_baseline.png"
        print(f"  📁 Bilinear path: {bilinear_path}")
        data['bilinear'] = load_image_safely(bilinear_path)
        
        # 2. Lafanetre - from handheld BurstSR results
        lafanetre_path = handheld_dir / handheld_sample / "aligned_output.png"
        print(f"  📁 Lafanetre path: {lafanetre_path}")
        data['lafanetre'] = load_image_safely(lafanetre_path)
        
        # 3. NIR
        nir_path = nir_dir / nir_sample / "prediction_aligned.png"
        print(f"  📁 NIR path: {nir_path}")
        data['nir'] = load_image_safely(nir_path)
        
        # 4. Ours (MSE)
        ours_path = ours_dir / ours_sample / "prediction_aligned.png"
        print(f"  📁 Ours path: {ours_path}")
        data['ours'] = load_image_safely(ours_path)
        
        # 5. HR Reference
        hr_path = ours_dir / ours_sample / "ground_truth.png"
        print(f"  📁 HR Reference path: {hr_path}")
        data['hr'] = load_image_safely(hr_path)
        
        # 6. LR Sample - load from satburst_synth dataset (unstandardized)
        print(f"Loading LR sample from satburst_synth dataset...")
        data['lr'] = load_satburst_lr_sample(sample_index, base_path, ours_sample, scale_factor)
        if data['lr'] is not None:
            print(f"✅ LR loaded: {data['lr'].shape}")
        else:
            print("❌ Failed to load LR")
    
    else:
        print(f"Unsupported dataset type: {dataset_type}")
        return None
    
    return data

def create_combined_comparison(sample_configs, scale_factor, output_path, base_path="/raid/home/sandej17/satellite_sr"):
    """Create a combined comparison with multiple samples from different datasets in rows and columns.
    
    Args:
        sample_configs: List of dicts with 'dataset_type' and 'sample_indices' keys
        scale_factor: Scale factor for results
        output_path: Output path for the image
        base_path: Base path for data
    """
    print(f"Creating combined comparison for sample configs: {sample_configs}")
    
    # Calculate total number of samples across all datasets
    total_samples = sum(len(config['sample_indices']) for config in sample_configs)
    n_cols = 6  # LR, Bilinear, Lafanetre, NIR, Ours (MSE), HR Reference
    n_rows = total_samples
    
    # Create the figure with better spacing - no separate header row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Column headers
    column_names = ['LR Sample', 'Bilinear', 'Lafenetre', 'NIR', 'Ours (MSE)', 'HR Reference']
    
    # Add column headers directly above the first row of images
    for col, name in enumerate(column_names):
        # Position text above the first row
        axes[0, col].text(0.5, 1.05, name, ha='center', va='bottom', 
                         transform=axes[0, col].transAxes, fontsize=20, fontweight='bold')
    
    # Process each dataset and its samples
    current_row = 0
    for config in sample_configs:
        dataset_type = config['dataset_type']
        sample_indices = config['sample_indices']
        
        print(f"\nProcessing {dataset_type} dataset with samples: {sample_indices}")
        
        for sample_index in sample_indices:
            print(f"\nProcessing sample index {sample_index} (row {current_row + 1})...")
            
            # Load data for this sample
            data = load_sample_data(sample_index, scale_factor, base_path, dataset_type)
            if data is None:
                print(f"Failed to load data for sample index {sample_index}")
                # Add empty row
                for col in range(n_cols):
                    axes[current_row, col].text(0.5, 0.5, f'Failed to load\n{dataset_type}', ha='center', va='center', 
                                              transform=axes[current_row, col].transAxes, fontsize=10)
                    axes[current_row, col].axis('off')
                current_row += 1
                continue
            
            # Display each column
            columns = ['lr', 'bilinear', 'lafanetre', 'nir', 'ours', 'hr']
            
            for col, key in enumerate(columns):
                if key in data and data[key] is not None:
                    axes[current_row, col].imshow(data[key])
                else:
                    axes[current_row, col].text(0.5, 0.5, f'{key}\n(Not Found)', ha='center', va='center', 
                                              transform=axes[current_row, col].transAxes, fontsize=10)
                axes[current_row, col].axis('off')
            
            current_row += 1
    
    # Remove the title completely
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.02, wspace=0.1)  # Minimal vertical spacing, especially for headers
    
    # Save as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1)
    
    plt.close()
    
    print(f"Combined comparison saved to: {output_path}")
    print(f"PDF version saved to: {pdf_path}")

def create_combined_comparison_single_dataset(sample_indices, scale_factor, output_path, base_path="/raid/home/sandej17/satellite_sr", dataset_type="burst_synth"):
    """Create a combined comparison with multiple samples from a single dataset (backward compatibility)."""
    sample_configs = [{'dataset_type': dataset_type, 'sample_indices': sample_indices}]
    create_combined_comparison(sample_configs, scale_factor, output_path, base_path)

def main():
    parser = argparse.ArgumentParser(description="Create combined comparison with multiple samples from different datasets")
    
    # Sample specification - support both single dataset and multi-dataset formats
    parser.add_argument("--sample_indices", nargs='+', type=int, 
                       help="List of sample indices to include (e.g., 0 1 2 3 4) - for single dataset mode")
    parser.add_argument("--dataset_type", type=str, default="burst_synth",
                       choices=["burst_synth", "satburst_synth"],
                       help="Dataset type to use for loading results - for single dataset mode")
    
    # Multi-dataset specification
    parser.add_argument("--burst_synth_samples", nargs='+', type=int, 
                       help="Sample indices for burst_synth dataset (e.g., 0 1 2)")
    parser.add_argument("--satburst_synth_samples", nargs='+', type=int, 
                       help="Sample indices for satburst_synth dataset (e.g., 0 1 2)")
    
    parser.add_argument("--scale_factor", type=int, default=4,
                       help="Scale factor for handheld results and LR generation")
    parser.add_argument("--output_path", type=str, default="combined_comparison.png",
                       help="Output path for the combined comparison image")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine which mode to use
    if args.burst_synth_samples is not None or args.satburst_synth_samples is not None:
        # Multi-dataset mode
        sample_configs = []
        if args.burst_synth_samples is not None:
            sample_configs.append({'dataset_type': 'burst_synth', 'sample_indices': args.burst_synth_samples})
        if args.satburst_synth_samples is not None:
            sample_configs.append({'dataset_type': 'satburst_synth', 'sample_indices': args.satburst_synth_samples})
        
        if not sample_configs:
            print("❌ Error: No samples specified for any dataset")
            return
        
        # Create the combined comparison
        create_combined_comparison(sample_configs, args.scale_factor, output_path)
        
        total_samples = sum(len(config['sample_indices']) for config in sample_configs)
        print(f"\n🎉 Combined comparison created successfully!")
        print(f"📁 Output: {output_path}")
        print(f"📊 Layout: {total_samples} rows × 6 columns")
        for config in sample_configs:
            print(f"🔍 {config['dataset_type']}: {config['sample_indices']}")
    
    elif args.sample_indices is not None:
        # Single dataset mode (backward compatibility)
        create_combined_comparison_single_dataset(
            args.sample_indices, 
            args.scale_factor, 
            output_path,
            dataset_type=args.dataset_type
        )
        
        print(f"\n🎉 Combined comparison created successfully!")
        print(f"📁 Output: {output_path}")
        print(f"📊 Layout: {len(args.sample_indices)} rows × 6 columns")
        print(f"🔍 Samples: {args.sample_indices}")
        print(f"📊 Dataset: {args.dataset_type}")
    
    else:
        print("❌ Error: No samples specified. Use --sample_indices for single dataset or --burst_synth_samples/--satburst_synth_samples for multi-dataset")

if __name__ == "__main__":
    main()
