#!/usr/bin/env python3
"""
Create comprehensive comparison images from multiple result folders.

This script takes sample IDs and their datasets, extracts ground truth and LR images
from the original datasets, and creates comparison grids showing results from different methods.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json
import argparse
from PIL import Image
import os

# Import dataset and model utilities
from data import get_dataset
from utils import bilinear_resize_torch, align_output_to_target, get_valid_mask
from handheld.evals_2 import get_gaussian_kernel, match_colors
from handheld.utils import align_kornia_brute_force
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure as ssim
import lpips

def load_image_safely(image_path):
    """Load image safely, handling different formats."""
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return None
    
    try:
        # Try loading with PIL first
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

def extract_ground_truth_and_lr(sample_id, dataset_name, args):
    """Extract ground truth and LR images from the original dataset."""
    print(f"Extracting images for sample {sample_id} from dataset {dataset_name}")
    
    # Set up args for this specific sample
    sample_args = argparse.Namespace(**vars(args))
    sample_args.sample_id = sample_id
    sample_args.dataset = dataset_name
    sample_args.scale_factor = getattr(args, 'scale_factor', 4.0)  # Add missing scale_factor
    
    # Handle dataset-specific setup
    if dataset_name == "satburst_synth":
        sample_args.root_satburst_synth = f"data/{sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"
    elif dataset_name == "burst_synth":
        if 'DATA_DIR_ABSOLUTE' in os.environ:
            sample_args.root_burst_synth = os.environ['DATA_DIR_ABSOLUTE']
        else:
            sample_args.root_burst_synth = "SyntheticBurstVal"
        try:
            sample_args.sample_id = int(sample_id)
        except ValueError:
            print(f"Warning: sample_id '{sample_id}' cannot be converted to integer for burst_synth dataset. Using 0 instead.")
            sample_args.sample_id = 0
    elif dataset_name == "worldstrat_test":
        sample_args.root_worldstrat_test = "worldstrat_test_data"
    
    try:
        # Get the dataset
        train_data = get_dataset(args=sample_args, name=dataset_name)
        
        # Extract ground truth
        hr_image = train_data.get_original_hr().cpu().numpy()
        if hr_image.ndim == 4:  # [1, H, W, C]
            hr_image = hr_image.squeeze(0)
        
        # Extract LR image
        if hasattr(train_data, 'get_lr_sample_hwc'):
            lr_original = train_data.get_lr_sample_hwc(0).cpu().numpy()  # H x W x 3
            lr_needs_unstandardize = True
        else:
            lr_any = train_data.get_lr_sample(0).cpu().numpy()  # might be CHW or HWC or multi-frame
            if lr_any.ndim == 3 and lr_any.shape[0] == 3:  # CHW -> HWC
                lr_original = np.transpose(lr_any, (1, 2, 0))
            elif lr_any.ndim == 3 and lr_any.shape[2] > 3:  # H, W, (3*T)
                H, W, C = lr_any.shape
                if C % 3 == 0:
                    T = C // 3
                    lr_original = lr_any.reshape(H, W, T, 3)[:, :, 0, :]
                else:
                    lr_original = lr_any[:, :, :3]
            else:
                lr_original = lr_any  # assume HWC
            lr_needs_unstandardize = False

        # Unstandardize LR if needed
        if lr_needs_unstandardize:
            lr_std = train_data.get_lr_std(0).cpu().numpy()
            lr_mean = train_data.get_lr_mean(0).cpu().numpy()
            if lr_std.ndim == 1:
                lr_std = lr_std.reshape(1, 1, -1)
            if lr_mean.ndim == 1:
                lr_mean = lr_mean.reshape(1, 1, -1)
            lr_original = lr_original * lr_std + lr_mean
        
        # Ensure images are in valid range
        hr_image = np.clip(hr_image, 0, 1)
        lr_original = np.clip(lr_original, 0, 1)
        
        return hr_image, lr_original
        
    except Exception as e:
        print(f"Error extracting images for sample {sample_id}: {e}")
        return None, None

def load_specific_method_results(sample_index, scale_factor, base_path="/raid/home/sandej17/satellite_sr"):
    """Load results from specific method paths for the 6-column layout using index-based matching."""
    results = {}
    base_path = Path(base_path)
    
    # Get available samples from each directory
    handheld_dir = base_path / f"handheld/results_synthetic_scale_{scale_factor}_shift_1/handheld/SyntheticBurst"
    nir_dir = base_path / "burst_synth_df4_nir"
    ours_dir = base_path / "burst_synth_df4_inr"
    
    # Get sorted lists of available samples
    handheld_samples = sorted([d.name for d in handheld_dir.iterdir() if d.is_dir()])
    nir_samples = sorted([d.name for d in nir_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    ours_samples = sorted([d.name for d in ours_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    
    print(f"Available samples:")
    print(f"  Handheld: {len(handheld_samples)} samples")
    print(f"  NIR: {len(nir_samples)} samples") 
    print(f"  Ours: {len(ours_samples)} samples")
    
    # Check if we have enough samples
    if sample_index >= len(handheld_samples) or sample_index >= len(nir_samples) or sample_index >= len(ours_samples):
        print(f"Warning: Sample index {sample_index} is out of range for available samples")
        return results
    
    # Get the sample names by index
    handheld_sample = handheld_samples[sample_index]
    nir_sample = nir_samples[sample_index]
    ours_sample = ours_samples[sample_index]
    
    print(f"Using samples by index {sample_index}:")
    print(f"  Handheld: {handheld_sample}")
    print(f"  NIR: {nir_sample}")
    print(f"  Ours: {ours_sample}")
    
    # 1. Bilinear - from handheld directory
    bilinear_path = handheld_dir / handheld_sample / "aligned_baseline.png"
    if bilinear_path.exists():
        results['Bilinear'] = {
            'type': 'individual',
            'prediction': bilinear_path,
            'sample_dir': bilinear_path.parent
        }
        print(f"✅ Found Bilinear: {bilinear_path}")
    else:
        print(f"❌ Bilinear not found: {bilinear_path}")
    
    # 2. Lafanetre - from handheld directory
    lafanetre_path = handheld_dir / handheld_sample / "aligned_output.png"
    if lafanetre_path.exists():
        results['Lafanetre'] = {
            'type': 'individual',
            'prediction': lafanetre_path,
            'sample_dir': lafanetre_path.parent
        }
        print(f"✅ Found Lafanetre: {lafanetre_path}")
    else:
        print(f"❌ Lafanetre not found: {lafanetre_path}")
    
    # 3. NIR - from burst_synth_df4_nir
    nir_path = nir_dir / nir_sample / "prediction_aligned.png"
    if nir_path.exists():
        results['NIR'] = {
            'type': 'individual',
            'prediction': nir_path,
            'sample_dir': nir_path.parent
        }
        print(f"✅ Found NIR: {nir_path}")
    else:
        print(f"❌ NIR not found: {nir_path}")
    
    # 4. Ours (MSE) - from burst_synth_df4_inr
    ours_path = ours_dir / ours_sample / "prediction_aligned.png"
    if ours_path.exists():
        results['Ours (MSE)'] = {
            'type': 'individual',
            'prediction': ours_path,
            'sample_dir': ours_path.parent
        }
        print(f"✅ Found Ours (MSE): {ours_path}")
    else:
        print(f"❌ Ours (MSE) not found: {ours_path}")
    
    # 5. HR Reference - from burst_synth_df4_inr
    hr_path = ours_dir / ours_sample / "ground_truth.png"
    if hr_path.exists():
        results['HR Reference'] = {
            'type': 'individual',
            'prediction': hr_path,
            'sample_dir': hr_path.parent
        }
        print(f"✅ Found HR Reference: {hr_path}")
    else:
        print(f"❌ HR Reference not found: {hr_path}")
    
    return results

def create_specific_comparison(sample_index, dataset_name, scale_factor, output_path, args):
    """Create a specific 6-column comparison: LR, Bilinear, Lafanetre, NIR, Ours (MSE), HR."""
    print(f"Creating specific comparison for sample index {sample_index}")
    
    # Load specific method results first to get the actual sample names
    method_results = load_specific_method_results(sample_index, scale_factor)
    if not method_results:
        print(f"No method results found for sample index {sample_index}")
        return False
    
    # For now, we'll skip the dataset extraction and just use the images from the method results
    # This avoids the complexity of matching sample IDs across different datasets
    print("Using images directly from method results (skipping dataset extraction)")
    
    # Create 6-column layout: LR, Bilinear, Lafanetre, NIR, Ours (MSE), HR
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    
    # Column 1: LR Sample (placeholder for now)
    axes[0].text(0.5, 0.5, f'LR Sample\nIndex {sample_index}', ha='center', va='center', 
                transform=axes[0].transAxes, fontsize=12, fontweight='bold')
    axes[0].set_title('LR Sample', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Column 2: Bilinear
    if 'Bilinear' in method_results:
        bilinear_img = load_image_safely(method_results['Bilinear']['prediction'])
        if bilinear_img is not None:
            axes[1].imshow(bilinear_img)
            axes[1].set_title('Bilinear', fontsize=14, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'Bilinear\n(Not Found)', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Bilinear', fontsize=14, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Bilinear\n(Not Found)', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Bilinear', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Column 3: Lafanetre
    if 'Lafanetre' in method_results:
        lafanetre_img = load_image_safely(method_results['Lafanetre']['prediction'])
        if lafanetre_img is not None:
            axes[2].imshow(lafanetre_img)
            axes[2].set_title('Lafanetre', fontsize=14, fontweight='bold')
        else:
            axes[2].text(0.5, 0.5, 'Lafanetre\n(Not Found)', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Lafanetre', fontsize=14, fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'Lafanetre\n(Not Found)', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Lafanetre', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Column 4: NIR
    if 'NIR' in method_results:
        nir_img = load_image_safely(method_results['NIR']['prediction'])
        if nir_img is not None:
            axes[3].imshow(nir_img)
            axes[3].set_title('NIR', fontsize=14, fontweight='bold')
        else:
            axes[3].text(0.5, 0.5, 'NIR\n(Not Found)', ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('NIR', fontsize=14, fontweight='bold')
    else:
        axes[3].text(0.5, 0.5, 'NIR\n(Not Found)', ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('NIR', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    # Column 5: Ours (MSE)
    if 'Ours (MSE)' in method_results:
        ours_img = load_image_safely(method_results['Ours (MSE)']['prediction'])
        if ours_img is not None:
            axes[4].imshow(ours_img)
            axes[4].set_title('Ours (MSE)', fontsize=14, fontweight='bold')
        else:
            axes[4].text(0.5, 0.5, 'Ours (MSE)\n(Not Found)', ha='center', va='center', transform=axes[4].transAxes)
            axes[4].set_title('Ours (MSE)', fontsize=14, fontweight='bold')
    else:
        axes[4].text(0.5, 0.5, 'Ours (MSE)\n(Not Found)', ha='center', va='center', transform=axes[4].transAxes)
        axes[4].set_title('Ours (MSE)', fontsize=14, fontweight='bold')
    axes[4].axis('off')
    
    # Column 6: HR Reference
    if 'HR Reference' in method_results:
        hr_img = load_image_safely(method_results['HR Reference']['prediction'])
        if hr_img is not None:
            axes[5].imshow(hr_img)
            axes[5].set_title('HR Reference', fontsize=14, fontweight='bold')
        else:
            axes[5].text(0.5, 0.5, 'HR Reference\n(Not Found)', ha='center', va='center', transform=axes[5].transAxes)
            axes[5].set_title('HR Reference', fontsize=14, fontweight='bold')
    else:
        axes[5].text(0.5, 0.5, 'HR Reference\n(Not Found)', ha='center', va='center', transform=axes[5].transAxes)
        axes[5].set_title('HR Reference', fontsize=14, fontweight='bold')
    axes[5].axis('off')
    
    plt.suptitle(f'Comprehensive Comparison - Sample Index: {sample_index}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Comprehensive comparison saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create specific 6-column comparison images using index-based matching")
    
    # Sample specification
    parser.add_argument("--sample_indices", nargs='+', type=int, required=True, 
                       help="List of sample indices to process (0, 1, 2, etc.)")
    parser.add_argument("--datasets", nargs='+', required=True,
                       help="List of datasets corresponding to each sample")
    
    # Dataset parameters
    parser.add_argument("--df", type=int, default=4, help="Downsampling factor")
    parser.add_argument("--scale_factor", type=int, default=4, help="Scale factor for handheld results")
    parser.add_argument("--lr_shift", type=float, default=1.0)
    parser.add_argument("--aug", type=str, default="none", choices=['none', 'light', 'medium', 'heavy'])
    
    # Output
    parser.add_argument("--output_dir", type=str, default="index_comparisons",
                       help="Output directory for comparison images")
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.sample_indices) != len(args.datasets):
        print("Error: Number of sample indices must match number of datasets")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Process each sample
    for sample_index, dataset_name in zip(args.sample_indices, args.datasets):
        print(f"\n{'='*60}")
        print(f"Processing sample index: {sample_index} from dataset: {dataset_name}")
        print(f"{'='*60}")
        
        output_path = output_dir / f"index_comparison_{sample_index}_{dataset_name}.png"
        
        success = create_specific_comparison(
            sample_index, dataset_name, args.scale_factor, output_path, args
        )
        
        if success:
            print(f"✅ Successfully created comparison for index {sample_index}")
        else:
            print(f"❌ Failed to create comparison for index {sample_index}")
    
    print(f"\nAll comparisons saved to: {output_dir}")

if __name__ == "__main__":
    main()
