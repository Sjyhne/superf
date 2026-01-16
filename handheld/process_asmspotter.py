#!/usr/bin/env python3
"""
Process ASMSpotter images through the handheld super-resolution pipeline.
Supports RGB images with arbitrary scale factors.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from handheld_super_resolution import process
import argparse

def load_images(data_path, num_samples=8, grayscale=False):
    """
    Load high-resolution ground truth and sample images from the specified path.
    
    Args:
        data_path: Path to the data directory containing images
        num_samples: Maximum number of sample images to load
        grayscale: Whether to convert images to grayscale or keep RGB
        
    Returns:
        hr_ground_truth: High-resolution ground truth image or None if not found
        burst: Numpy array of sample images with shape [B, H, W] or [B, H, W, 3]
    """
    data_path = Path(data_path)
    
    # Load HR ground truth image if it exists
    hr_ground_truth = None
    hr_path = data_path / 'hr_ground_truth.png'
    if hr_path.exists():
        print(f"Loading ground truth: {hr_path.name}")
        hr_img = cv2.imread(str(hr_path))
        if hr_img is not None:
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            if grayscale and len(hr_img.shape) == 3:
                hr_img = np.dot(hr_img, [0.2989, 0.5870, 0.1140])
            hr_ground_truth = hr_img.astype(np.float32) / 255.0
            print(f"Loaded HR ground truth with shape {hr_ground_truth.shape}")
    
    # Find all sample images
    sample_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
        sample_files.extend(sorted(data_path.glob(f'sample_*{ext}')))
    
    if not sample_files:
        raise ValueError(f"No sample images found in {data_path}")
    
    print(f"Found {len(sample_files)} sample images")
    
    # Load and convert sample images
    samples = []
    for i, img_path in enumerate(sample_files):
        if i >= num_samples:
            break
            
        print(f"Loading {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load image {img_path}, skipping")
            continue
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale if requested
        if grayscale and len(img.shape) == 3:
            img = np.dot(img, [0.2989, 0.5870, 0.1140])
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        samples.append(img)
    
    if not samples:
        raise ValueError("No valid sample images could be loaded")
    
    # Stack into burst array [B, H, W] or [B, H, W, 3]
    burst = np.stack(samples, axis=0)
    print(f"Loaded {len(samples)} sample images with shape {burst.shape}")
    
    return hr_ground_truth, burst

def process_rgb(burst, exposures, options, params):
    """
    Process an RGB burst by separately processing each color channel.
    
    Args:
        burst: RGB burst with shape [B, H, W, 3]
        exposures: Exposure values with shape [B]
        options: Processing options dictionary
        params: Processing parameters dictionary
        
    Returns:
        RGB output image with shape [H, W, 3]
    """
    print("Processing RGB image by channels...")
    r_channel = burst[:, :, :, 0]
    g_channel = burst[:, :, :, 1]
    b_channel = burst[:, :, :, 2]
    
    # Process each channel separately
    r_output = process(r_channel, exposures, options, params)
    g_output = process(g_channel, exposures, options, params)
    b_output = process(b_channel, exposures, options, params)
    
    # Combine channels back into RGB
    output_rgb = np.stack([r_output, g_output, b_output], axis=2)
    return output_rgb

def crop_borders(img, margin=16):
    """
    Crop border pixels from an image to avoid edge artifacts in evaluation.
    
    Args:
        img: Input image array
        margin: Number of pixels to crop from each edge
        
    Returns:
        Cropped image array
    """
    if len(img.shape) == 3:  # RGB
        return img[margin:-margin, margin:-margin, :]
    else:  # Grayscale
        return img[margin:-margin, margin:-margin]

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process images through the handheld super-resolution pipeline")
    parser.add_argument("--data-path", type=str, default="../data/Landcover-1295513_rgb/scale_2_shift_1.0px_aug_none",
                        help="Path to directory containing sample images")
    parser.add_argument("--output-dir", type=str, default="results/Landcover-1295513_rgb",
                        help="Directory to save output images")
    parser.add_argument("--scale", type=int, default=2,
                        help="Scale factor for super-resolution")
    parser.add_argument("--grayscale", action="store_true",
                        help="Process images as grayscale")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Maximum number of sample images to use")
    parser.add_argument("--crop-margin", type=int, default=16,
                        help="Number of pixels to crop from edges for evaluation")
    args = parser.parse_args()
    
    # Set paths
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images
    hr_ground_truth, burst = load_images(data_path, num_samples=args.num_samples, grayscale=args.grayscale)
    
    # Save burst as npy file
    np_path = output_dir / 'burst.npy'
    np.save(np_path, burst)
    print(f"Saved burst to {np_path}")
    
    # Create exposure arrays with proper format
    num_frames = burst.shape[0]
    exposures = np.ones(num_frames, dtype=np.float32)  # Exposures for all frames
    
    # Process burst using parameters
    options = {'verbose': 1}
    params = {
        'scale': args.scale,
        'base detail': True,
        'alignment': 'Fnet',
    }
    
    print(f"\nProcessing burst with {args.scale}x super-resolution...")
    
    is_rgb = len(burst.shape) == 4 and burst.shape[3] == 3
    
    if is_rgb and not args.grayscale:
        output_img = process_rgb(burst, exposures, options, params)
    else:
        output_img = process(burst, exposures, options, params)
    
    # Create bilinear interpolation baseline if HR ground truth is available
    bilinear_img = None
    if hr_ground_truth is not None:
        # Get target dimensions from HR ground truth
        target_h, target_w = hr_ground_truth.shape[:2]
        
        # Apply bilinear interpolation to the first LR sample
        if is_rgb and not args.grayscale:
            bilinear_img = cv2.resize(burst[0], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            bilinear_img = cv2.resize(burst[0], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate PSNR for bilinear baseline with cropping
        if bilinear_img.shape == hr_ground_truth.shape:
            # Crop borders for evaluation to avoid edge artifacts
            hr_crop = crop_borders(hr_ground_truth, args.crop_margin)
            bilinear_crop = crop_borders(bilinear_img, args.crop_margin)
            
            bilinear_mse = np.mean((hr_crop - bilinear_crop) ** 2)
            bilinear_psnr = 10 * np.log10(1.0 / bilinear_mse)
            print(f"Bilinear PSNR (cropped): {bilinear_psnr:.2f} dB")
    
    # Save results
    output_path = output_dir / f'output_scale{args.scale}x.png'
    ref_path = output_dir / 'reference.png'
    
    # Save bilinear interpolation result if available
    if bilinear_img is not None:
        bilinear_path = output_dir / f'bilinear_scale{args.scale}x.png'
        if is_rgb and not args.grayscale:
            cv2.imwrite(str(bilinear_path), cv2.cvtColor((bilinear_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(bilinear_path), (bilinear_img * 255).astype(np.uint8))
        print(f"Saved bilinear baseline to {bilinear_path}")
    
    # Convert to uint8 for saving
    if len(output_img.shape) == 3 and output_img.shape[2] == 3:  # RGB
        cv2.imwrite(str(output_path), cv2.cvtColor((output_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    else:  # Grayscale
        cv2.imwrite(str(output_path), (output_img * 255).astype(np.uint8))
        
    # Save reference image (first sample image)
    if is_rgb and not args.grayscale:
        cv2.imwrite(str(ref_path), cv2.cvtColor((burst[0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(ref_path), (burst[0] * 255).astype(np.uint8))
        
    print(f"Saved output to {output_path}")
    print(f"Saved reference to {ref_path}")
    
    # Create comparison visualization including ground truth if available
    if hr_ground_truth is not None:
        # Generate two comparisons: one with full images and one with cropped regions
        
        # 1. Full image comparison (original)
        plt.figure(figsize=(20, 5))
        
        plt.subplot(151)
        if is_rgb and not args.grayscale:
            plt.imshow(hr_ground_truth)
        else:
            plt.imshow(hr_ground_truth, cmap='gray', vmin=0, vmax=1)
        plt.title('HR Ground Truth (Full)')
        plt.axis('off')
        
        plt.subplot(152)
        if is_rgb and not args.grayscale:
            plt.imshow(burst[0])
        else:
            plt.imshow(burst[0], cmap='gray', vmin=0, vmax=1)
        plt.title('LR Reference Frame')
        plt.axis('off')
        
        plt.subplot(153)
        if bilinear_img is not None:
            if is_rgb and not args.grayscale:
                plt.imshow(bilinear_img)
            else:
                plt.imshow(bilinear_img, cmap='gray', vmin=0, vmax=1)
            plt.title('Bilinear Interpolation (Full)')
        else:
            plt.text(0.5, 0.5, "No bilinear baseline", ha='center', va='center')
            plt.title('Bilinear Interpolation')
        plt.axis('off')
        
        plt.subplot(154)
        if is_rgb and not args.grayscale:
            plt.imshow(output_img)
        else:
            plt.imshow(output_img, cmap='gray', vmin=0, vmax=1)
        plt.title('Super-resolved Output (Full)')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'comparison_full_scale{args.scale}x.png')
        plt.close()
        
        # 2. Cropped region comparison (for PSNR evaluation)
        if hr_ground_truth.shape == output_img.shape:
            # First ensure we have the cropped versions
            hr_crop = crop_borders(hr_ground_truth, args.crop_margin)
            output_crop = crop_borders(output_img, args.crop_margin)
            bilinear_crop = None
            if bilinear_img is not None:
                bilinear_crop = crop_borders(bilinear_img, args.crop_margin)
                
                # Calculate metrics on cropped regions
                bilinear_mse = np.mean((hr_crop - bilinear_crop) ** 2)
                bilinear_psnr = 10 * np.log10(1.0 / bilinear_mse)
                
            mse = np.mean((hr_crop - output_crop) ** 2)
            psnr = 10 * np.log10(1.0 / mse)
            
            # Create visualization
            plt.figure(figsize=(20, 5))
            
            plt.subplot(151)
            if is_rgb and not args.grayscale:
                plt.imshow(hr_crop)
            else:
                plt.imshow(hr_crop, cmap='gray', vmin=0, vmax=1)
            plt.title('HR Ground Truth\n(Evaluation Region)')
            plt.axis('off')
            
            # Add LR image with proportionally scaled crop
            plt.subplot(152)
            # Scale the crop margin proportionally to the resolution difference
            lr_margin = max(1, int(args.crop_margin / args.scale))
            lr_crop = crop_borders(burst[0], lr_margin)
            if is_rgb and not args.grayscale:
                plt.imshow(lr_crop)
            else:
                plt.imshow(lr_crop, cmap='gray', vmin=0, vmax=1)
            plt.title(f'LR Reference Frame\n(Scaled {args.scale}x for display)')
            plt.axis('off')
            
            print(f"Using {args.crop_margin} pixel margin for HR/SR images and {lr_margin} pixel margin for LR image (proportional to scale factor)")
            
            plt.subplot(153)
            if bilinear_crop is not None:
                if is_rgb and not args.grayscale:
                    plt.imshow(bilinear_crop)
                else:
                    plt.imshow(bilinear_crop, cmap='gray', vmin=0, vmax=1)
                plt.title(f'Bilinear Interpolation\nPSNR: {bilinear_psnr:.2f} dB')
            plt.axis('off')
            
            plt.subplot(154)
            if is_rgb and not args.grayscale:
                plt.imshow(output_crop)
            else:
                plt.imshow(output_crop, cmap='gray', vmin=0, vmax=1)
            plt.title(f'Super-resolved Output\nPSNR: {psnr:.2f} dB')
            plt.axis('off')
            
            # Show PSNR improvement
            if bilinear_crop is not None:
                plt.subplot(155)
                psnr_improvement = psnr - bilinear_psnr
                plt.text(0.5, 0.5, f"PSNR Improvement:\n{psnr_improvement:.2f} dB", 
                       fontsize=14, ha='center', va='center')
                plt.axis('off')
                print(f"PSNR Improvement over bilinear: {psnr_improvement:.2f} dB")
            
            plt.tight_layout()
            plt.savefig(output_dir / f'comparison_scale{args.scale}x.png')
            plt.close()
            
            # Save cropped regions used for evaluation
            crop_dir = output_dir / 'cropped_evaluation'
            crop_dir.mkdir(exist_ok=True)
            
            # Save LR crop
            if is_rgb and not args.grayscale:
                cv2.imwrite(str(crop_dir / 'lr_cropped.png'), 
                          cv2.cvtColor((lr_crop * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(crop_dir / 'lr_cropped.png'), (lr_crop * 255).astype(np.uint8))
            
            if is_rgb and not args.grayscale:
                cv2.imwrite(str(crop_dir / 'hr_cropped.png'), 
                          cv2.cvtColor((hr_crop * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                if bilinear_crop is not None:
                    cv2.imwrite(str(crop_dir / 'bilinear_cropped.png'), 
                              cv2.cvtColor((bilinear_crop * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(crop_dir / 'sr_cropped.png'), 
                          cv2.cvtColor((output_crop * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(crop_dir / 'hr_cropped.png'), (hr_crop * 255).astype(np.uint8))
                if bilinear_crop is not None:
                    cv2.imwrite(str(crop_dir / 'bilinear_cropped.png'), (bilinear_crop * 255).astype(np.uint8))
                cv2.imwrite(str(crop_dir / 'sr_cropped.png'), (output_crop * 255).astype(np.uint8))
            
            print(f"Saved cropped evaluation regions to {crop_dir}")
    else:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        if is_rgb and not args.grayscale:
            plt.imshow(burst[0])
        else:
            plt.imshow(burst[0], cmap='gray', vmin=0, vmax=1)
        plt.title('LR Reference Frame')
        plt.axis('off')
        
        plt.subplot(132)
        if is_rgb and not args.grayscale:
            plt.imshow(output_img)
        else:
            plt.imshow(output_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Super-resolved Output ({args.scale}x)')
        plt.axis('off')
    
        plt.tight_layout()
        plt.savefig(output_dir / f'comparison_scale{args.scale}x.png')
        plt.close()
    
    print(f"Processing complete. Output size: {output_img.shape}")

if __name__ == "__main__":
    main() 