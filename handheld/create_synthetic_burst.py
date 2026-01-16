#!/usr/bin/env python3
"""
Create a synthetic burst dataset from a high-resolution (HR) image.

This script generates a synthetic burst with random shifts, adds realistic noise profiles,
and outputs the data in the format required by the handheld super-resolution pipeline.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from scipy import ndimage

def load_hr_image(hr_image_path, grayscale=False):
    """
    Load a high-resolution image from the specified path.
    
    Args:
        hr_image_path: Path to the HR image file
        grayscale: Whether to convert image to grayscale or keep RGB
        
    Returns:
        hr_image: High-resolution image as numpy array
    """
    hr_path = Path(hr_image_path)
    
    if not hr_path.exists():
        raise ValueError(f"Input image not found: {hr_path}")
    
    print(f"Loading HR image: {hr_path.name}")
    hr_img = cv2.imread(str(hr_path))
    
    if hr_img is None:
        raise ValueError(f"Could not load image: {hr_path}")
        
    # Convert BGR to RGB
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale if requested
    if grayscale and len(hr_img.shape) == 3:
        hr_img = np.dot(hr_img, [0.2989, 0.5870, 0.1140])
    
    # Normalize to [0, 1]
    hr_img = hr_img.astype(np.float32) / 255.0
    
    print(f"Loaded HR image with shape {hr_img.shape}")
    return hr_img

def pick_warps(n_warps, R_max=2):
    """
    Generate random warps/shifts for creating a burst.
    
    Args:
        n_warps: Number of warps to generate
        R_max: Maximum radius of shift
        
    Returns:
        warps: Array of random shifts with shape [n_warps, 2]
    """
    warps = R_max * (np.random.random((n_warps, 2)) - 0.5)  # between -R_max and R_max
    while True:
        norms = np.abs(warps[:, 0]) + np.abs(warps[:, 1])
        invalid = warps[norms > R_max]
        n_invalid = invalid.shape[0]
        if n_invalid == 0:
            break
        else:
            invalid = R_max * (np.random.random((n_invalid, 2)) - 0.5)
    
    return warps

def apply_blur(hr_image, sigma_blur=0.3):
    """
    Apply Gaussian blur to the HR image to simulate camera PSF.
    
    Args:
        hr_image: High-resolution input image
        sigma_blur: Gaussian blur sigma
        
    Returns:
        blurred_image: Blurred HR image
    """
    print(f"Applying Gaussian blur with sigma={sigma_blur}...")
    
    if len(hr_image.shape) == 3:  # RGB
        blurred = np.empty_like(hr_image)
        for c in range(hr_image.shape[2]):
            # Apply blur to each channel separately
            blurred[..., c] = gaussian_filter(hr_image[..., c], sigma=sigma_blur)
    else:  # Grayscale
        blurred = gaussian_filter(hr_image, sigma=sigma_blur)
    
    return blurred

def generate_warped_images(hr_blurred, warps):
    """
    Generate shifted versions of the HR image using the provided warps.
    
    Args:
        hr_blurred: Blurred HR image
        warps: Array of warps/shifts
        
    Returns:
        HR_warped: Array of warped images [num_frames, height, width, channels]
    """
    print("Generating warped images...")
    num_frames = warps.shape[0] + 1  # +1 for reference frame
    
    # Initialize array for warped images
    if len(hr_blurred.shape) == 3:  # RGB
        HR_warped = np.empty((num_frames, *hr_blurred.shape), dtype=np.float32)
    else:  # Grayscale
        HR_warped = np.empty((num_frames, *hr_blurred.shape), dtype=np.float32)
    
    # Add reference frame (unwarped)
    HR_warped[0] = hr_blurred
    
    # Apply warps to create shifted versions
    for j in tqdm(range(warps.shape[0]), desc="Applying shifts"):
        if len(hr_blurred.shape) == 3:  # RGB
            # Add 0 shift for color channel dimension
            shift_params = (*warps[j], 0)
            shifted = ndimage.shift(hr_blurred, shift_params)
            
            # Maintain average intensity per channel
            for c in range(hr_blurred.shape[2]):
                shifted[..., c] = shifted[..., c] * (
                    np.mean(hr_blurred[..., c]) / 
                    np.mean(shifted[..., c])
                )
        else:  # Grayscale
            shifted = ndimage.shift(hr_blurred, warps[j])
            # Normalize to preserve mean intensity
            shifted = shifted * np.mean(hr_blurred) / np.mean(shifted)
        
        HR_warped[j+1] = shifted
    
    # Clip values to [0, 1] range
    return np.clip(HR_warped, 0, 1)

def downsample_images(HR_warped, factor=2):
    """
    Downsample the HR images by the specified factor.
    
    Args:
        HR_warped: Array of warped HR images
        factor: Downsampling factor
        
    Returns:
        downsampled: Array of downsampled images
    """
    print(f"Downsampling images by factor {factor}...")
    
    if len(HR_warped.shape) == 4:  # RGB
        downsampled = HR_warped[:, ::factor, ::factor, :]
    else:  # Grayscale
        downsampled = HR_warped[:, ::factor, ::factor]
    
    return downsampled

def generate_exposure_ratios(num_frames, exposure_variation=True):
    """
    Generate random exposure ratios for burst sequence.
    
    Args:
        num_frames: Number of frames in the burst
        exposure_variation: Whether to create varied exposures
        
    Returns:
        ratios: Array of exposure ratios
    """
    if not exposure_variation:
        # All frames have the same exposure
        return np.ones(num_frames, dtype=np.float32)
    
    print("Generating exposure ratios...")
    
    # Random values between 1.2 and 1.4
    alphas = np.random.random(num_frames - 1) * (1.4 - 1.2) + 1.2
    
    # Apply random powers for wider exposure range
    choices = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    powers = np.random.choice(choices, size=num_frames - 1)
    ratios = alphas**powers
    
    # Add reference ratio (1.0) for first frame
    ratios = np.concatenate(([1.0], ratios))
    
    return ratios

def apply_noise(downsampled, ratios):
    """
    Apply realistic sensor noise to the downsampled images.
    
    Args:
        downsampled: Array of downsampled images
        ratios: Array of exposure ratios
        
    Returns:
        noised: Array of images with noise applied
    """
    print("Applying realistic sensor noise...")
    
    # Noise model parameters (from Synt_L1B_ME.py)
    a = 0.26
    b = 27
    bit_depth = 4095  # 12-bit normalization factor
    
    # Generate normalized noise
    normalised_noises = np.random.normal(0, 1, size=downsampled.shape)
    
    # Calculate noise standard deviation
    if len(downsampled.shape) == 4:  # RGB
        noise_std = np.sqrt(
            a * ratios[:, None, None, None] * downsampled * bit_depth + b
        ) / bit_depth
        
        # Apply noise and exposure ratios
        noised = downsampled * ratios[:, None, None, None] + noise_std * normalised_noises
    else:  # Grayscale
        noise_std = np.sqrt(
            a * ratios[:, None, None] * downsampled * bit_depth + b
        ) / bit_depth
        
        # Apply noise and exposure ratios
        noised = downsampled * ratios[:, None, None] + noise_std * normalised_noises
    
    # Clip values to [0, 1] range
    return np.clip(noised, 0, 1)

def generate_jittered_ratios(ratios, jitter_rates=[0, 0.05, 0.2]):
    """
    Generate jittered exposure ratios for different noise levels.
    
    Args:
        ratios: Original exposure ratios
        jitter_rates: List of jitter rates to generate
        
    Returns:
        jittered_ratios_dict: Dictionary mapping jitter rates to jittered ratios
    """
    print("Generating jittered exposure ratios...")
    
    jittered_ratios_dict = {}
    num_frames = len(ratios)
    
    for jitter_rate in jitter_rates:
        # First frame always has reference exposure
        jitter_noise = np.zeros(num_frames)
        
        # Apply jitter to remaining frames
        if jitter_rate > 0:
            jitter_noise[1:] = (np.random.random(num_frames - 1) * 2 - 1) * jitter_rate
        
        # Apply jitter to ratios
        noised_ratios = (1 + jitter_noise) * ratios
        jittered_ratios_dict[int(100 * jitter_rate)] = noised_ratios
    
    return jittered_ratios_dict

def save_visualization(burst, output_dir, scale_factor=1):
    """
    Save visualization of the generated burst.
    
    Args:
        burst: Generated burst
        output_dir: Output directory
        scale_factor: Scale factor for display
    """
    num_frames = burst.shape[0]
    
    # Create a grid visualization of all frames in the burst
    cols = min(4, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    plt.figure(figsize=(cols * 4, rows * 4))
    
    for i in range(num_frames):
        plt.subplot(rows, cols, i + 1)
        
        if len(burst.shape) == 4:  # RGB
            plt.imshow(burst[i])
        else:  # Grayscale
            plt.imshow(burst[i], cmap='gray', vmin=0, vmax=1)
            
        plt.title(f"Frame {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'burst_visualization.png')
    plt.close()
    
    # Save individual frames
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(exist_ok=True)
    
    for i in range(num_frames):
        if len(burst.shape) == 4:  # RGB
            cv2.imwrite(
                str(frames_dir / f'frame_{i:02d}.png'),
                cv2.cvtColor((burst[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
        else:  # Grayscale
            cv2.imwrite(
                str(frames_dir / f'frame_{i:02d}.png'),
                (burst[i] * 255).astype(np.uint8)
            )

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create a synthetic burst dataset from a high-resolution image")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input high-resolution image (PNG format)")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_burst",
                        help="Directory to save generated dataset")
    parser.add_argument("--scale", type=int, default=2,
                        help="Downsampling factor for generating LR images")
    parser.add_argument("--grayscale", action="store_true",
                        help="Process image as grayscale")
    parser.add_argument("--num-frames", type=int, default=10,
                        help="Number of frames to generate in the synthetic burst")
    parser.add_argument("--max-shift", type=float, default=2.0,
                        help="Maximum shift (in pixels) for burst generation")
    parser.add_argument("--blur-sigma", type=float, default=0.3,
                        help="Blur sigma for simulating camera PSF")
    parser.add_argument("--exposure-variation", action="store_true", default=True,
                        help="Create burst with varied exposures (multi-exposure)")
    parser.add_argument("--viz", action="store_true",
                        help="Save visualization of the generated burst")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Set paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the high-resolution image
    hr_image = load_hr_image(input_path, grayscale=args.grayscale)
    
    # Save the original HR image
    hr_gt_path = output_dir / 'hr_ground_truth.png'
    if len(hr_image.shape) == 3:  # RGB
        cv2.imwrite(str(hr_gt_path), cv2.cvtColor((hr_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    else:  # Grayscale
        cv2.imwrite(str(hr_gt_path), (hr_image * 255).astype(np.uint8))
    print(f"Saved original HR image to {hr_gt_path}")
    
    # Apply blur to HR image
    hr_blurred = apply_blur(hr_image, sigma_blur=args.blur_sigma)
    
    # Generate random warps
    warps = pick_warps(args.num_frames - 1, R_max=args.max_shift)
    
    # Generate warped images
    HR_warped = generate_warped_images(hr_blurred, warps)
    
    # Downsample images
    downsampled = downsample_images(HR_warped, factor=args.scale)
    
    # Generate exposure ratios
    ratios = generate_exposure_ratios(args.num_frames, exposure_variation=args.exposure_variation)
    
    # Apply noise
    noised = apply_noise(downsampled, ratios)
    
    # Generate jittered ratios
    jittered_ratios = generate_jittered_ratios(ratios)
    
    # Save results
    np.save(output_dir / 'bursts.npy', noised)
    np.save(output_dir / 'ratios_gt.npy', ratios)
    
    # Save jittered ratios
    for jitter_rate, jittered_ratio in jittered_ratios.items():
        np.save(output_dir / f'ratios_noised_{jitter_rate}.npy', jittered_ratio)
    
    # Create metadata file
    with open(output_dir / 'dataset_info.txt', 'w') as f:
        f.write("Synthetic Burst Dataset\n")
        f.write("=" * 50 + "\n")
        f.write(f"Original image: {input_path.name}\n")
        f.write(f"Number of frames: {args.num_frames}\n")
        f.write(f"Downsampling factor: {args.scale}x\n")
        f.write(f"Maximum shift: {args.max_shift} pixels\n")
        f.write(f"Blur sigma: {args.blur_sigma}\n")
        f.write(f"Multi-exposure: {args.exposure_variation}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write("\nNoise model parameters:\n")
        f.write("a = 0.26, b = 27 (based on 12-bit normalization)\n")
        f.write("\nGenerated files:\n")
        f.write("- bursts.npy: Burst sequence with noise\n")
        f.write("- ratios_gt.npy: Ground truth exposure ratios\n")
        for jitter_rate in jittered_ratios.keys():
            f.write(f"- ratios_noised_{jitter_rate}.npy: Jittered ratios ({jitter_rate/100:.2f} jitter)\n")
    
    # Save visualization
    if args.viz:
        save_visualization(noised, output_dir)
    
    print("\nDataset generation complete!")
    print(f"Generated {args.num_frames} frames with {args.scale}x downsampling")
    print(f"Output saved to {output_dir}")
    
    # Print example command for using the dataset
    print("\nExample command for super-resolution:")
    print(f"python process_asmspotter.py --data-path {output_dir} --scale {args.scale}")

if __name__ == "__main__":
    main() 