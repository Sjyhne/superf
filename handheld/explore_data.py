#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import sys
from tqdm import tqdm

def load_npy_file(file_path):
    """Load a .npy file and return its contents."""
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

def get_basic_stats(data):
    """Calculate basic statistics of the data."""
    stats = {
        'shape': data.shape,
        'dtype': data.dtype,
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'num_zeros': np.sum(data == 0),
        'num_nan': np.sum(np.isnan(data)),
        'num_inf': np.sum(np.isinf(data))
    }
    return stats

def save_image(img_data, save_path):
    """Save a single image as PNG at its original resolution."""
    # Ensure the data is in the correct range [0, 1]
    img_data = np.clip(img_data, 0, 1)
    
    # Calculate figure size in inches to match the original pixel dimensions
    dpi = 100
    height, width = img_data.shape
    figsize = (width/dpi, height/dpi)
    
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img_data, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

def save_sample_grid(sample_images, save_path, num_cols=5):
    """Save a grid of images from one sample at their original resolution."""
    num_images = len(sample_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    # Get dimensions of a single image
    height, width = sample_images[0].shape
    dpi = 100
    
    # Calculate total figure size in inches
    total_width = width * num_cols
    total_height = height * num_rows
    figsize = (total_width/dpi, total_height/dpi)
    
    plt.figure(figsize=figsize, dpi=dpi)
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(sample_images[i], cmap='gray')
        plt.axis('off')
        plt.title(f'Frame {i}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def save_data_as_images(data, output_dir):
    """Save 4D array data as organized PNG images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples, num_frames, height, width = data.shape
    print(f"\nSaving {num_samples} samples with {num_frames} frames each...")
    
    # Create a summary file
    with open(output_dir / 'dataset_info.txt', 'w') as f:
        f.write(f"Dataset Summary:\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Frames per sample: {num_frames}\n")
        f.write(f"Image dimensions: {height}x{width}\n")
        f.write(f"Value range: [{np.min(data):.3f}, {np.max(data):.3f}]\n")
        f.write(f"Mean value: {np.mean(data):.3f}\n")
        f.write(f"Std deviation: {np.std(data):.3f}\n")
    
    # Save individual frames and sample grids
    for sample_idx in tqdm(range(num_samples), desc="Saving samples"):
        # Create sample directory
        sample_dir = output_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Save individual frames
        frames_dir = sample_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        for frame_idx in range(num_frames):
            frame_path = frames_dir / f"frame_{frame_idx:03d}.png"
            save_image(data[sample_idx, frame_idx], frame_path)
        
        # Save grid visualization of all frames in this sample
        grid_path = sample_dir / "sample_grid.png"
        save_sample_grid(data[sample_idx], grid_path)

def explore_data(file_path, output_dir=None):
    """Main function to explore and visualize .npy data."""
    # Load data
    data = load_npy_file(file_path)
    
    # Get and print basic statistics
    stats = get_basic_stats(data)
    print("\nData Statistics:")
    print("-" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    if output_dir:
        save_data_as_images(data, output_dir)
        print(f"\nData saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Explore and visualize .npy files')
    parser.add_argument('--file_path', type=str, help='Path to the .npy file')
    parser.add_argument('--output-dir', '-o', type=str, default='data', 
                       help='Directory to save visualizations (default: data)')
    args = parser.parse_args()
    
    explore_data(args.file_path, args.output_dir)

if __name__ == "__main__":
    main()
