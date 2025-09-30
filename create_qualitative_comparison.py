#!/usr/bin/env python3
"""
Create qualitative comparison with LR / Bilinear / Prediction / HR layout.
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

def load_sample_images(sample_path):
    """Load all images for a single sample."""
    sample_path = Path(sample_path)
    
    if not sample_path.exists():
        print(f"Sample path does not exist: {sample_path}")
        return None
    
    print(f"Loading images from: {sample_path}")
    
    # Define the image files and their keys
    image_files = {
        'lr': 'lr_original.png',
        'bilinear': 'bilinear_baseline.png', 
        'prediction': 'model_output_aligned.png',
        'hr': 'ground_truth.png'
    }
    
    data = {}
    for key, filename in image_files.items():
        image_path = sample_path / filename
        print(f"  📁 {key.title()} path: {image_path}")
        data[key] = load_image_safely(image_path)
    
    return data

def create_qualitative_comparison(sample_paths, output_path, base_path="/raid/home/sandej17/satellite_sr"):
    """Create a qualitative comparison with LR / Bilinear / Prediction / HR layout.
    
    Args:
        sample_paths: List of sample paths (relative to base_path)
        output_path: Output path for the image
        base_path: Base path for data
    """
    base_path = Path(base_path)
    
    print(f"Creating qualitative comparison for {len(sample_paths)} samples")
    
    n_cols = 4  # LR, Bilinear, Prediction, HR
    n_rows = len(sample_paths)
    
    # Create the figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Column headers
    column_names = ['LR', 'Bilinear', 'Prediction', 'HR']
    
    # Add column headers directly above the first row of images
    for col, name in enumerate(column_names):
        axes[0, col].text(0.5, 1.02, name, ha='center', va='bottom', 
                         transform=axes[0, col].transAxes, fontsize=14, fontweight='bold')
    
    # Process each sample
    for row, sample_path in enumerate(sample_paths):
        print(f"\nProcessing sample {row + 1}/{len(sample_paths)}: {sample_path}")
        
        # Load data for this sample
        data = load_sample_images(base_path / sample_path)
        if data is None:
            print(f"Failed to load data for sample: {sample_path}")
            # Add empty row
            for col in range(n_cols):
                axes[row, col].text(0.5, 0.5, f'Failed to load\n{sample_path}', ha='center', va='center', 
                                  transform=axes[row, col].transAxes, fontsize=10)
                axes[row, col].axis('off')
            continue
        
        # Display each column
        columns = ['lr', 'bilinear', 'prediction', 'hr']
        
        for col, key in enumerate(columns):
            if key in data and data[key] is not None:
                axes[row, col].imshow(data[key])
            else:
                axes[row, col].text(0.5, 0.5, f'{key}\n(Not Found)', ha='center', va='center', 
                                  transform=axes[row, col].transAxes, fontsize=10)
            axes[row, col].axis('off')
    
    # Remove the title completely
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01, wspace=0.05)  # Minimal vertical spacing, especially for headers
    
    # Save as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1)
    
    plt.close()
    
    print(f"Qualitative comparison saved to: {output_path}")
    print(f"PDF version saved to: {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Create qualitative comparison with LR / Bilinear / Prediction / HR layout")
    
    # Sample specification
    parser.add_argument("--samples", nargs='+', type=str, required=True,
                       help="List of sample paths (e.g., 'single_samples/burst_synth/6' 'single_samples/worldstrat_test/Amnesty POI-17-2-3')")
    parser.add_argument("--output_path", type=str, default="qualitative_comparison.png",
                       help="Output path for the qualitative comparison image")
    parser.add_argument("--base_path", type=str, default="/raid/home/sandej17/satellite_sr",
                       help="Base path for data")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the qualitative comparison
    create_qualitative_comparison(args.samples, output_path, args.base_path)
    
    print(f"\n🎉 Qualitative comparison created successfully!")
    print(f"📁 Output: {output_path}")
    print(f"📊 Layout: {len(args.samples)} rows × 4 columns")
    print(f"🔍 Samples: {args.samples}")

if __name__ == "__main__":
    main()
