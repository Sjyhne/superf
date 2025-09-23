#!/usr/bin/env python3
"""
Create separate plots showing actual metric values for different fourier scales and downsample factors.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_data():
    """Load the fourier exploration results CSV file."""
    csv_path = Path("/raid/home/sandej17/satellite_sr/fourier_results/fourier_exploration_results_20250917_000757.csv")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    # Load the data, filtering for aggregated results only
    df = pd.read_csv(csv_path)
    
    # Filter for aggregated results (not per-sample results)
    aggregated_df = df[df['result_type'].isna()].copy()  # Aggregated results have NaN in result_type
    
    print(f"Loaded {len(aggregated_df)} aggregated results")
    print(f"Fourier scales: {sorted(aggregated_df['fourier_scale'].unique())}")
    print(f"Downsample factors: {sorted(aggregated_df['df'].unique())}")
    
    return aggregated_df

def create_psnr_plot(df, output_dir):
    """Create plot showing PSNR values for model vs bilinear."""
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    fourier_scales = sorted(df['fourier_scale'].unique())
    downsample_factors = sorted(df['df'].unique())
    
    # Colors for different downsample factors
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for i, df_val in enumerate(downsample_factors):
        subset = df[df['df'] == df_val]
        
        # Plot model PSNR only
        plt.plot(subset['fourier_scale'], subset['model_psnr'], 
                marker=markers[i], linestyle='-', 
                label=f'Model PSNR (DF={df_val})', linewidth=2, markersize=8, 
                color=colors[i])
    
    plt.xlabel('Fourier Scale')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Values Across Fourier Scales and Downsample Factors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'psnr_values_by_fourier_scale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PSNR plot to {output_path}")

def create_ssim_plot(df, output_dir):
    """Create plot showing SSIM values for model vs bilinear."""
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    fourier_scales = sorted(df['fourier_scale'].unique())
    downsample_factors = sorted(df['df'].unique())
    
    # Colors for different downsample factors
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for i, df_val in enumerate(downsample_factors):
        subset = df[df['df'] == df_val]
        
        # Plot model SSIM only
        plt.plot(subset['fourier_scale'], subset['model_ssim'], 
                marker=markers[i], linestyle='-', 
                label=f'Model SSIM (DF={df_val})', linewidth=2, markersize=8, 
                color=colors[i])
    
    plt.xlabel('Fourier Scale')
    plt.ylabel('SSIM')
    plt.title('SSIM Values Across Fourier Scales and Downsample Factors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.ylim(0, 1)  # SSIM is bounded between 0 and 1
    
    plt.tight_layout()
    output_path = output_dir / 'ssim_values_by_fourier_scale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SSIM plot to {output_path}")

def create_lpips_plot(df, output_dir):
    """Create plot showing LPIPS values for model vs bilinear."""
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    fourier_scales = sorted(df['fourier_scale'].unique())
    downsample_factors = sorted(df['df'].unique())
    
    # Colors for different downsample factors
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for i, df_val in enumerate(downsample_factors):
        subset = df[df['df'] == df_val]
        
        # Plot model LPIPS only (note: lower is better for LPIPS)
        plt.plot(subset['fourier_scale'], subset['model_lpips'], 
                marker=markers[i], linestyle='-', 
                label=f'Model LPIPS (DF={df_val})', linewidth=2, markersize=8, 
                color=colors[i])
    
    plt.xlabel('Fourier Scale')
    plt.ylabel('LPIPS (lower is better)')
    plt.title('LPIPS Values Across Fourier Scales and Downsample Factors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'lpips_values_by_fourier_scale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved LPIPS plot to {output_path}")

def create_alignment_error_plot(df, output_dir):
    """Create plot showing alignment error values."""
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    fourier_scales = sorted(df['fourier_scale'].unique())
    downsample_factors = sorted(df['df'].unique())
    
    # Colors for different downsample factors
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for i, df_val in enumerate(downsample_factors):
        subset = df[df['df'] == df_val]
        
        # Plot alignment error
        plt.plot(subset['fourier_scale'], subset['alignment_error'], 
                marker=markers[i], linestyle='-', 
                label=f'Alignment Error (DF={df_val})', linewidth=2, markersize=8, 
                color=colors[i])
    
    plt.xlabel('Fourier Scale')
    plt.ylabel('Alignment Error')
    plt.title('Alignment Error Across Fourier Scales and Downsample Factors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'alignment_error_by_fourier_scale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Alignment Error plot to {output_path}")

def create_training_time_plot(df, output_dir):
    """Create plot showing training time values."""
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    fourier_scales = sorted(df['fourier_scale'].unique())
    downsample_factors = sorted(df['df'].unique())
    
    # Colors for different downsample factors
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for i, df_val in enumerate(downsample_factors):
        subset = df[df['df'] == df_val]
        
        # Plot training time
        plt.plot(subset['fourier_scale'], subset['mean_training_time_minutes'], 
                marker=markers[i], linestyle='-', 
                label=f'Training Time (DF={df_val})', linewidth=2, markersize=8, 
                color=colors[i])
    
    plt.xlabel('Fourier Scale')
    plt.ylabel('Training Time (minutes)')
    plt.title('Training Time Across Fourier Scales and Downsample Factors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'training_time_by_fourier_scale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Training Time plot to {output_path}")

def main():
    """Main function to create all plots."""
    # Set up output directory
    output_dir = Path("/raid/home/sandej17/satellite_sr/fourier_results/separate_metric_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Load data
    df = load_data()
    
    # Create individual plots
    print("Creating PSNR plot...")
    create_psnr_plot(df, output_dir)
    
    print("Creating SSIM plot...")
    create_ssim_plot(df, output_dir)
    
    print("Creating LPIPS plot...")
    create_lpips_plot(df, output_dir)
    
    print("Creating Alignment Error plot...")
    create_alignment_error_plot(df, output_dir)
    
    print("Creating Training Time plot...")
    create_training_time_plot(df, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
