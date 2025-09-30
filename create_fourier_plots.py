#!/usr/bin/env python3
"""
Create PSNR vs Fourier scale plots from fourier_exploration results.
One figure per downsampling factor (2x, 4x, 8x).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def extract_experiment_info(folder_name):
    """Extract dataset, loss_type, df, and fourier_scale from folder name."""
    # Pattern: {dataset}_{loss_type}_df{df}_fs{fourier_scale}
    pattern = r'(.+)_(mse|gnll)_df(\d+)_fs(\d+)'
    match = re.match(pattern, folder_name)
    
    if match:
        dataset = match.group(1)
        loss_type = match.group(2)
        df = int(match.group(3))
        fourier_scale = int(match.group(4))
        return dataset, loss_type, df, fourier_scale
    return None, None, None, None

def load_fourier_results(results_folder):
    """Load all Fourier exploration results into a DataFrame."""
    results_folder = Path(results_folder)
    data = []
    
    for exp_folder in results_folder.iterdir():
        if not exp_folder.is_dir():
            continue
            
        # Extract experiment info from folder name
        dataset, loss_type, df, fourier_scale = extract_experiment_info(exp_folder.name)
        
        if df is None or fourier_scale is None:
            print(f"Skipping folder with unexpected name: {exp_folder.name}")
            continue
        
        # Load summary statistics
        stats_file = exp_folder / "summary_statistics.json"
        if not stats_file.exists():
            print(f"No summary_statistics.json found in {exp_folder}")
            continue
            
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # Extract PSNR data
            psnr_data = stats.get('psnr', {})
            
            data.append({
                'dataset': dataset,
                'loss_type': loss_type,
                'df': df,
                'fourier_scale': fourier_scale,
                'model_psnr': psnr_data.get('model_mean', 0),
                'model_psnr_std': psnr_data.get('model_std', 0),
                'bilinear_psnr': psnr_data.get('bilinear_mean', 0),
                'bilinear_psnr_std': psnr_data.get('bilinear_std', 0),
                'psnr_improvement': psnr_data.get('improvement_mean', 0),
                'psnr_improvement_std': psnr_data.get('improvement_std', 0),
                'total_samples': stats.get('total_samples', 0)
            })
            
        except Exception as e:
            print(f"Error loading {stats_file}: {e}")
            continue
    
    return pd.DataFrame(data)

def create_psnr_plots(df, output_folder):
    """Create PSNR vs Fourier scale plots, one per downsampling factor."""
    output_folder = Path(output_folder)
    plots_folder = output_folder / "analysis_plots"
    plots_folder.mkdir(exist_ok=True)
    
    # Set style with much larger fonts
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 26
    })
    
    # Define colors and styles to match the reference
    colors = {'satburst_synth': '#0066CC', 'burst_synth': '#CC0000'}  # True blue and true red
    line_styles = {'mse': '-', 'gnll': ':'}  # Solid for MSE, dotted for GNLL
    markers = {'mse': 's', 'gnll': 'o'}  # Square for MSE, circle for GNLL
    marker_sizes = {'mse': 12, 'gnll': 12}
    
    # Dataset name mapping
    dataset_names = {
        'satburst_synth': 'SatSynthBurst',
        'burst_synth': 'SyntheticBurst'
    }
    
    # Loss type name mapping
    loss_names = {
        'mse': 'MSE Loss',
        'gnll': 'GNLL Loss'
    }
    
    # Create one plot per downsampling factor
    for df_val in sorted(df['df'].unique()):
        df_subset = df[df['df'] == df_val].sort_values('fourier_scale')
        
        if df_subset.empty:
            continue
        
        plt.figure(figsize=(16, 10))
        
        # Plot lines for each dataset/loss combination
        for dataset in df['dataset'].unique():
            for loss_type in df['loss_type'].unique():
                subset = df_subset[(df_subset['dataset'] == dataset) & 
                                 (df_subset['loss_type'] == loss_type)]
                
                if subset.empty:
                    continue
                
                # Create label with proper names
                dataset_name = dataset_names.get(dataset, dataset)
                loss_name = loss_names.get(loss_type, loss_type)
                label = f"{dataset_name} - {loss_name}"
                
                # Plot without error bars
                plt.plot(
                    subset['fourier_scale'],
                    subset['model_psnr'],
                    marker=markers[loss_type],
                    linestyle=line_styles[loss_type],
                    linewidth=3.5,
                    markersize=marker_sizes[loss_type],
                    color=colors[dataset],
                    label=label,
                    markeredgewidth=1,
                    markeredgecolor='white'
                )
        
        plt.xlabel('Fourier Scale')
        plt.ylabel('PSNR (dB)')
        plt.grid(True, alpha=0.3)
        
        # Find the best legend position that doesn't interfere with data
        # Check if there's space in upper right, otherwise try other positions
        y_max = df_subset['model_psnr'].max()
        y_min = df_subset['model_psnr'].min()
        y_range = y_max - y_min
        
        # If the data is concentrated in upper part, place legend in lower left
        if y_max > y_min + 0.7 * y_range:
            legend_loc = 'lower left'
        else:
            legend_loc = 'upper right'
            
        plt.legend(loc=legend_loc, framealpha=0.9)
        plt.tight_layout()
        
        # Force integer ticks on x-axis
        xticks = sorted(df_subset['fourier_scale'].unique())
        plt.xticks(xticks, [str(int(x)) for x in xticks])
        
        # Save the plot in both PNG and PDF formats
        base_filename = f'psnr_vs_fourier_df{int(df_val)}'
        
        # Save as PNG
        png_file = plots_folder / f'{base_filename}.png'
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {png_file}")
        
        # Save as PDF
        pdf_file = plots_folder / f'{base_filename}.pdf'
        plt.savefig(pdf_file, bbox_inches='tight')
        print(f"Saved: {pdf_file}")
        
        plt.close()

def main():
    import sys
    
    # Allow custom results folder as command line argument
    if len(sys.argv) > 1:
        results_folder = sys.argv[1]
    else:
        results_folder = "fourier_exploration_newest"
    
    output_folder = results_folder
    
    print("Loading Fourier exploration results...")
    df = load_fourier_results(results_folder)
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"Loaded {len(df)} experiment results")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Loss types: {df['loss_type'].unique()}")
    print(f"Downsampling factors: {sorted(df['df'].unique())}")
    print(f"Fourier scales: {sorted(df['fourier_scale'].unique())}")
    
    # Save raw data
    df.to_csv(f"{output_folder}/fourier_results_summary.csv", index=False)
    print(f"Raw data saved to: {output_folder}/fourier_results_summary.csv")
    
    # Create plots
    print("Creating PSNR vs Fourier scale plots...")
    create_psnr_plots(df, output_folder)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
