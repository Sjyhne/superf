#!/usr/bin/env python3
"""
Exploratory script to test different Fourier feature scales across different downsampling factors.
This script runs comprehensive experiments to understand how optimal Fourier scale varies with df.

Usage: python explore_fourier_scales.py --output_folder fourier_exploration --dataset satburst_synth

The script will test:
- Fourier scales: [1, 3, 5, 10, 20] 
- Downsampling factors: [2, 4, 8]
- All combinations across all sample IDs

Results are saved in a structured format for easy analysis and plotting.
"""

import subprocess
import sys
import argparse
from pathlib import Path
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time

# Import functions from benchmark_all_samples.py to reuse existing functionality
from benchmark_all_samples import find_all_sample_ids, aggregate_results

# find_all_sample_ids is now imported from benchmark_all_samples.py

def run_experiment_configuration(fourier_scale, df, args, output_folder):
    """Run benchmark_all_samples.py for a specific fourier_scale and df configuration."""
    
    # Create experiment-specific subfolder
    exp_name = f"fs{fourier_scale}_df{df}"
    exp_folder = output_folder / exp_name
    exp_folder.mkdir(exist_ok=True)
    
    # Use benchmark_all_samples.py to ensure consistency
    cmd = [
        "python", "benchmark_all_samples.py",
        "--output_folder", str(exp_folder),
        "--dataset", args.dataset,
        "--data_root", args.data_root,
        "--df", str(df),
        "--scale_factor", str(df),  # Keep scale_factor same as df
        "--fourier_scale", str(fourier_scale),
        "--lr_shift", str(args.lr_shift),
        "--num_samples", str(args.num_samples),
        "--aug", args.aug,
        "--model", args.model,
        "--input_projection", "fourier",  # Use generic fourier
        "--network_depth", str(args.network_depth),
        "--network_hidden_dim", str(args.network_hidden_dim),
        "--projection_dim", str(args.projection_dim),
        "--seed", str(args.seed),
        "--iters", str(args.iters),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--device", str(args.device)
    ]
    
    # Add optional flags
    if args.use_gnll:
        cmd.append("--use_gnll")
    
    print(f"Running configuration: Fourier_Scale={fourier_scale}, DF={df}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Configuration completed successfully!")
        return True, None, exp_folder
    except subprocess.CalledProcessError as e:
        print(f"❌ Configuration failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False, str(e), exp_folder

def aggregate_experiment_results(output_folder, fourier_scales, dfs):
    """Aggregate results from all experiments into a comprehensive dataset using benchmark_all_samples.py results."""
    
    all_results = []
    
    for fourier_scale, df in product(fourier_scales, dfs):
        exp_name = f"fs{fourier_scale}_df{df}"
        exp_folder = output_folder / exp_name
        
        # Look for aggregated results file created by benchmark_all_samples.py
        aggregated_files = list(exp_folder.glob("aggregated_results_*.json"))
        
        if not aggregated_files:
            print(f"No aggregated results found for {exp_name}")
            continue
        
        # Use the most recent aggregated results file
        aggregated_file = max(aggregated_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(aggregated_file, 'r') as f:
                data = json.load(f)
            
            # Extract overall aggregated metrics
            overall_metrics = data.get('aggregated_metrics', {})
            timing_metrics = data.get('timing_metrics', {})
            model_name = data.get('model_name', '')
            
            # Create result entry for this configuration
            config_result = {
                'fourier_scale': fourier_scale,
                'df': df,
                'exp_name': exp_name,
                'model_name': model_name,
                
                # Performance metrics (overall across all samples)
                'model_psnr': overall_metrics.get('mean_model_psnr', np.nan),
                'model_psnr_std': overall_metrics.get('std_model_psnr', np.nan),
                'model_ssim': overall_metrics.get('mean_model_ssim', np.nan),
                'model_ssim_std': overall_metrics.get('std_model_ssim', np.nan),
                'model_lpips': overall_metrics.get('mean_model_lpips', np.nan),
                'model_lpips_std': overall_metrics.get('std_model_lpips', np.nan),
                
                'bilinear_psnr': overall_metrics.get('mean_bilinear_psnr', np.nan),
                'bilinear_psnr_std': overall_metrics.get('std_bilinear_psnr', np.nan),
                'bilinear_ssim': overall_metrics.get('mean_bilinear_ssim', np.nan),
                'bilinear_ssim_std': overall_metrics.get('std_bilinear_ssim', np.nan),
                'bilinear_lpips': overall_metrics.get('mean_bilinear_lpips', np.nan),
                'bilinear_lpips_std': overall_metrics.get('std_bilinear_lpips', np.nan),
                
                # Improvements over bilinear
                'psnr_improvement': overall_metrics.get('mean_psnr_improvement', np.nan),
                'ssim_improvement': overall_metrics.get('mean_ssim_improvement', np.nan),
                'lpips_improvement': overall_metrics.get('mean_lpips_improvement', np.nan),
                
                # Alignment metrics
                'alignment_error': overall_metrics.get('mean_alignment_error', np.nan),
                'alignment_error_std': overall_metrics.get('std_alignment_error', np.nan),
                'final_dx': overall_metrics.get('mean_final_dx', np.nan),
                'final_dx_std': overall_metrics.get('std_final_dx', np.nan),
                'final_dy': overall_metrics.get('mean_final_dy', np.nan),
                'final_dy_std': overall_metrics.get('std_final_dy', np.nan),
                
                # Sample statistics
                'num_samples': overall_metrics.get('num_samples', 0),
                'num_sample_ids': overall_metrics.get('num_sample_ids', 0),
                
                # Timing
                'mean_training_time_minutes': timing_metrics.get('mean_training_time_minutes', np.nan),
                'total_training_time_minutes': timing_metrics.get('total_training_time_minutes', np.nan),
                'min_training_time_minutes': timing_metrics.get('min_training_time_minutes', np.nan),
                'max_training_time_minutes': timing_metrics.get('max_training_time_minutes', np.nan),
                
                # File reference
                'aggregated_file': str(aggregated_file)
            }
            
            all_results.append(config_result)
            
            # Also extract per-sample results if needed for detailed analysis
            sample_results = data.get('sample_results', [])
            for sample_result in sample_results:
                sample_metrics = sample_result.get('metrics', {})
                sample_training_metrics = sample_result.get('training_metrics', {})
                sample_id = sample_result.get('sample_id', 'unknown')
                
                per_sample_result = {
                    'sample_id': sample_id,
                    'fourier_scale': fourier_scale,
                    'df': df,
                    'exp_name': exp_name,
                    'result_type': 'per_sample',
                    
                    # Per-sample metrics
                    'model_psnr': sample_metrics.get('mean_model_psnr', np.nan),
                    'model_ssim': sample_metrics.get('mean_model_ssim', np.nan),
                    'model_lpips': sample_metrics.get('mean_model_lpips', np.nan),
                    'bilinear_psnr': sample_metrics.get('mean_bilinear_psnr', np.nan),
                    'bilinear_ssim': sample_metrics.get('mean_bilinear_ssim', np.nan),
                    'bilinear_lpips': sample_metrics.get('mean_bilinear_lpips', np.nan),
                    
                    'psnr_improvement': sample_metrics.get('mean_model_psnr', np.nan) - sample_metrics.get('mean_bilinear_psnr', np.nan) if not np.isnan(sample_metrics.get('mean_model_psnr', np.nan)) and not np.isnan(sample_metrics.get('mean_bilinear_psnr', np.nan)) else np.nan,
                    'ssim_improvement': sample_metrics.get('mean_model_ssim', np.nan) - sample_metrics.get('mean_bilinear_ssim', np.nan) if not np.isnan(sample_metrics.get('mean_model_ssim', np.nan)) and not np.isnan(sample_metrics.get('mean_bilinear_ssim', np.nan)) else np.nan,
                    'lpips_improvement': sample_metrics.get('mean_bilinear_lpips', np.nan) - sample_metrics.get('mean_model_lpips', np.nan) if not np.isnan(sample_metrics.get('mean_model_lpips', np.nan)) and not np.isnan(sample_metrics.get('mean_bilinear_lpips', np.nan)) else np.nan,
                    
                    'alignment_error': sample_metrics.get('mean_alignment_error', np.nan),
                    'final_dx': sample_metrics.get('mean_final_dx', np.nan),
                    'final_dy': sample_metrics.get('mean_final_dy', np.nan),
                    'training_time_minutes': sample_training_metrics.get('training_time_minutes', np.nan),
                    
                    'source_file': sample_result.get('file_path', '')
                }
                
                all_results.append(per_sample_result)
                
        except Exception as e:
            print(f"Error reading {aggregated_file}: {e}")
            continue
    
    return pd.DataFrame(all_results)

def create_analysis_plots(df, output_folder):
    """Create comprehensive analysis plots."""
    
    plots_folder = output_folder / "analysis_plots"
    plots_folder.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Filter to configuration-level results (not per-sample)
    config_df = df[df.get('result_type', 'config') != 'per_sample'].copy()
    
    if config_df.empty:
        print("No configuration-level results found for plotting")
        return None
    
    # 1. Heatmap: PSNR improvement by Fourier scale and DF
    plt.figure(figsize=(10, 6))
    pivot_psnr = config_df.pivot(index='fourier_scale', columns='df', values='psnr_improvement')
    sns.heatmap(pivot_psnr, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0)
    plt.title('Mean PSNR Improvement by Fourier Scale and Downsampling Factor')
    plt.xlabel('Downsampling Factor (df)')
    plt.ylabel('Fourier Scale')
    plt.tight_layout()
    plt.savefig(plots_folder / 'psnr_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Line plot: Performance vs Fourier scale for each DF
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['psnr_improvement', 'ssim_improvement', 'lpips_improvement']
    titles = ['PSNR Improvement', 'SSIM Improvement', 'LPIPS Improvement (lower is better)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        for df_val in sorted(config_df['df'].unique()):
            df_subset = config_df[config_df['df'] == df_val]
            
            # Sort by fourier_scale for proper line plotting
            df_subset = df_subset.sort_values('fourier_scale')
            
            ax.plot(df_subset['fourier_scale'], df_subset[metric], 
                   marker='o', label=f'DF={df_val}', linewidth=2, markersize=6)
        
        ax.set_xlabel('Fourier Scale')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if metric == 'lpips_improvement':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(plots_folder / 'performance_vs_fourier_scale.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Bar plot: Performance comparison for each combination
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    metrics = ['psnr_improvement', 'ssim_improvement', 'lpips_improvement', 'mean_training_time_minutes']
    titles = ['PSNR Improvement', 'SSIM Improvement', 'LPIPS Improvement', 'Training Time (minutes)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Create combined labels for x-axis
        config_df['combo_label'] = config_df['fourier_scale'].astype(str) + '_DF' + config_df['df'].astype(str)
        
        bars = ax.bar(config_df['combo_label'], config_df[metric])
        ax.set_xlabel('Fourier Scale_DF')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Configuration')
        ax.tick_params(axis='x', rotation=45)
        
        # Color bars based on performance
        if metric in ['psnr_improvement', 'ssim_improvement']:
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            # Color positive improvements green, negative red
            for j, bar in enumerate(bars):
                val = config_df[metric].iloc[j]
                if val > 0:
                    bar.set_color('green')
                    bar.set_alpha(0.7)
                else:
                    bar.set_color('red')
                    bar.set_alpha(0.7)
        elif metric == 'lpips_improvement':
            # For LPIPS, positive improvement (lower LPIPS) is good
            for j, bar in enumerate(bars):
                val = config_df[metric].iloc[j]
                if val > 0:
                    bar.set_color('green')
                    bar.set_alpha(0.7)
                else:
                    bar.set_color('red')
                    bar.set_alpha(0.7)
    
    plt.tight_layout()
    plt.savefig(plots_folder / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation matrix
    plt.figure(figsize=(10, 8))
    numeric_cols = ['fourier_scale', 'df', 'model_psnr', 'model_ssim', 'model_lpips', 
                   'psnr_improvement', 'ssim_improvement', 'lpips_improvement', 
                   'alignment_error', 'mean_training_time_minutes']
    
    # Only include columns that exist in the dataframe
    available_cols = [col for col in numeric_cols if col in config_df.columns]
    
    correlation_matrix = config_df[available_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Key Metrics')
    plt.tight_layout()
    plt.savefig(plots_folder / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Best configuration analysis
    plt.figure(figsize=(12, 8))
    
    # Find best fourier scale for each DF based on PSNR improvement
    best_per_df = config_df.loc[config_df.groupby('df')['psnr_improvement'].idxmax()]
    
    # Create a bar plot
    bars = plt.bar(best_per_df['df'], best_per_df['psnr_improvement'])
    plt.title('Best Fourier Scale per Downsampling Factor (by PSNR Improvement)')
    plt.xlabel('Downsampling Factor')
    plt.ylabel('PSNR Improvement (dB)')
    
    # Add value labels on bars
    for i, (_, row) in enumerate(best_per_df.iterrows()):
        plt.text(row['df'], row['psnr_improvement'] + 0.01, 
                f"FS={row['fourier_scale']:.0f}", 
                ha='center', va='bottom', fontweight='bold')
    
    # Color bars
    for bar in bars:
        bar.set_color('steelblue')
        bar.set_alpha(0.8)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(plots_folder / 'best_fourier_scales.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_per_df

def create_summary_report(df, best_configs, output_folder):
    """Create a comprehensive summary report."""
    
    report_file = output_folder / "exploration_summary_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FOURIER SCALE EXPLORATION SUMMARY REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Filter to config-level results for overview
        config_df = df[df.get('result_type', 'config') != 'per_sample']
        
        # Basic statistics
        f.write("EXPERIMENT OVERVIEW:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total configurations: {len(config_df)}\n")
        if 'sample_id' in df.columns:
            per_sample_df = df[df.get('result_type', 'config') == 'per_sample']
            f.write(f"Sample IDs tested: {per_sample_df['sample_id'].nunique() if len(per_sample_df) > 0 else 'N/A'}\n")
        f.write(f"Fourier scales tested: {sorted(config_df['fourier_scale'].unique())}\n")
        f.write(f"Downsampling factors tested: {sorted(config_df['df'].unique())}\n\n")
        
        # Best configurations
        f.write("BEST FOURIER SCALES BY DOWNSAMPLING FACTOR:\n")
        f.write("-"*50 + "\n")
        for _, row in best_configs.iterrows():
            f.write(f"DF={row['df']}: Best Fourier Scale = {row['fourier_scale']:.0f} "
                   f"(PSNR improvement: {row['psnr_improvement']:.3f} dB)\n")
        f.write("\n")
        
        # Overall statistics
        f.write("OVERALL PERFORMANCE STATISTICS:\n")
        f.write("-"*40 + "\n")
        
        for metric in ['psnr_improvement', 'ssim_improvement', 'lpips_improvement']:
            if metric in config_df.columns:
                f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                stats = config_df[metric].describe()
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value:.4f}\n")
        
        # Performance by configuration
        f.write("\nPERFORMANCE BY CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        
        # Create a simple summary table
        f.write(f"{'Fourier Scale':<15} {'DF':<5} {'PSNR Imp':<10} {'SSIM Imp':<10} {'LPIPS Imp':<11} {'Train Time':<12}\n")
        f.write("-" * 75 + "\n")
        
        for _, row in config_df.iterrows():
            f.write(f"{row['fourier_scale']:<15.1f} "
                   f"{row['df']:<5.0f} "
                   f"{row['psnr_improvement']:<10.3f} "
                   f"{row['ssim_improvement']:<10.4f} "
                   f"{row['lpips_improvement']:<11.4f} "
                   f"{row.get('mean_training_time_minutes', 0):<12.1f}\n")
        
        f.write("\n")
        
        # Key insights
        f.write("KEY INSIGHTS:\n")
        f.write("-"*20 + "\n")
        
        # Find if there's a trend
        best_scales = best_configs['fourier_scale'].values
        if len(set(best_scales)) == 1:
            f.write(f"• Consistent optimal Fourier scale: {best_scales[0]:.0f} across all DFs\n")
        else:
            f.write("• Optimal Fourier scale varies with downsampling factor:\n")
            for _, row in best_configs.iterrows():
                f.write(f"  - DF={row['df']}: Fourier Scale {row['fourier_scale']:.0f}\n")
        
        # Performance ranges
        f.write(f"• PSNR improvement range: {config_df['psnr_improvement'].min():.3f} to {config_df['psnr_improvement'].max():.3f} dB\n")
        best_idx = config_df['psnr_improvement'].idxmax()
        f.write(f"• Best overall configuration: Fourier Scale {config_df.loc[best_idx, 'fourier_scale']:.0f}, "
               f"DF={config_df.loc[best_idx, 'df']}\n")
        
        if 'mean_training_time_minutes' in config_df.columns:
            time_col = 'mean_training_time_minutes'
        else:
            time_col = None
        
        if time_col and not config_df[time_col].isna().all():
            f.write(f"• Training time range: {config_df[time_col].min():.1f} to {config_df[time_col].max():.1f} minutes\n")

def main():
    parser = argparse.ArgumentParser(description="Explore Fourier Scale Performance Across Different Downsampling Factors")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="satburst_synth", 
                       choices=["satburst_synth", "worldstrat", "burst_synth"])
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory")
    
    # Experiment parameters
    parser.add_argument("--fourier_scales", nargs='+', type=float, default=[1, 3, 5, 10, 20],
                       help="List of Fourier scales to test")
    parser.add_argument("--dfs", nargs='+', type=int, default=[2, 4, 8],
                       help="List of downsampling factors to test")
    
    # Model parameters (same as benchmark_all_samples.py)
    parser.add_argument("--lr_shift", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--aug", type=str, default="none", choices=['none', 'light', 'medium', 'heavy'])
    parser.add_argument("--model", type=str, default="mlp", 
                       choices=["mlp", "siren", "wire", "linear", "conv", "thera", "nir"])
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--use_gnll", action="store_true")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="7", help="CUDA device number")
    
    # Output parameters
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for results")
    parser.add_argument("--skip_experiments", action="store_true", 
                       help="Skip running experiments, only do analysis on existing results")
    
    args = parser.parse_args()
    
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)
    
    print("="*80)
    print("FOURIER SCALE EXPLORATION")
    print("="*80)
    print(f"Testing Fourier scales: {args.fourier_scales}")
    print(f"Testing downsampling factors: {args.dfs}")
    print(f"Total combinations: {len(args.fourier_scales) * len(args.dfs)}")
    
    if not args.skip_experiments:
        # Find all sample IDs
        sample_ids = find_all_sample_ids(args.data_root)
        
        if not sample_ids:
            print("No sample IDs found in data directory!")
            return
        
        print(f"Sample IDs to test: {len(sample_ids)}")
        print(f"Total experiments: {len(sample_ids) * len(args.fourier_scales) * len(args.dfs)}")
        
        # Run all experiments
        successful_runs = 0
        failed_runs = 0
        total_experiments = len(sample_ids) * len(args.fourier_scales) * len(args.dfs)
        
        # Run experiments by configuration (each configuration runs all samples)
        total_configurations = len(args.fourier_scales) * len(args.dfs)
        print(f"\nStarting {total_configurations} configurations...")
        start_time = time.time()
        
        for i, (fourier_scale, df) in enumerate(product(args.fourier_scales, args.dfs)):
            print(f"\n[{i+1}/{total_configurations}] ", end="")
            success, error, exp_folder = run_experiment_configuration(fourier_scale, df, args, output_folder)
            
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
                print(f"❌ Configuration failed - {error}")
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("EXPERIMENTS COMPLETE")
        print(f"{'='*80}")
        print(f"Successful configurations: {successful_runs}")
        print(f"Failed configurations: {failed_runs}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"Average time per configuration: {elapsed_time/total_configurations:.1f} seconds")
    
    # Aggregate and analyze results
    print("\nAggregating results...")
    results_df = aggregate_experiment_results(output_folder, args.fourier_scales, args.dfs)
    
    if results_df.empty:
        print("No results found for analysis!")
        return
    
    print(f"Found {len(results_df)} result entries for analysis")
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_folder / f"fourier_exploration_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Raw results saved to: {results_file}")
    
    # Create analysis plots
    print("Creating analysis plots...")
    best_configs = create_analysis_plots(results_df, output_folder)
    
    # Create summary report
    print("Creating summary report...")
    create_summary_report(results_df, best_configs, output_folder)
    
    # Print quick summary
    print(f"\n{'='*80}")
    print("EXPLORATION SUMMARY")
    print(f"{'='*80}")
    print("\nBest Fourier scales by downsampling factor:")
    for _, row in best_configs.iterrows():
        print(f"  DF={row['df']}: Fourier Scale {row['fourier_scale']:.0f} "
              f"(PSNR improvement: {row['psnr_improvement']:.3f} dB)")
    
    print(f"\nAll results and plots saved in: {output_folder}")
    print("Check the analysis_plots/ folder for visualizations")
    print("Check exploration_summary_report.txt for detailed analysis")

if __name__ == "__main__":
    main()
