#!/usr/bin/env python3
"""
Analysis-only script for Fourier scale exploration results.
Use this when you want to re-analyze existing results without re-running experiments.

Usage: python analyze_fourier_results.py --results_folder fourier_exploration_20250916_123456
"""

import argparse
from pathlib import Path
import sys
from typing import Optional
import json

# Add the main script to path to import functions
sys.path.append(str(Path(__file__).parent))

from explore_fourier_scales import aggregate_experiment_results, create_analysis_plots, create_summary_report
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_psnr_vs_fourier_per_df(results_df: pd.DataFrame, output_folder: Path) -> Optional[pd.DataFrame]:
    """
    Create line plots of PSNR (dB) vs Fourier feature scale with one figure per
    downsampling factor (DF). Saves figures into analysis_plots/.

    Returns the configuration-level dataframe used for plotting for potential
    downstream use.
    """
    plots_folder = output_folder / "analysis_plots"
    plots_folder.mkdir(exist_ok=True)

    # Keep only configuration-level rows
    if 'result_type' in results_df.columns:
        config_df = results_df[results_df['result_type'] != 'per_sample'].copy()
    else:
        config_df = results_df.copy()

    if config_df.empty:
        print("No configuration-level results found for PSNR plots")
        return None

    # Use a clean consistent style
    plt.style.use('default')
    sns.set_palette("colorblind")

    # Ensure numeric sorting on Fourier scale
    config_df = config_df.sort_values(['df', 'fourier_scale'])

    # Plot one figure per DF
    for df_val in sorted(config_df['df'].unique()):
        df_subset = config_df[config_df['df'] == df_val].sort_values('fourier_scale')

        if df_subset.empty:
            continue

        plt.figure(figsize=(8, 4.5))

        # Model PSNR only (no bilinear baseline)
        if {'fourier_scale', 'model_psnr'}.issubset(df_subset.columns):
            x_vals = df_subset['fourier_scale'].astype(int)
            plt.plot(
                x_vals,
                df_subset['model_psnr'],
                marker='o', linestyle='-', linewidth=2, markersize=6,
                label='Model PSNR'
            )

        plt.xlabel('Fourier Scale')
        plt.ylabel('PSNR (dB)')
        # No title per request
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Force integer ticks on x-axis
        xticks = sorted(df_subset['fourier_scale'].astype(int).unique())
        plt.xticks(xticks, [str(x) for x in xticks])

        out_path = plots_folder / f'psnr_vs_fourier_df{int(df_val)}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {out_path}")

    return config_df


def fallback_scan_aggregated_json(output_folder: Path) -> pd.DataFrame:
    """
    Fallback: scan recursively for aggregated_results_*.json files in arbitrary
    folder layouts, extract DF and Fourier scale from the first sample's
    training_info, and build a configuration-level dataframe compatible with
    plotting functions.
    """
    json_files = list(output_folder.rglob("aggregated_results_*.json"))
    rows = []

    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed reading {jf}: {e}")
            continue

        overall = data.get('aggregated_metrics', {})
        timing = data.get('timing_metrics', {})
        model_name = data.get('model_name', '')

        # Try to infer df and fourier_scale from first sample's training_info
        fourier_scale = None
        df_val = None
        samples = data.get('sample_results', [])
        if samples:
            ti = samples[0].get('training_info', {})
            fourier_scale = ti.get('fourier_scale', None)
            df_val = ti.get('downsample_factor', None)

        if fourier_scale is None or df_val is None:
            # Try to parse from path pieces like fs{scale}_df{df}
            parts = jf.as_posix().split('/')
            for p in parts:
                if p.startswith('fs') and '_df' in p:
                    try:
                        fourier_scale = float(p.split('fs')[1].split('_')[0])
                        df_val = int(p.split('_df')[1])
                    except Exception:
                        pass
                    break

        if fourier_scale is None or df_val is None:
            # Can't use this file without df/scale
            print(f"Skipping {jf} (couldn't infer df/fourier_scale)")
            continue

        rows.append({
            'result_type': 'config',
            'fourier_scale': float(fourier_scale),
            'df': int(df_val),
            'exp_name': jf.parent.name,
            'model_name': model_name,
            'model_psnr': overall.get('mean_model_psnr'),
            'model_psnr_std': overall.get('std_model_psnr'),
            'model_ssim': overall.get('mean_model_ssim'),
            'model_ssim_std': overall.get('std_model_ssim'),
            'model_lpips': overall.get('mean_model_lpips'),
            'model_lpips_std': overall.get('std_model_lpips'),
            'bilinear_psnr': overall.get('mean_bilinear_psnr'),
            'bilinear_psnr_std': overall.get('std_bilinear_psnr'),
            'bilinear_ssim': overall.get('mean_bilinear_ssim'),
            'bilinear_ssim_std': overall.get('std_bilinear_ssim'),
            'bilinear_lpips': overall.get('mean_bilinear_lpips'),
            'bilinear_lpips_std': overall.get('std_bilinear_lpips'),
            'psnr_improvement': overall.get('mean_psnr_improvement'),
            'ssim_improvement': overall.get('mean_ssim_improvement'),
            'lpips_improvement': overall.get('mean_lpips_improvement'),
            'mean_training_time_minutes': timing.get('mean_training_time_minutes'),
            'total_training_time_minutes': timing.get('total_training_time_minutes'),
            'aggregated_file': str(jf),
        })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Analyze existing Fourier exploration results")
    parser.add_argument("--results_folder", type=str, default="fourier_results",
                       help="Folder containing existing experiment results (default: fourier_results)")
    parser.add_argument("--fourier_scales", nargs='+', type=float, default=[1, 3, 5, 10, 20],
                       help="List of Fourier scales that were tested")
    parser.add_argument("--dfs", nargs='+', type=int, default=[2, 4, 8],
                       help="List of downsampling factors that were tested")
    
    args = parser.parse_args()
    
    output_folder = Path(args.results_folder)
    
    if not output_folder.exists():
        print(f"Results folder not found: {output_folder}")
        return
    
    print("="*80)
    print("FOURIER SCALE EXPLORATION - ANALYSIS ONLY")
    print("="*80)
    print(f"Analyzing results from: {output_folder}")
    
    # Aggregate and analyze results
    print("\nAggregating results...")
    results_df = aggregate_experiment_results(output_folder, args.fourier_scales, args.dfs)
    
    # Fallback: scan any aggregated_results_*.json under the folder
    if results_df.empty:
        print("No structured config folders found. Trying fallback scan...")
        results_df = fallback_scan_aggregated_json(output_folder)
        if results_df.empty:
            print("No results found for analysis!")
            return
    
    # Ensure result_type column exists for downstream plotting functions
    if 'result_type' not in results_df.columns:
        results_df['result_type'] = 'config'
    
    print(f"Found {len(results_df)} result entries for analysis")
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_folder / f"fourier_exploration_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Raw results saved to: {results_file}")
    
    # Create targeted PSNR vs Fourier plots (one per DF)
    print("Creating PSNR vs Fourier plots (one per DF)...")
    _ = plot_psnr_vs_fourier_per_df(results_df, output_folder)

    # Create broader analysis plots as well
    print("Creating additional analysis plots...")
    best_configs = create_analysis_plots(results_df, output_folder)
    
    if best_configs is not None:
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
    
    print(f"\nAnalysis complete! Results saved in: {output_folder}")
    print("Check the analysis_plots/ folder for visualizations")
    print("Check exploration_summary_report.txt for detailed analysis")

if __name__ == "__main__":
    main()
