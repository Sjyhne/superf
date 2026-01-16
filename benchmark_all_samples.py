#!/usr/bin/env python3
"""
Benchmark script that runs on all sample IDs in the dataset folder
Usage: python benchmark_all_samples.py --output_folder results --dataset satburst_synth --df 4 --model mlp
"""

import argparse
from pathlib import Path
import os
import json
from datetime import datetime
import pandas as pd
import sys

# Import the benchmark function directly
from benchmark_models import main as benchmark_main

def find_all_sample_ids(data_root, dataset_type="satburst_synth"):
    """Find all sample IDs in the data folder."""
    sample_ids = []
    
    if 'DATA_DIR_ABSOLUTE' in os.environ:
        data_path = Path(os.environ['DATA_DIR_ABSOLUTE'])
    else:
        data_path = Path(data_root)
    
    if data_path.exists():
        # For burst_synth, look in the bursts subdirectory
        if dataset_type == "burst_synth":
            bursts_path = data_path / "bursts"
            if bursts_path.exists():
                for item in bursts_path.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        # Check if corresponding gt directory exists
                        gt_dir = data_path / "gt" / item.name
                        if gt_dir.exists():
                            sample_ids.append(item.name)
        else:
            # Look for sample directories
            for item in data_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # For worldstrat_test, check if it has hr and lr subdirectories
                    if dataset_type == "worldstrat_test":
                        hr_dir = item / "hr"
                        lr_dir = item / "lr"
                        if hr_dir.exists() and lr_dir.exists():
                            sample_ids.append(item.name)
                    else:
                        # For other datasets, just check if it's a directory
                        sample_ids.append(item.name)
    
    return sorted(sample_ids)

def run_benchmark_for_sample(sample_id, args, output_folder):
    """Run benchmark for a single sample ID."""
    print(f"\n{'='*80}")
    print(f"Running benchmark for sample: {sample_id}")
    print(f"Model: {args.model} with {args.input_projection}")
    print(f"{'='*80}")
    
    try:
        # Temporarily modify sys.argv to pass arguments to benchmark_main
        original_argv = sys.argv.copy()
        
        # Build the argument list
        benchmark_args = [
            "benchmark_models.py",  # script name
            "--output_folder", str(output_folder),
            "--dataset", args.dataset,
            "--sample_id", sample_id,
            "--df", str(args.df),
            "--scale_factor", str(args.scale_factor),
            "--lr_shift", str(args.lr_shift),
            "--num_samples", str(args.num_samples),
            "--aug", args.aug,
            "--model", args.model,
            "--network_depth", str(args.network_depth),
            "--network_hidden_dim", str(args.network_hidden_dim),
            "--projection_dim", str(args.projection_dim),
            "--input_projection", args.input_projection,
            "--fourier_scale", str(args.fourier_scale),
            "--seed", str(args.seed),
            "--iters", str(args.iters),
            "--learning_rate", str(args.learning_rate),
            "--weight_decay", str(args.weight_decay),
            "--device", str(args.device),
        ]
        
        # Add optional flags
        if args.use_gnll:
            benchmark_args.append("--use_gnll")
        if args.use_base_frame:
            benchmark_args.append("--use_base_frame")
        if args.use_direct_param_T:
            benchmark_args.append("--use_direct_param_T")
        
        # Set sys.argv and call the benchmark function
        sys.argv = benchmark_args
        benchmark_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print("✅ Success!")
        return True, None
        
    except Exception as e:
        # Restore original argv in case of error
        sys.argv = original_argv
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def aggregate_results(output_folder, model_name):
    """Aggregate results from all sample runs."""
    # Find all result files for this model - use a more flexible pattern
    result_files = list(Path(output_folder).glob(f"benchmark_{model_name}_*.json"))
    
    # If no files found with the expected pattern, try to find any benchmark files
    if not result_files:
        result_files = list(Path(output_folder).glob("benchmark_*.json"))
        print(f"Found {len(result_files)} benchmark files total")
        for f in result_files:
            print(f"  - {f.name}")
    
    if not result_files:
        print(f"No results found for model: {model_name}")
        return None
    
    print(f"Found {len(result_files)} result files for aggregation")
    
    aggregated_results = {
        'model_name': model_name,
        'sample_results': [],
        'aggregated_metrics': {}
    }
    
    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_bilinear_psnr = []
    all_bilinear_ssim = []
    all_bilinear_lpips = []
    all_alignment_errors = []
    all_training_times = []
    all_final_dx = []
    all_final_dy = []
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract sample ID from model_info
            sample_id = data.get('model_info', {}).get('sample_id', 'unknown')
            
            # Get aggregated metrics for this sample
            metrics = data.get('aggregated_metrics', {})
            training_info = data.get('model_info', {})
            training_metrics = data.get('timing_metrics', {})  # Changed from 'training_metrics' to 'timing_metrics'
            
            sample_result = {
                'sample_id': sample_id,
                'file_path': str(result_file),
                'metrics': metrics,
                'training_info': training_info,
                'training_metrics': training_metrics
            }
            
            aggregated_results['sample_results'].append(sample_result)
            
            # Collect metrics for overall aggregation
            if metrics:
                all_psnr.append(metrics.get('mean_model_psnr', 0))
                all_ssim.append(metrics.get('mean_model_ssim', 0))
                all_lpips.append(metrics.get('mean_model_lpips', 0))
                all_bilinear_psnr.append(metrics.get('mean_bilinear_psnr', 0))
                all_bilinear_ssim.append(metrics.get('mean_bilinear_ssim', 0))
                all_bilinear_lpips.append(metrics.get('mean_bilinear_lpips', 0))
                all_alignment_errors.append(metrics.get('mean_alignment_error', 0))
                all_final_dx.append(metrics.get('mean_final_dx', 0))
                all_final_dy.append(metrics.get('mean_final_dy', 0))
            
            if training_metrics:  # Changed from training_info to training_metrics
                all_training_times.append(training_metrics.get('training_time_seconds', 0))
            
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
            continue
    
    # Calculate overall aggregated metrics
    if all_psnr:
        aggregated_results['aggregated_metrics'] = {
            'mean_model_psnr': sum(all_psnr) / len(all_psnr),
            'std_model_psnr': (sum([(x - sum(all_psnr)/len(all_psnr))**2 for x in all_psnr]) / len(all_psnr))**0.5,
            'mean_bilinear_psnr': sum(all_bilinear_psnr) / len(all_bilinear_psnr),
            'std_bilinear_psnr': (sum([(x - sum(all_bilinear_psnr)/len(all_bilinear_psnr))**2 for x in all_bilinear_psnr]) / len(all_bilinear_psnr))**0.5,
            'mean_psnr_improvement': sum([m - b for m, b in zip(all_psnr, all_bilinear_psnr)]) / len(all_psnr),
            'mean_model_ssim': sum(all_ssim) / len(all_ssim),
            'std_model_ssim': (sum([(x - sum(all_ssim)/len(all_ssim))**2 for x in all_ssim]) / len(all_ssim))**0.5,
            'mean_bilinear_ssim': sum(all_bilinear_ssim) / len(all_bilinear_ssim),
            'std_bilinear_ssim': (sum([(x - sum(all_bilinear_ssim)/len(all_bilinear_ssim))**2 for x in all_bilinear_ssim]) / len(all_bilinear_ssim))**0.5,
            'mean_ssim_improvement': sum([m - b for m, b in zip(all_ssim, all_bilinear_ssim)]) / len(all_ssim),
            'mean_model_lpips': sum(all_lpips) / len(all_lpips),
            'std_model_lpips': (sum([(x - sum(all_lpips)/len(all_lpips))**2 for x in all_lpips]) / len(all_lpips))**0.5,
            'mean_bilinear_lpips': sum(all_bilinear_lpips) / len(all_bilinear_lpips),
            'std_bilinear_lpips': (sum([(x - sum(all_bilinear_lpips)/len(all_bilinear_lpips))**2 for x in all_bilinear_lpips]) / len(all_bilinear_lpips))**0.5,
            'mean_lpips_improvement': sum([b - m for m, b in zip(all_lpips, all_bilinear_lpips)]) / len(all_lpips),
            'mean_alignment_error': sum(all_alignment_errors) / len(all_alignment_errors),
            'std_alignment_error': (sum([(x - sum(all_alignment_errors)/len(all_alignment_errors))**2 for x in all_alignment_errors]) / len(all_alignment_errors))**0.5,
            'mean_final_dx': sum(all_final_dx) / len(all_final_dx),
            'std_final_dx': (sum([(x - sum(all_final_dx)/len(all_final_dx))**2 for x in all_final_dx]) / len(all_final_dx))**0.5,
            'mean_final_dy': sum(all_final_dy) / len(all_final_dy),
            'std_final_dy': (sum([(x - sum(all_final_dy)/len(all_final_dy))**2 for x in all_final_dy]) / len(all_final_dy))**0.5,
            'num_samples': len(all_psnr),
            'num_sample_ids': len(aggregated_results['sample_results'])
        }
    
    # Add timing metrics
    if all_training_times:
        aggregated_results['timing_metrics'] = {
            'mean_training_time_seconds': sum(all_training_times) / len(all_training_times),
            'std_training_time_seconds': (sum([(x - sum(all_training_times)/len(all_training_times))**2 for x in all_training_times]) / len(all_training_times))**0.5,
            'mean_training_time_minutes': sum(all_training_times) / len(all_training_times) / 60.0,
            'total_training_time_minutes': sum(all_training_times) / 60.0,
            'min_training_time_minutes': min(all_training_times) / 60.0,
            'max_training_time_minutes': max(all_training_times) / 60.0
        }
    
    return aggregated_results

def main():
    parser = argparse.ArgumentParser(description="Benchmark All Sample IDs - Same args as optimize.py")
    
    # Same parameters as optimize.py - EXACT SAME DEFAULTS
    parser.add_argument("--dataset", type=str, default="satburst_synth", 
                       choices=["satburst_synth", "worldstrat", "burst_synth", "worldstrat_test"])
    parser.add_argument("--df", type=int, default=4, help="Downsampling factor, or upsampling factor for the data")
    parser.add_argument("--scale_factor", type=float, default=4, help="scale factor for the input training grid")
    parser.add_argument("--lr_shift", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--aug", type=str, default="none", choices=['none', 'light', 'medium', 'heavy'])
    
    # Model parameters - SAME AS optimize.py
    parser.add_argument("--model", type=str, default="mlp", 
                       choices=["mlp", "siren", "wire", "linear", "conv", "thera", "nir"])
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--input_projection", type=str, default="fourier_10", 
                       choices=["linear", "fourier_10", "fourier_5", "fourier_20", "fourier_40", "fourier", "legendre", "none"])
    parser.add_argument("--fourier_scale", type=float, default=10.0)
    parser.add_argument("--use_gnll", action="store_true")
    # Transformation control flags (same style as --use_gnll)
    parser.add_argument("--use_base_frame", action="store_true",
                   help="Enable using first frame as base frame (disabled by default)")
    parser.add_argument("--use_direct_param_T", action="store_true",
                   help="Enable directly-parameterized affine T (disabled by default)")
    
    # Training parameters - SAME AS optimize.py
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="7", help="CUDA device number")
    
    # Benchmark-specific parameters
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for results")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory")
    parser.add_argument("--worldstrat_test_root", type=str, default="worldstrat_test_data", help="Root worldstrat test data directory")
    
    args = parser.parse_args()
    
    # Find all sample IDs
    if args.dataset == "worldstrat_test":
        sample_ids = find_all_sample_ids(args.worldstrat_test_root, "worldstrat_test")
    elif args.dataset == "burst_synth":
        sample_ids = find_all_sample_ids("SyntheticBurstVal", "burst_synth")
    else:
        sample_ids = find_all_sample_ids(args.data_root, args.dataset)
    
    if not sample_ids:
        print("No sample IDs found in data directory!")
        return
    
    print(f"Found {len(sample_ids)} sample IDs: {sample_ids}")
    
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)
    
    print(f"Starting benchmark across {len(sample_ids)} sample IDs...")
    print(f"Model: {args.model} with {args.input_projection}")
    print(f"Total runs: {len(sample_ids)}")
    
    successful_runs = 0
    failed_runs = 0
    
    # Run benchmark for each sample ID
    for sample_id in sample_ids:
        success, error = run_benchmark_for_sample(sample_id, args, output_folder)
        
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
            print(f"Failed: {sample_id} - {error}")
    
    print(f"\n{'='*100}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*100}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Results saved in: {output_folder}")
    
    # List all files in output folder for debugging
    print(f"\nFiles in output folder:")
    for f in output_folder.glob("*.json"):
        print(f"  - {f.name}")
    
    # Aggregate results
    print("\nAggregating results...")
    model_name = f"{args.model}_{args.input_projection}_{args.network_depth}layers"
    aggregated_results = aggregate_results(output_folder, model_name)
    
    if aggregated_results:
        # Save aggregated results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        aggregated_file = output_folder / f"aggregated_results_{timestamp}.json"
        
        with open(aggregated_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        
        # Create summary table
        metrics = aggregated_results['aggregated_metrics']
        timing_metrics = aggregated_results.get('timing_metrics', {})
        
        summary_data = [{
            'Model': aggregated_results['model_name'],
            'Sample IDs': metrics.get('num_sample_ids', 0),
            'Mean PSNR': f"{metrics.get('mean_model_psnr', 0):.2f}",
            'Std PSNR': f"{metrics.get('std_model_psnr', 0):.2f}",
            'Mean Bilinear PSNR': f"{metrics.get('mean_bilinear_psnr', 0):.2f}",
            'PSNR Improvement': f"{metrics.get('mean_psnr_improvement', 0):.2f}",
            'Mean SSIM': f"{metrics.get('mean_model_ssim', 0):.4f}",
            'Mean LPIPS': f"{metrics.get('mean_model_lpips', 0):.4f}",
            'Mean Alignment Error': f"{metrics.get('mean_alignment_error', 0):.4f}",
            'Mean Training Time (min)': f"{timing_metrics.get('mean_training_time_minutes', 0):.2f}",
            'Total Training Time (min)': f"{timing_metrics.get('total_training_time_minutes', 0):.2f}"
        }]
        
        df = pd.DataFrame(summary_data)
        csv_file = output_folder / f"summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\nAggregated results saved to: {aggregated_file}")
        print(f"Summary CSV saved to: {csv_file}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS ALL SAMPLE IDS")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        # Print per-sample results
        print(f"\n{'='*80}")
        print("PER-SAMPLE RESULTS")
        print(f"{'='*80}")
        print(f"{'Sample ID':<20} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'Align Error':<10} {'Time (min)':<10}")
        print("-" * 80)
        for sample_result in aggregated_results['sample_results']:
            metrics = sample_result['metrics']
            training_metrics = sample_result['training_metrics']
            print(f"{sample_result['sample_id']:<20} "
                  f"{metrics.get('mean_model_psnr', 0):<8.2f} "
                  f"{metrics.get('mean_model_ssim', 0):<8.4f} "
                  f"{metrics.get('mean_model_lpips', 0):<8.4f} "
                  f"{metrics.get('mean_alignment_error', 0):<10.4f} "
                  f"{training_metrics.get('training_time_minutes', 0):<10.2f}")

if __name__ == "__main__":
    main()
