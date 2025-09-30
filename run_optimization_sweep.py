#!/usr/bin/env python3
"""
Comprehensive optimization sweep script that runs through many different configurations.
"""

import argparse
import subprocess
import sys
import os
import itertools
import json
from pathlib import Path
from datetime import datetime
import time

class OptimizationSweep:
    def __init__(self, base_args):
        self.base_args = base_args
        self.results = []
        self.start_time = datetime.now()
        
    def get_all_combinations(self):
        """Generate all combinations of hyperparameters to test."""
        
        # Define parameter grids
        param_grids = {
            'model': ['mlp', 'siren', 'wire'],
            'network_depth': [4, 6, 8],
            'network_hidden_dim': [128, 256, 512],
            'learning_rate': [1e-4, 1e-3, 1e-2],
            'weight_decay': [1e-6, 1e-5, 1e-4],
            'projection_dim': [64, 128, 256],
            'input_projection': ['linear', 'fourier', 'fourier_10', 'fourier_20'],
            'iters': [500, 1000, 2000],
            'df': [2, 4, 8],
            'scale_factor': [2, 4, 8],
            'use_gnll': [False, True]
        }
        
        # Create all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        combinations = list(itertools.product(*param_values))
        print(f"Generated {len(combinations)} total combinations")
        
        return combinations, param_names
    
    def filter_combinations(self, combinations, param_names):
        """Filter combinations based on constraints and user preferences."""
        
        filtered = []
        for combo in combinations:
            # Convert to dict for easier access
            params = dict(zip(param_names, combo))
            
            # Apply filters
            if self.should_skip_combination(params):
                continue
                
            filtered.append(params)
        
        print(f"Filtered to {len(filtered)} valid combinations")
        return filtered
    
    def should_skip_combination(self, params):
        """Define rules for which combinations to skip."""
        
        # Skip certain expensive combinations for faster sweeps
        if params['network_hidden_dim'] == 512 and params['iters'] >= 2000:
            return True
            
        # Skip GNLL with certain models if not compatible
        if params['use_gnll'] and params['model'] in ['wire']:
            return True
            
        # Skip very deep networks with high hidden dims (memory issues)
        if params['network_depth'] >= 8 and params['network_hidden_dim'] >= 512:
            return True
            
        return False
    
    def create_output_dir(self, params, run_id):
        """Create output directory for this configuration."""
        
        # Create descriptive name
        name_parts = [
            f"model_{params['model']}",
            f"depth_{params['network_depth']}",
            f"hidden_{params['network_hidden_dim']}",
            f"lr_{params['learning_rate']}",
            f"iters_{params['iters']}",
            f"df_{params['df']}"
        ]
        
        if params['use_gnll']:
            name_parts.append("gnll")
            
        config_name = "_".join(name_parts)
        output_dir = Path(self.base_args.output_base) / f"run_{run_id:03d}_{config_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def build_command(self, params, output_dir):
        """Build the optimization command for this configuration."""
        
        cmd = [
            sys.executable, "optimize.py",
            "--dataset", self.base_args.dataset,
            "--sample_id", str(self.base_args.sample_id),
            "--df", str(params['df']),
            "--scale_factor", str(params['scale_factor']),
            "--model", params['model'],
            "--network_depth", str(params['network_depth']),
            "--network_hidden_dim", str(params['network_hidden_dim']),
            "--projection_dim", str(params['projection_dim']),
            "--input_projection", params['input_projection'],
            "--learning_rate", str(params['learning_rate']),
            "--weight_decay", str(params['weight_decay']),
            "--iters", str(params['iters']),
            "--device", str(self.base_args.device),
            "--output_folder", str(output_dir)
        ]
        
        # Add optional flags
        if params['use_gnll']:
            cmd.append("--use_gnll")
            
        if self.base_args.multi_sample:
            cmd.append("--multi_sample")
            
        return cmd
    
    def run_single_optimization(self, params, run_id):
        """Run optimization for a single configuration."""
        
        print(f"\n{'='*80}")
        print(f"🚀 RUN {run_id}: {params['model']} | depth={params['network_depth']} | hidden={params['network_hidden_dim']} | lr={params['learning_rate']}")
        print(f"{'='*80}")
        
        # Create output directory
        output_dir = self.create_output_dir(params, run_id)
        
        # Build command
        cmd = self.build_command(params, output_dir)
        
        # Run optimization
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            # Parse results
            success = True
            error_msg = None
            final_loss = self.extract_final_loss(result.stdout)
            final_psnr = self.extract_final_psnr(result.stdout)
            
        except subprocess.TimeoutExpired:
            success = False
            error_msg = "Timeout (1 hour)"
            final_loss = None
            final_psnr = None
        except subprocess.CalledProcessError as e:
            success = False
            error_msg = f"Exit code {e.returncode}: {e.stderr[-200:]}"  # Last 200 chars
            final_loss = None
            final_psnr = None
        except Exception as e:
            success = False
            error_msg = str(e)
            final_loss = None
            final_psnr = None
            
        end_time = time.time()
        duration = end_time - start_time
        
        # Store results
        result_data = {
            'run_id': run_id,
            'params': params,
            'output_dir': str(output_dir),
            'success': success,
            'duration': duration,
            'final_loss': final_loss,
            'final_psnr': final_psnr,
            'error_msg': error_msg,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result_data)
        
        # Print summary
        if success:
            print(f"✅ SUCCESS: Loss={final_loss:.6f}, PSNR={final_psnr:.2f}dB, Duration={duration:.1f}s")
        else:
            print(f"❌ FAILED: {error_msg}")
            
        return result_data
    
    def extract_final_loss(self, stdout):
        """Extract final loss from optimization output."""
        lines = stdout.split('\n')
        for line in reversed(lines):
            if 'Final test loss:' in line:
                try:
                    return float(line.split('Final test loss:')[1].split()[0])
                except:
                    pass
        return None
    
    def extract_final_psnr(self, stdout):
        """Extract final PSNR from optimization output."""
        lines = stdout.split('\n')
        for line in reversed(lines):
            if 'Final PSNR:' in line:
                try:
                    return float(line.split('Final PSNR:')[1].split()[0])
                except:
                    pass
        return None
    
    def save_results(self):
        """Save all results to JSON file."""
        
        results_file = Path(self.base_args.output_base) / f"sweep_results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_runs': len(self.results),
                'successful_runs': sum(1 for r in self.results if r['success']),
                'base_args': vars(self.base_args),
                'results': self.results
            }, f, indent=2)
            
        print(f"\n💾 Results saved to: {results_file}")
        return results_file
    
    def print_summary(self):
        """Print summary of all runs."""
        
        successful_runs = [r for r in self.results if r['success']]
        failed_runs = [r for r in self.results if not r['success']]
        
        print(f"\n{'='*80}")
        print(f"📊 SWEEP SUMMARY")
        print(f"{'='*80}")
        print(f"Total runs: {len(self.results)}")
        print(f"Successful: {len(successful_runs)}")
        print(f"Failed: {len(failed_runs)}")
        
        if successful_runs:
            print(f"\n🏆 TOP 5 RESULTS (by PSNR):")
            top_results = sorted(successful_runs, key=lambda x: x['final_psnr'] or 0, reverse=True)[:5]
            
            for i, result in enumerate(top_results, 1):
                params = result['params']
                print(f"{i}. PSNR: {result['final_psnr']:.2f}dB | "
                      f"Model: {params['model']} | "
                      f"Depth: {params['network_depth']} | "
                      f"Hidden: {params['network_hidden_dim']} | "
                      f"LR: {params['learning_rate']}")
        
        if failed_runs:
            print(f"\n❌ FAILED RUNS:")
            for result in failed_runs[:5]:  # Show first 5 failures
                params = result['params']
                print(f"- Run {result['run_id']}: {result['error_msg'][:100]}...")
    
    def run_sweep(self, max_runs=None):
        """Run the complete optimization sweep."""
        
        print(f"🎯 Starting Optimization Sweep")
        print(f"Dataset: {self.base_args.dataset}")
        print(f"Sample ID: {self.base_args.sample_id}")
        print(f"Multi-sample: {self.base_args.multi_sample}")
        print(f"Output base: {self.base_args.output_base}")
        
        # Generate combinations
        combinations, param_names = self.get_all_combinations()
        filtered_combinations = self.filter_combinations(combinations, param_names)
        
        # Limit number of runs if specified
        if max_runs:
            filtered_combinations = filtered_combinations[:max_runs]
            print(f"Limited to {max_runs} runs")
        
        # Run each configuration
        for run_id, params in enumerate(filtered_combinations, 1):
            print(f"\n🔄 Progress: {run_id}/{len(filtered_combinations)}")
            self.run_single_optimization(params, run_id)
            
            # Save intermediate results every 10 runs
            if run_id % 10 == 0:
                self.save_results()
        
        # Final summary
        self.save_results()
        self.print_summary()

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive optimization sweep")
    
    # Required arguments
    parser.add_argument("--dataset", required=True, choices=['satburst_synth', 'burst_synth', 'worldstrat', 'worldstrat_test'],
                       help="Dataset to use")
    parser.add_argument("--sample_id", type=int, default=0, help="Sample ID (ignored if multi_sample)")
    parser.add_argument("--output_base", required=True, help="Base output directory")
    
    # Optional arguments
    parser.add_argument("--multi_sample", action="store_true", help="Use multi-sample optimization")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--max_runs", type=int, help="Maximum number of runs (for testing)")
    
    args = parser.parse_args()
    
    # Create output base directory
    Path(args.output_base).mkdir(parents=True, exist_ok=True)
    
    # Run sweep
    sweep = OptimizationSweep(args)
    sweep.run_sweep(max_runs=args.max_runs)

if __name__ == "__main__":
    main()
