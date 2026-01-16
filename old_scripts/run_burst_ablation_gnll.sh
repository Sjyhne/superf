#!/bin/bash
# Simple bash script to run Python commands in sequence
# Usage: ./run_sequence.sh

echo "============================================================================"
echo "RUNNING PYTHON COMMANDS IN SEQUENCE"
echo "============================================================================"

# Example Python commands - modify these as needed
echo "Running burst_synth for INR"

python optimize.py --dataset burst_synth --multi_sample --output_folder new_uncertainty_burst_2k_3e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 2 --aug light --iters 2000 --learning_rate 3e-3 --use_gnll --fourier_scale 3
python optimize.py --dataset burst_synth --multi_sample --output_folder new_uncertainty_burst_2k_2e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 2 --aug light --iters 2000 --learning_rate 2e-3 --use_gnll --fourier_scale 3
python optimize.py --dataset burst_synth --multi_sample --output_folder new_uncertainty_burst_2k_1e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 2 --aug light --iters 2000 --learning_rate 1e-3 --use_gnll --fourier_scale 3
python optimize.py --dataset burst_synth --multi_sample --output_folder new_uncertainty_burst_5k_3e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 2 --aug light --iters 5000 --learning_rate 3e-3 --use_gnll --fourier_scale 3
python optimize.py --dataset burst_synth --multi_sample --output_folder new_uncertainty_burst_5k_2e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 2 --aug light --iters 5000 --learning_rate 2e-3 --use_gnll --fourier_scale 3
python optimize.py --dataset burst_synth --multi_sample --output_folder new_uncertainty_burst_5k_1e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 2 --aug light --iters 5000 --learning_rate 1e-3 --use_gnll --fourier_scale 3

echo "============================================================================"
echo "ALL COMMANDS COMPLETED!"
echo "============================================================================"
