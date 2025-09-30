#!/bin/bash
# Simple bash script to run Python commands in sequence
# Usage: ./run_sequence.sh

echo "============================================================================"
echo "RUNNING PYTHON COMMANDS IN SEQUENCE"
echo "============================================================================"

# Example Python commands - modify these as needed
echo "Running burst_synth for INR"
# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df2_inr --df 2 --scale_factor 2 --fourier_scale 3 --device 7
# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df2_inr_gnll --df 2 --scale_factor 2 --fourier_scale 3 --device 7 --use_gnll
# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df4_inr --df 4 --scale_factor 4 --fourier_scale 3 --device 7
# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df4_inr_gnll --df 4 --scale_factor 4 --fourier_scale 3 --device 7 --use_gnll
# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df8_inr --df 8 --scale_factor 8 --fourier_scale 3 --device 7
# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df8_inr_gnll --df 8 --scale_factor 8 --fourier_scale 3 --device 7 --use_gnll

python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df2_nir_2k --df 2 --scale_factor 1 --fourier_scale 3 --device 7 --iters 2000 --no_base_frame --no_direct_param_T
python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df2_nir_5k --df 2 --scale_factor 1 --fourier_scale 3 --device 7 --iters 5000 --no_base_frame --no_direct_param_T
python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df4_nir_2k --df 4 --scale_factor 1 --fourier_scale 3 --device 7 --iters 2000 --no_base_frame --no_direct_param_T
python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df4_nir_5k --df 4 --scale_factor 1 --fourier_scale 3 --device 7 --iters 5000 --no_base_frame --no_direct_param_T
python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df8_nir_2k --df 8 --scale_factor 1 --fourier_scale 3 --device 7 --iters 2000 --no_base_frame --no_direct_param_T
python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df8_nir_5k --df 8 --scale_factor 1 --fourier_scale 3 --device 7 --iters 5000 --no_base_frame --no_direct_param_T


echo "============================================================================"
echo "ALL COMMANDS COMPLETED!"
echo "============================================================================"
