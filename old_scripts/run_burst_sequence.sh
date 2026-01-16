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

# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df2_nir_2k --df 2 --scale_factor 1 --fourier_scale 3 --device 7 --iters 2000 --no_base_frame --no_direct_param_T
# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df2_nir_5k --df 2 --scale_factor 1 --fourier_scale 3 --device 7 --iters 5000 --no_base_frame --no_direct_param_T
# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df4_nir_2k --df 4 --scale_factor 1 --fourier_scale 3 --device 7 --iters 2000 --no_base_frame --no_direct_param_T
# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df4_nir_5k --df 4 --scale_factor 1 --fourier_scale 3 --device 7 --iters 5000 --no_base_frame --no_direct_param_T
# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df8_nir_2k --df 8 --scale_factor 1 --fourier_scale 3 --device 7 --iters 2000 --no_base_frame --no_direct_param_T
# python optimize.py --dataset burst_synth --multi_sample --output_folder burst_df8_nir_5k --df 8 --scale_factor 1 --fourier_scale 3 --device 7 --iters 5000 --no_base_frame --no_direct_param_T

#python optimize.py --dataset burst_synth --multi_sample --output_folder 1_table2_burst_no_FF_no_MF_no_A --df 4 --scale_factor 4 --num_samples 1 --input_projection none --device 5 --projection_dim 2 --aug light --fourier_scale 3
#python optimize.py --dataset burst_synth --multi_sample --output_folder 2_table2_burst_yes_FF_no_MF_no_A --df 4 --scale_factor 4 --num_samples 1 --device 5 --aug light --fourier_scale 3
#python optimize.py --dataset burst_synth --multi_sample --output_folder 3_table2_burst_yes_FF_yes_MF_no_A --df 4 --scale_factor 4 --num_samples 16 --device 5 --aug light --fourier_scale 3

python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_2k_3e-3_mse --df 4 --scale_factor 4 --num_samples 16 --device 7 --aug light --iters 2000 --learning_rate 3e-3
python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_2k_2e-3_mse --df 4 --scale_factor 4 --num_samples 16 --device 7 --aug light --iters 2000 --learning_rate 2e-3
python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_2k_1e-3_mse --df 4 --scale_factor 4 --num_samples 16 --device 7 --aug light --iters 2000 --learning_rate 1e-3
python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_5k_3e-3_mse --df 4 --scale_factor 4 --num_samples 16 --device 7 --aug light --iters 5000 --learning_rate 3e-3
python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_5k_2e-3_mse --df 4 --scale_factor 4 --num_samples 16 --device 7 --aug light --iters 5000 --learning_rate 2e-3
python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_5k_1e-3_mse --df 4 --scale_factor 4 --num_samples 16 --device 7 --aug light --iters 5000 --learning_rate 1e-3

echo "============================================================================"
echo "ALL COMMANDS COMPLETED!"
echo "============================================================================"
