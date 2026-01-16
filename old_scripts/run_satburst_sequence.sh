#!/bin/bash
# Simple bash script to run Python commands in sequence
# Usage: ./run_sequence.sh

echo "============================================================================"
echo "RUNNING PYTHON COMMANDS IN SEQUENCE"
echo "============================================================================"

# Example Python commands - modify these as needed
echo "Running burst_synth for INR"
# python optimize.py --dataset satburst_synth --multi_sample --output_folder satburst_df2_inr --df 2 --scale_factor 2 --fourier_scale 10 --device 6 --aug light
# python optimize.py --dataset satburst_synth --multi_sample --output_folder satburst_df2_inr_gnll --df 2 --scale_factor 2 --fourier_scale 10 --device 6 --use_gnll --aug light
# python optimize.py --dataset satburst_synth --multi_sample --output_folder satburst_df4_inr --df 4 --scale_factor 4 --fourier_scale 10 --device 6 --aug light
# python optimize.py --dataset satburst_synth --multi_sample --output_folder satburst_df4_inr_gnll --df 4 --scale_factor 4 --fourier_scale 10 --device 6 --use_gnll --aug light
# python optimize.py --dataset satburst_synth --multi_sample --output_folder satburst_df8_inr --df 8 --scale_factor 8 --fourier_scale 10 --device 6 --aug light
# python optimize.py --dataset satburst_synth --multi_sample --output_folder satburst_df8_inr_gnll --df 8 --scale_factor 8 --fourier_scale 10 --device 6 --use_gnll --aug light


# python optimize.py --dataset satburst_synth --multi_sample --output_folder 1_nir_auglight --df 4 --scale_factor 1 --fourier_scale 10 --device 4 --iters 2000 --aug light --no_base_frame --no_direct_param_T --use_color_shift
# python optimize.py --dataset satburst_synth --multi_sample --output_folder 2_inr_direct_T_auglight --df 4 --scale_factor 1 --fourier_scale 10 --device 4 --iters 2000 --aug light --no_base_frame --use_color_shift
# python optimize.py --dataset satburst_synth --multi_sample --output_folder 3_inr_super_sampling_auglight --df 4 --scale_factor 4 --fourier_scale 10 --device 4 --iters 2000 --aug light --no_base_frame --no_direct_param_T --use_color_shift
# python optimize.py --dataset satburst_synth --multi_sample --output_folder 4_inr_fixed_base_frame_auglight --df 4 --scale_factor 1 --fourier_scale 10 --device 4 --iters 2000 --aug light --no_direct_param_T --use_color_shift
# python optimize.py --dataset satburst_synth --multi_sample --output_folder 5_inr_super_sampling_fixed_base_frame_auglight --df 4 --scale_factor 4 --fourier_scale 10 --device 4 --iters 2000 --aug light --no_direct_param_T --use_color_shift
# python optimize.py --dataset satburst_synth --multi_sample --output_folder 6_inr_direct_T_fixed_base_frame_auglight --df 4 --scale_factor 1 --fourier_scale 10 --device 4 --iters 2000 --aug light --use_color_shift
# python optimize.py --dataset satburst_synth --multi_sample --output_folder 7_inr_direct_T_super_sampling_auglight --df 4 --scale_factor 4 --fourier_scale 10 --device 4 --iters 2000 --aug light --no_base_frame --use_color_shift
# python optimize.py --dataset satburst_synth --multi_sample --output_folder 8_inr_fixed_base_frame_direct_T_super_sampling_auglight --df 4 --scale_factor 4 --fourier_scale 10 --device 4 --iters 2000 --aug light --use_color_shift

# python optimize.py --dataset satburst_synth --multi_sample --output_folder 1_table2_satburst_no_FF_no_MF_no_A --df 4 --scale_factor 4 --num_samples 1 --input_projection none --device 4 --projection_dim 2 --aug light
# python optimize.py --dataset satburst_synth --multi_sample --output_folder 2_table2_satburst_yes_FF_no_MF_no_A --df 4 --scale_factor 4 --num_samples 1 --device 4 --aug light
# python optimize.py --dataset satburst_synth --multi_sample --output_folder 3_table2_satburst_yes_FF_yes_MF_no_A --df 4 --scale_factor 4 --num_samples 16 --device 4 --aug light


#python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_2k_2e-3_direct_gnll --df 4 --scale_factor 4 --num_samples 16 --device 2 --aug light --iters 2000 --learning_rate 2e-3 --use_direct_gnll --fourier_scale 10
#python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_2k_2e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 2 --aug light --iters 2000 --learning_rate 2e-3 --use_gnll --fourier_scale 10

python optimize_with_fs_hr.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_satburst_2k_2e-3_mse --df 4 --scale_factor 4 --num_samples 16 --device 2 --aug light --iters 2000 --learning_rate 2e-3 --fourier_scale 5 --optimize_with_hr


# python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_2k_2e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 6 --aug light --iters 2000 --learning_rate 2e-3 --use_gnll
# python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_2k_1e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 6 --aug light --iters 2000 --learning_rate 1e-3 --use_gnll
# python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_5k_3e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 6 --aug light --iters 5000 --learning_rate 3e-3 --use_gnll
# python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_5k_2e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 6 --aug light --iters 5000 --learning_rate 2e-3 --use_gnll
# python optimize.py --dataset satburst_synth --multi_sample --output_folder new_uncertainty_5k_1e-3_gnll --df 4 --scale_factor 4 --num_samples 16 --device 6 --aug light --iters 5000 --learning_rate 1e-3 --use_gnll


echo "============================================================================"
echo "ALL COMMANDS COMPLETED!"
echo "============================================================================"
