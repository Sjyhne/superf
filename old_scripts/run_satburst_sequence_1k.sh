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

python optimize.py --dataset satburst_synth --multi_sample --output_folder satburst_df2_nir_2k --df 2 --scale_factor 1 --fourier_scale 10 --device 5 --iters 2000 --aug light --no_base_frame --no_direct_param_T --color_shift
python optimize.py --dataset satburst_synth --multi_sample --output_folder satburst_df4_nir_2k --df 4 --scale_factor 1 --fourier_scale 10 --device 5 --iters 2000 --aug light --no_base_frame --no_direct_param_T --color_shift
python optimize.py --dataset satburst_synth --multi_sample --output_folder satburst_df8_nir_2k --df 8 --scale_factor 1 --fourier_scale 10 --device 5 --iters 2000 --aug light --no_base_frame --no_direct_param_T --color_shift


echo "============================================================================"
echo "ALL COMMANDS COMPLETED!"
echo "============================================================================"
