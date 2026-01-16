#!/bin/bash

# Example script showing how to use the new --visualize_variance flag
# This will generate variance visualizations for each LR sample when using GNLL

echo "Running optimization with variance visualization for burst_synth dataset..."

# Example 1: Single sample with variance visualization
python optimize.py \
    --dataset worldstrat_test \
    --sample_id "Amnesty POI-17-2-3" \
    --df 4 \
    --use_gnll \
    --visualize_variance \
    --model mlp \
    --network_depth 4 \
    --network_hidden_dim 256 \
    --input_projection fourier_10 \
    --iters 2000 \
    --learning_rate 2e-3 \
    --device 0

echo "Variance visualizations will be saved in:"
echo "  single_samples/burst_synth/0/variance_visualizations/"
echo ""
echo "This directory will contain:"
echo "  - sample_XXX_variance_analysis.png: Individual variance analysis for each LR sample"
echo "  - sample_XXX_variance.npy: Raw variance maps as numpy arrays"
echo "  - sample_XXX_output.npy: Model outputs for each LR sample"
echo "  - variance_summary.txt: Summary of the variance analysis"
