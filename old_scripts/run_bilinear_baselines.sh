#!/bin/bash
# Compute bilinear baselines for worldstrat_sweet and worldstrat_bitter datasets
# Usage: ./run_bilinear_baselines.sh

echo "============================================================================"
echo "COMPUTING BILINEAR BASELINES FOR WORLDSTRAT DATASETS"
echo "============================================================================"

# Parameters
DF=4
SCALE_FACTOR=4
DEVICE="cuda:0"
SAVE_IMAGES=false  # Set to true to save comparison images for each sample

# Output directories
SWEET_OUTPUT="bilinear_baselines/worldstrat_sweet_df${DF}_sf${SCALE_FACTOR}"
BITTER_OUTPUT="bilinear_baselines/worldstrat_bitter_df${DF}_sf${SCALE_FACTOR}"

echo ""
echo "Computing bilinear baseline for worldstrat_sweet..."
echo "Output: ${SWEET_OUTPUT}"
python3 compute_bilinear_baselines.py \
    --dataset worldstrat_sweet \
    --df ${DF} \
    --scale_factor ${SCALE_FACTOR} \
    --device ${DEVICE} \
    --output_folder ${SWEET_OUTPUT} \
    $([ "$SAVE_IMAGES" = "true" ] && echo "--save_images")

echo ""
echo "Computing bilinear baseline for worldstrat_bitter..."
echo "Output: ${BITTER_OUTPUT}"
python3 compute_bilinear_baselines.py \
    --dataset worldstrat_bitter \
    --df ${DF} \
    --scale_factor ${SCALE_FACTOR} \
    --device ${DEVICE} \
    --output_folder ${BITTER_OUTPUT} \
    $([ "$SAVE_IMAGES" = "true" ] && echo "--save_images")

echo ""
echo "============================================================================"
echo "BILINEAR BASELINES COMPUTED SUCCESSFULLY"
echo "============================================================================"
echo ""
echo "Results saved to:"
echo "  - ${SWEET_OUTPUT}/bilinear_baseline_results.json"
echo "  - ${SWEET_OUTPUT}/bilinear_baseline_report.txt"
echo "  - ${BITTER_OUTPUT}/bilinear_baseline_results.json"
echo "  - ${BITTER_OUTPUT}/bilinear_baseline_report.txt"

