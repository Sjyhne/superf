#!/bin/bash
# Simple runner script for Fourier scale exploration
# Usage: ./run_fourier_exploration.sh [output_root]

OUTPUT_ROOT=${1:-"fourier_exploration_$(date +%Y%m%d_%H%M%S)"}

DATASETS=("satburst_synth" "burst_synth")
FOURIER_SCALES=(1 3 5 10 20)
DFS=(2 4 8)

echo "============================================================================"
echo "FOURIER SCALE EXPLORATION"
echo "============================================================================"
echo "Output root: $OUTPUT_ROOT"
echo "Testing Fourier scales: ${FOURIER_SCALES[*]} across DF: ${DFS[*]}"
echo "Running for datasets: ${DATASETS[*]} with losses: MSE and GNLL"
echo "============================================================================"

mkdir -p "$OUTPUT_ROOT"

for DATASET in "${DATASETS[@]}"; do
  for LOSS in mse gnll; do
    # Subfolder indicates dataset and loss type
    SUBFOLDER="${OUTPUT_ROOT}/${DATASET}_${LOSS}"
    echo "--- Running dataset=${DATASET}, loss=${LOSS} -> ${SUBFOLDER} ---"

    CMD=(
      python explore_fourier_scales.py
        --output_folder "$SUBFOLDER"
        --dataset "$DATASET"
        --fourier_scales "${FOURIER_SCALES[@]}"
        --dfs "${DFS[@]}"
        --model mlp
        --network_depth 4
        --network_hidden_dim 256
        --projection_dim 256
        --iters 2000
        --learning_rate 2e-3
        --weight_decay 0.05
        --device 7
    )

    # Append GNLL flag only when requested
    if [[ "$LOSS" == "gnll" ]]; then
      CMD+=(--use_gnll)
    fi

    # Run
    "${CMD[@]}"

    echo "Saved results to: ${SUBFOLDER}"
    echo "  - ${SUBFOLDER}/analysis_plots/ (visualizations)"
    echo "  - ${SUBFOLDER}/exploration_summary_report.txt (detailed report)"
    echo "  - ${SUBFOLDER}/fourier_exploration_results_*.csv (raw data)"
    echo
  done
done

echo "============================================================================"
echo "EXPLORATION COMPLETE!"
echo "Root folder: $OUTPUT_ROOT"
echo "============================================================================"
