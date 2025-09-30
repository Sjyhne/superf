#!/bin/bash
# Updated runner script for Fourier scale exploration using current optimize.py
# Usage: ./run_fourier_exploration_updated.sh [output_root] [gpu_indices]
# Example: ./run_fourier_exploration_updated.sh results "0 1 2 3"

OUTPUT_ROOT=${1:-"fourier_exploration_newest"}
GPU_INDICES=${2:-"1 2 3 4"}

DATASETS=("burst_synth" "satburst_synth")
FOURIER_SCALES=(1 3 5 10 20)
DFS=(2 4 8)

# Convert GPU indices to array
read -ra GPU_ARRAY <<< "$GPU_INDICES"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================================"
echo "FOURIER SCALE EXPLORATION (UPDATED) - MULTI-GPU"
echo "============================================================================"
echo "Output root: $OUTPUT_ROOT"
echo "Available GPUs: ${GPU_ARRAY[*]} (${NUM_GPUS} total)"
echo "Testing Fourier scales: ${FOURIER_SCALES[*]} across DF: ${DFS[*]}"
echo "Running for datasets: ${DATASETS[*]} with losses: MSE and GNLL"
echo "Using current optimize.py with --multi_sample flag"
echo "============================================================================"

mkdir -p "$OUTPUT_ROOT"

# Create array of all experiment configurations
declare -a EXPERIMENTS=()
for DATASET in "${DATASETS[@]}"; do
  for LOSS in mse gnll; do
    for DF in "${DFS[@]}"; do
      for FS in "${FOURIER_SCALES[@]}"; do
        EXPERIMENTS+=("${DATASET}_${LOSS}_df${DF}_fs${FS}")
      done
    done
  done
done

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
echo "Total experiments to run: $TOTAL_EXPERIMENTS"
echo "Will distribute across ${NUM_GPUS} GPUs"
echo

# Function to run experiment on specific GPU
run_experiment() {
  local exp_config=$1
  local gpu_id=$2
  
  # Parse experiment configuration - handle underscore in dataset names
  if [[ "$exp_config" == burst_synth_* ]]; then
    local DATASET="burst_synth"
    local LOSS="${exp_config#burst_synth_}"
    local LOSS="${LOSS%%_df*}"
    local DF="${exp_config#*_df}"
    local DF="${DF%%_fs*}"
    local FS="${exp_config#*_fs}"
  elif [[ "$exp_config" == satburst_synth_* ]]; then
    local DATASET="satburst_synth"
    local LOSS="${exp_config#satburst_synth_}"
    local LOSS="${LOSS%%_df*}"
    local DF="${exp_config#*_df}"
    local DF="${DF%%_fs*}"
    local FS="${exp_config#*_fs}"
  else
    echo "Error: Unknown experiment format: $exp_config"
    return 1
  fi
  
  local SUBFOLDER="${OUTPUT_ROOT}/${exp_config}"
  echo "[GPU ${gpu_id}] Running dataset=${DATASET}, loss=${LOSS}, df=${DF}, fs=${FS} -> ${SUBFOLDER}"

  CMD=(
    python optimize.py
      --dataset "$DATASET"
      --multi_sample
      --output_folder "$SUBFOLDER"
      --df "$DF"
      --scale_factor "$DF"
      --fourier_scale "$FS"
      --model mlp
      --network_depth 4
      --network_hidden_dim 256
      --projection_dim 256
      --iters 2000
      --learning_rate 2e-3
      --weight_decay 0.05
      --device "$gpu_id"
  )

  # Append GNLL flag only when requested
  if [[ "$LOSS" == "gnll" ]]; then
    CMD+=(--use_gnll)
  fi

  # Run experiment
  "${CMD[@]}"
  
  if [ $? -eq 0 ]; then
    echo "[GPU ${gpu_id}] ✅ Completed: ${exp_config}"
  else
    echo "[GPU ${gpu_id}] ❌ Failed: ${exp_config}"
  fi
}

# Distribute experiments across GPUs
declare -a PIDS=()
for i in "${!EXPERIMENTS[@]}"; do
  exp="${EXPERIMENTS[$i]}"
  gpu_idx=$((i % NUM_GPUS))
  gpu_id="${GPU_ARRAY[$gpu_idx]}"
  
  # Run experiment in background
  run_experiment "$exp" "$gpu_id" &
  PIDS+=($!)
  
  # Limit concurrent processes to number of GPUs
  if [ ${#PIDS[@]} -ge $NUM_GPUS ]; then
    # Wait for any process to complete
    wait -n
    # Remove completed PIDs
    for j in "${!PIDS[@]}"; do
      if ! kill -0 "${PIDS[$j]}" 2>/dev/null; then
        unset PIDS[$j]
      fi
    done
  fi
done

# Wait for all remaining processes to complete
echo "Waiting for all experiments to complete..."
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

# Count successful experiments
SUCCESSFUL=0
FAILED=0
for exp in "${EXPERIMENTS[@]}"; do
  if [ -f "${OUTPUT_ROOT}/${exp}/summary_statistics.json" ]; then
    ((SUCCESSFUL++))
  else
    ((FAILED++))
  fi
done

echo "============================================================================"
echo "EXPLORATION COMPLETE!"
echo "============================================================================"
echo "Results Summary:"
echo "  ✅ Successful: $SUCCESSFUL/$TOTAL_EXPERIMENTS"
echo "  ❌ Failed: $FAILED/$TOTAL_EXPERIMENTS"
echo "  📁 Root folder: $OUTPUT_ROOT"
echo ""
echo "Each successful experiment includes:"
echo "  - Aggregated statistics (mean, std, min, max)"
echo "  - Individual sample results with color-aligned comparisons"
echo "  - Summary visualizations across all samples"
echo ""
echo "To analyze results, run:"
echo "  python create_fourier_plots.py --results_folder $OUTPUT_ROOT"
echo "============================================================================"
