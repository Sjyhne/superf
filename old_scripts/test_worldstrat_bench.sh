#!/bin/bash
# Quick test script to verify experiments can run before full benchmark
# Usage: ./test_worldstrat_bench.sh
# Results will be saved in: worldstrat_test_results/

echo "============================================================================"
echo "TEST RUN FOR WORLDSTRAT BENCHMARK"
echo "============================================================================"
echo "This script runs a quick test with minimal parameters to verify everything works."
echo ""

# Base output directory for test results
BASE_OUTPUT_DIR="worldstrat_test_results"

# Minimal hyperparameter values for quick testing
LEARNING_RATES=(2e-3)
FOURIER_SCALES=(5)
ITERS=(100)  # Reduced iterations for quick test

# Model variants to run (test all three)
USE_MSE=true
USE_REGULAR_GNLL=true
USE_SEPARATE_UD=true

# GPU assignments: same as main benchmark
DEVICE_MSE=3
DEVICE_GNLL=1
DEVICE_SEPARATE_UD=2

# Common parameters
DF=4
SCALE_FACTOR=4
NUM_SAMPLES=8

# Test only one dataset to speed things up
TEST_DATASET="worldstrat_bitter"  # Change to "worldstrat_sweet" if preferred

# Function to create output folder name with normalized format for easy sorting
create_output_folder() {
    local dataset=$1
    local loss_type=$2
    local lr=$3
    local fs=$4
    local iter=$5
    
    # Normalize learning rate format (replace e with E and remove dots for consistency)
    local lr_str=$(echo "$lr" | sed 's/e-/e-/g' | sed 's/\.0//g')
    # Format: dataset_loss_lr{lr}_fs{fs}_iter{iter}
    echo "${BASE_OUTPUT_DIR}/${dataset}_${loss_type}_lr${lr_str}_fs${fs}_iter${iter}"
}

# Function to run experiment
run_experiment() {
    local dataset=$1
    local output_folder=$2
    local learning_rate=$3
    local fourier_scale=$4
    local iters=$5
    local device=$6
    local use_gnll=$7
    local use_separate_ud=$8
    local exp_num=$9
    local total_exp=${10}
    
    local gnll_flag=""
    local separate_ud_flag=""
    local loss_type="mse"
    
    # use_gnll and use_separate_ud can be used together
    if [ "$use_separate_ud" = "true" ]; then
        separate_ud_flag="--use_separate_ud"
        loss_type="separate_ud"
    fi
    if [ "$use_gnll" = "true" ]; then
        gnll_flag="--use_gnll"
        if [ "$loss_type" = "mse" ]; then
            loss_type="gnll"
        fi
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$exp_num/$total_exp] Running: dataset=$dataset, loss=$loss_type, lr=$learning_rate, fs=$fourier_scale, iters=$iters, device=$device, separate_ud=$use_separate_ud"
    
    python optimize.py \
        --dataset "$dataset" \
        --multi_sample \
        --output_folder "$output_folder" \
        --df $DF \
        --scale_factor $SCALE_FACTOR \
        --num_samples $NUM_SAMPLES \
        --device $device \
        --iters $iters \
        --learning_rate $learning_rate \
        --fourier_scale $fourier_scale \
        $gnll_flag \
        $separate_ud_flag
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $output_folder"
    else
        echo "❌ Failed: $output_folder"
    fi
    echo ""
}

# Counter for experiments
loss_count=0
[ "$USE_MSE" = "true" ] && loss_count=$((loss_count + 1))
[ "$USE_REGULAR_GNLL" = "true" ] && loss_count=$((loss_count + 1))
[ "$USE_SEPARATE_UD" = "true" ] && loss_count=$((loss_count + 1))
total_experiments=$((${#LEARNING_RATES[@]} * ${#FOURIER_SCALES[@]} * ${#ITERS[@]} * $loss_count))
current=0

echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Test dataset: $TEST_DATASET"
echo "Total test experiments: $total_experiments"
echo ""
echo "GPU Assignment:"
echo "  - GPU $DEVICE_MSE: MSE baseline"
echo "  - GPU $DEVICE_GNLL: Regular GNLL"
echo "  - GPU $DEVICE_SEPARATE_UD: Separate UD"
echo ""
echo "Execution Mode: PARALLEL"
echo "  - All variants (MSE, GNLL, Separate UD) will run simultaneously"
echo "  - Each variant uses its own GPU, so there's no GPU conflict"
echo ""
echo "Press Ctrl+C to cancel..."
sleep 3

# Create base directory
mkdir -p "$BASE_OUTPUT_DIR"

# Run test experiments
echo "============================================================================"
echo "RUNNING TEST EXPERIMENTS"
echo "============================================================================"

for lr in "${LEARNING_RATES[@]}"; do
    for fs in "${FOURIER_SCALES[@]}"; do
        for iter in "${ITERS[@]}"; do
            # Array to store background job PIDs
            pids=()
            
            # MSE variant (GPU 3) - run in background
            if [ "$USE_MSE" = "true" ]; then
                current=$((current + 1))
                output_mse=$(create_output_folder "$TEST_DATASET" "mse" "$lr" "$fs" "$iter")
                run_experiment "$TEST_DATASET" "$output_mse" "$lr" "$fs" "$iter" "$DEVICE_MSE" "false" "false" "$current" "$total_experiments" &
                pids+=($!)
            fi
            
            # Regular GNLL variant (GPU 1) - run in background
            if [ "$USE_REGULAR_GNLL" = "true" ]; then
                current=$((current + 1))
                output_gnll=$(create_output_folder "$TEST_DATASET" "gnll" "$lr" "$fs" "$iter")
                run_experiment "$TEST_DATASET" "$output_gnll" "$lr" "$fs" "$iter" "$DEVICE_GNLL" "true" "false" "$current" "$total_experiments" &
                pids+=($!)
            fi
            
            # Separate UD variant (GPU 2) - run in background
            if [ "$USE_SEPARATE_UD" = "true" ]; then
                current=$((current + 1))
                output_separate_ud=$(create_output_folder "$TEST_DATASET" "separate_ud" "$lr" "$fs" "$iter")
                run_experiment "$TEST_DATASET" "$output_separate_ud" "$lr" "$fs" "$iter" "$DEVICE_SEPARATE_UD" "true" "true" "$current" "$total_experiments" &
                pids+=($!)
            fi
            
            # Wait for all background jobs for this hyperparameter combination to complete
            echo "Waiting for ${#pids[@]} parallel experiments to complete (lr=$lr, fs=$fs, iter=$iter)..."
            for pid in "${pids[@]}"; do
                wait $pid
                if [ $? -eq 0 ]; then
                    echo "✅ Background job $pid completed successfully"
                else
                    echo "❌ Background job $pid failed"
                fi
            done
            echo ""
        done
    done
done

echo "============================================================================"
echo "TEST EXPERIMENTS COMPLETED!"
echo "============================================================================"
echo "Results saved in: $BASE_OUTPUT_DIR/"
echo ""
echo "Summary:"
echo "  - Total test experiments completed: $current"
echo "  - MSE experiments: GPU $DEVICE_MSE"
echo "  - GNLL experiments: GPU $DEVICE_GNLL"
echo "  - Separate UD experiments: GPU $DEVICE_SEPARATE_UD"
echo ""
echo "If all experiments completed successfully, you can now run the full benchmark:"
echo "  ./run_worldstrat_bench.sh"
echo ""
echo "To check test results:"
echo "  1. Check summary_statistics.json in each experiment folder"
echo "  2. List all test experiments: ls -d $BASE_OUTPUT_DIR/*"
echo "============================================================================"

