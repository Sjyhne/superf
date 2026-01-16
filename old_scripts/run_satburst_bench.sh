#!/bin/bash
# Hyperparameter sweep for satburst_synth and burst_synth datasets
# Usage: ./run_satburst_bench.sh
# Results will be saved in: satburst_hyperparameter_sweep/

echo "============================================================================"
echo "HYPERPARAMETER SWEEP FOR SATBURST_SYNTH AND BURST_SYNTH DATASETS"
echo "============================================================================"

# Base output directories for each dataset (separate results)
BASE_OUTPUT_DIR_BURST="burst_hyperparameter_sweep"
BASE_OUTPUT_DIR_SATBURST="satburst_hyperparameter_sweep"

# Hyperparameter values to sweep
LEARNING_RATES=(2e-3)
FOURIER_SCALES=(3)
ITERS=(2000)
# DF_SCALES and SCALE_FACTORS must be paired (same index = same pair)
DF_SCALES=(2 4 8)
SCALE_FACTORS=(2 4 8)

# Model variants to run
USE_MSE=true  # Set to true to run MSE baseline
USE_REGULAR_GNLL=true  # Set to true to run regular GNLL
USE_SEPARATE_UD=true  # Set to true to use separate UD parameters for each sample

# GPU assignments: one GPU per model variant
# GPU 0: MSE baseline
# GPU 1: Regular GNLL
# GPU 2: Separate UD
DEVICE_MSE=3
DEVICE_GNLL=1
DEVICE_SEPARATE_UD=2

# Common parameters (used as defaults, but will be overridden by sweep values)
NUM_SAMPLES=16
LR_SHIFT=1.0
AUG="light"
NO_VARIANCE_VIZ=true  # Set to true to skip variance visualizations (faster, saves disk space)

# Initialize arrays (will be populated by discovery functions)
SATBURST_SAMPLE_IDS=()
BURST_SAMPLE_IDS=()

# Function to discover all sample IDs for satburst_synth
discover_satburst_samples() {
    local data_root="data"
    if [ ! -d "$data_root" ]; then
        echo "Warning: data directory '$data_root' not found. No satburst_synth samples will be processed."
        echo ""
        return
    fi
    
    # Find all directories in data/ that don't start with '.'
    local sample_dirs=($(find "$data_root" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -exec basename {} \; | sort))
    
    if [ ${#sample_dirs[@]} -eq 0 ]; then
        echo "Warning: No sample directories found in '$data_root'. No satburst_synth samples will be processed."
        echo ""
        return
    fi
    
    echo "Found ${#sample_dirs[@]} satburst_synth samples: ${sample_dirs[*]:0:5}..."
    SATBURST_SAMPLE_IDS=("${sample_dirs[@]}")
}

# Function to discover all sample IDs for burst_synth
discover_burst_samples() {
    # Check for DATA_DIR_ABSOLUTE environment variable
    if [ -n "$DATA_DIR_ABSOLUTE" ]; then
        local data_root="$DATA_DIR_ABSOLUTE"
    else
        local data_root="SyntheticBurstVal"
    fi
    
    local gt_dir="$data_root/gt"
    if [ ! -d "$gt_dir" ]; then
        echo "Warning: GT directory '$gt_dir' not found. No burst_synth samples will be processed."
        echo ""
        return
    fi
    
    # Find all numeric directories in gt/
    local sample_dirs=($(find "$gt_dir" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -exec basename {} \; | sort -n))
    
    if [ ${#sample_dirs[@]} -eq 0 ]; then
        echo "Warning: No numeric sample directories found in '$gt_dir'. No burst_synth samples will be processed."
        echo ""
        return
    fi
    
    echo "Found ${#sample_dirs[@]} burst_synth samples: ${sample_dirs[*]:0:5}..."
    BURST_SAMPLE_IDS=("${sample_dirs[@]}")
}

# Discover sample IDs automatically
echo "Discovering sample IDs..."
discover_satburst_samples
discover_burst_samples

# Check if we have any samples to process
if [ ${#SATBURST_SAMPLE_IDS[@]} -eq 0 ] && [ ${#BURST_SAMPLE_IDS[@]} -eq 0 ]; then
    echo "❌ Error: No samples found for either dataset!"
    echo "   - satburst_synth: Check that 'data/' directory exists with sample subdirectories"
    echo "   - burst_synth: Check that 'SyntheticBurstVal/gt/' exists (or set DATA_DIR_ABSOLUTE env var)"
    exit 1
fi

# Function to create output folder name with normalized format for easy sorting
create_output_folder() {
    local dataset=$1
    local loss_type=$2
    local lr=$3
    local fs=$4
    local iter=$5
    local df=$6
    local scale_factor=$7
    
    # Choose base directory based on dataset
    local base_dir
    if [ "$dataset" = "burst_synth" ]; then
        base_dir="$BASE_OUTPUT_DIR_BURST"
    elif [ "$dataset" = "satburst_synth" ]; then
        base_dir="$BASE_OUTPUT_DIR_SATBURST"
    else
        base_dir="hyperparameter_sweep"
    fi
    
    # Normalize learning rate format
    local lr_str=$(echo "$lr" | sed 's/e-/e-/g' | sed 's/\.0//g')
    # Format: loss_lr{lr}_fs{fs}_df{df}_sf{scale_factor}_iter{iter} (dataset is in base_dir name)
    # optimize.py will create sample subdirectories inside this folder
    echo "${base_dir}/${loss_type}_lr${lr_str}_fs${fs}_df${df}_sf${scale_factor}_iter${iter}"
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
    local df=${11}
    local scale_factor=${12}
    
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
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$exp_num/$total_exp] Running: dataset=$dataset, loss=$loss_type, lr=$learning_rate, fs=$fourier_scale, df=$df, sf=$scale_factor, iters=$iters, device=$device, separate_ud=$use_separate_ud"
    echo "  Processing ALL samples in $dataset dataset..."
    echo "  Flags: gnll_flag='$gnll_flag', separate_ud_flag='$separate_ud_flag'"
    
    # Build command with dataset-specific arguments
    # Note: --sample_id is ignored when --multi_sample is used, but we pass a dummy value for compatibility
    if [ "$dataset" = "satburst_synth" ]; then
        cmd=(python optimize.py
            --dataset "$dataset"
            --sample_id "dummy"
            --multi_sample
            --output_folder "$output_folder"
            --df $df
            --scale_factor $scale_factor
            --num_samples $NUM_SAMPLES
            --lr_shift $LR_SHIFT
            --aug $AUG
            --device $device
            --iters $iters
            --learning_rate $learning_rate
            --fourier_scale $fourier_scale)
        
        # Add flags only if they are set
        [ -n "$gnll_flag" ] && cmd+=("$gnll_flag")
        [ -n "$separate_ud_flag" ] && cmd+=("$separate_ud_flag")
        [ "$NO_VARIANCE_VIZ" = "true" ] && cmd+=("--no_variance_viz")
        
        "${cmd[@]}"
    elif [ "$dataset" = "burst_synth" ]; then
        cmd=(python optimize.py
            --dataset "$dataset"
            --sample_id "0"
            --multi_sample
            --output_folder "$output_folder"
            --df $df
            --scale_factor $scale_factor
            --num_samples $NUM_SAMPLES
            --device $device
            --iters $iters
            --learning_rate $learning_rate
            --fourier_scale $fourier_scale)
        
        # Add flags only if they are set
        [ -n "$gnll_flag" ] && cmd+=("$gnll_flag")
        [ -n "$separate_ud_flag" ] && cmd+=("$separate_ud_flag")
        [ "$NO_VARIANCE_VIZ" = "true" ] && cmd+=("--no_variance_viz")
        
        "${cmd[@]}"
    else
        echo "❌ Unknown dataset: $dataset"
        return 1
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $output_folder"
    else
        echo "❌ Failed: $output_folder"
    fi
    echo ""
}

# Counter for experiments
# Calculate total based on which loss types are enabled
loss_count=0
[ "$USE_MSE" = "true" ] && loss_count=$((loss_count + 1))
[ "$USE_REGULAR_GNLL" = "true" ] && loss_count=$((loss_count + 1))
[ "$USE_SEPARATE_UD" = "true" ] && loss_count=$((loss_count + 1))

# Calculate total experiments: datasets * learning_rates * fourier_scales * df_scale_pairs * iters * loss_types
# Note: Each experiment processes ALL samples in the dataset (via --multi_sample)
# DF_SCALES and SCALE_FACTORS are paired, so we use the length of one array
df_scale_pair_count=${#DF_SCALES[@]}
total_satburst_experiments=$((${#LEARNING_RATES[@]} * ${#FOURIER_SCALES[@]} * $df_scale_pair_count * ${#ITERS[@]} * $loss_count))
total_burst_experiments=$((${#LEARNING_RATES[@]} * ${#FOURIER_SCALES[@]} * $df_scale_pair_count * ${#ITERS[@]} * $loss_count))
total_experiments=$((total_satburst_experiments + total_burst_experiments))
current=0

echo "Output directories:"
echo "  - burst_synth: $BASE_OUTPUT_DIR_BURST"
echo "  - satburst_synth: $BASE_OUTPUT_DIR_SATBURST"
echo ""
echo "Total experiments: $total_experiments"
echo "  Calculation: (learning_rates × fourier_scales × df_scale_pairs × iterations × loss_types)"
echo "  Note: Each experiment processes ALL samples in the dataset via --multi_sample"
echo "  - burst_synth: $total_burst_experiments experiments (${#LEARNING_RATES[@]} LR × ${#FOURIER_SCALES[@]} FS × $df_scale_pair_count DF/SF pairs × ${#ITERS[@]} iter × $loss_count loss_types)"
echo "  - satburst_synth: $total_satburst_experiments experiments (${#LEARNING_RATES[@]} LR × ${#FOURIER_SCALES[@]} FS × $df_scale_pair_count DF/SF pairs × ${#ITERS[@]} iter × $loss_count loss_types)"
echo ""
echo "GPU Assignment:"
echo "  - GPU $DEVICE_MSE: MSE baseline"
echo "  - GPU $DEVICE_GNLL: Regular GNLL"
echo "  - GPU $DEVICE_SEPARATE_UD: Separate UD"
echo ""
echo "Execution Mode: PARALLEL"
echo "  - All variants (MSE, GNLL, Separate UD) will run simultaneously"
echo "  - Each variant uses its own GPU, so there's no GPU conflict"
echo "  - Experiments for each hyperparameter combination run in parallel"
echo ""

# Create base directories for each dataset
mkdir -p "$BASE_OUTPUT_DIR_BURST"
mkdir -p "$BASE_OUTPUT_DIR_SATBURST"

# Sweep for burst_synth (run first)
if [ ${#BURST_SAMPLE_IDS[@]} -gt 0 ]; then
    echo "============================================================================"
    echo "BURST_SYNTH EXPERIMENTS"
    echo "============================================================================"

    for lr in "${LEARNING_RATES[@]}"; do
        for fs in "${FOURIER_SCALES[@]}"; do
            # Iterate over paired DF_SCALES and SCALE_FACTORS (same index = same pair)
            for i in "${!DF_SCALES[@]}"; do
                df="${DF_SCALES[$i]}"
                scale_factor="${SCALE_FACTORS[$i]}"
                for iter in "${ITERS[@]}"; do
                    echo "Running hyperparameter combination: lr=$lr, fs=$fs, df=$df, sf=$scale_factor, iter=$iter"
                    
                    # Array to store background job PIDs
                    pids=()
                    
                    # Launch experiments for each loss type in parallel (each processes ALL samples)
                    # MSE variant (GPU 3) - run in background
                    if [ "$USE_MSE" = "true" ]; then
                        current=$((current + 1))
                        output_mse=$(create_output_folder "burst_synth" "mse" "$lr" "$fs" "$iter" "$df" "$scale_factor")
                        run_experiment "burst_synth" "$output_mse" "$lr" "$fs" "$iter" "$DEVICE_MSE" "false" "false" "$current" "$total_experiments" "$df" "$scale_factor" &
                        pids+=($!)
                    fi
                    
                    # Regular GNLL variant (GPU 1) - run in background
                    if [ "$USE_REGULAR_GNLL" = "true" ]; then
                        current=$((current + 1))
                        output_gnll=$(create_output_folder "burst_synth" "gnll" "$lr" "$fs" "$iter" "$df" "$scale_factor")
                        run_experiment "burst_synth" "$output_gnll" "$lr" "$fs" "$iter" "$DEVICE_GNLL" "true" "false" "$current" "$total_experiments" "$df" "$scale_factor" &
                        pids+=($!)
                    fi
                    
                    # Separate UD variant (GPU 2) - run in background
                    if [ "$USE_SEPARATE_UD" = "true" ]; then
                        current=$((current + 1))
                        output_separate_ud=$(create_output_folder "burst_synth" "separate_ud" "$lr" "$fs" "$iter" "$df" "$scale_factor")
                        run_experiment "burst_synth" "$output_separate_ud" "$lr" "$fs" "$iter" "$DEVICE_SEPARATE_UD" "true" "true" "$current" "$total_experiments" "$df" "$scale_factor" &
                        pids+=($!)
                    fi
                    
                    # Wait for all background jobs for this hyperparameter combination to complete
                    echo "Waiting for ${#pids[@]} parallel experiments to complete..."
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
    done
else
    echo "Skipping burst_synth experiments (no samples found)"
    echo ""
fi

# Sweep for satburst_synth (run after burst_synth)
if [ ${#SATBURST_SAMPLE_IDS[@]} -gt 0 ]; then
    echo "============================================================================"
    echo "SATBURST_SYNTH EXPERIMENTS"
    echo "============================================================================"

    for lr in "${LEARNING_RATES[@]}"; do
        for fs in "${FOURIER_SCALES[@]}"; do
            # Iterate over paired DF_SCALES and SCALE_FACTORS (same index = same pair)
            for i in "${!DF_SCALES[@]}"; do
                df="${DF_SCALES[$i]}"
                scale_factor="${SCALE_FACTORS[$i]}"
                for iter in "${ITERS[@]}"; do
                    echo "Running hyperparameter combination: lr=$lr, fs=$fs, df=$df, sf=$scale_factor, iter=$iter"
                    
                    # Array to store background job PIDs
                    pids=()
                    
                    # Launch experiments for each loss type in parallel (each processes ALL samples)
                    # MSE variant (GPU 3) - run in background
                    if [ "$USE_MSE" = "true" ]; then
                        current=$((current + 1))
                        output_mse=$(create_output_folder "satburst_synth" "mse" "$lr" "$fs" "$iter" "$df" "$scale_factor")
                        run_experiment "satburst_synth" "$output_mse" "$lr" "$fs" "$iter" "$DEVICE_MSE" "false" "false" "$current" "$total_experiments" "$df" "$scale_factor" &
                        pids+=($!)
                    fi
                    
                    # Regular GNLL variant (GPU 1) - run in background
                    if [ "$USE_REGULAR_GNLL" = "true" ]; then
                        current=$((current + 1))
                        output_gnll=$(create_output_folder "satburst_synth" "gnll" "$lr" "$fs" "$iter" "$df" "$scale_factor")
                        run_experiment "satburst_synth" "$output_gnll" "$lr" "$fs" "$iter" "$DEVICE_GNLL" "true" "false" "$current" "$total_experiments" "$df" "$scale_factor" &
                        pids+=($!)
                    fi
                    
                    # Separate UD variant (GPU 2) - run in background
                    if [ "$USE_SEPARATE_UD" = "true" ]; then
                        current=$((current + 1))
                        output_separate_ud=$(create_output_folder "satburst_synth" "separate_ud" "$lr" "$fs" "$iter" "$df" "$scale_factor")
                        run_experiment "satburst_synth" "$output_separate_ud" "$lr" "$fs" "$iter" "$DEVICE_SEPARATE_UD" "true" "true" "$current" "$total_experiments" "$df" "$scale_factor" &
                        pids+=($!)
                    fi
                    
                    # Wait for all background jobs for this hyperparameter combination to complete
                    echo "Waiting for ${#pids[@]} parallel experiments to complete..."
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
    done
else
    echo "Skipping satburst_synth experiments (no samples found)"
    echo ""
fi

echo "============================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "============================================================================"
echo "Results saved in separate directories:"
echo "  - burst_synth: $BASE_OUTPUT_DIR_BURST/"
echo "  - satburst_synth: $BASE_OUTPUT_DIR_SATBURST/"
echo ""
echo "Summary:"
echo "  - Total experiments completed: $current"
echo "  - MSE experiments: GPU $DEVICE_MSE"
echo "  - GNLL experiments: GPU $DEVICE_GNLL"
echo "  - Separate UD experiments: GPU $DEVICE_SEPARATE_UD"
echo ""
echo "To visualize results, run:"
echo "  python visualize_hyperparameter_sweep.py --input_dir $BASE_OUTPUT_DIR_BURST"
echo "  python visualize_hyperparameter_sweep.py --input_dir $BASE_OUTPUT_DIR_SATBURST"
echo ""
echo "To compare results manually, you can:"
echo "  1. Check summary_statistics.json in each experiment folder"
echo "  2. List all experiments:"
echo "     - burst_synth: ls -d $BASE_OUTPUT_DIR_BURST/*"
echo "     - satburst_synth: ls -d $BASE_OUTPUT_DIR_SATBURST/*"
echo "  3. Filter by model type:"
echo "     - MSE: ls -d $BASE_OUTPUT_DIR_BURST/*mse* $BASE_OUTPUT_DIR_SATBURST/*mse*"
echo "     - GNLL: ls -d $BASE_OUTPUT_DIR_BURST/*gnll* $BASE_OUTPUT_DIR_SATBURST/*gnll*"
echo "     - Separate UD: ls -d $BASE_OUTPUT_DIR_BURST/*separate_ud* $BASE_OUTPUT_DIR_SATBURST/*separate_ud*"
echo ""
echo "Note: Experiments run in parallel using different GPUs, so there's no"
echo "      GPU conflict. Each hyperparameter combination runs all three"
echo "      model variants simultaneously on separate GPUs."
echo "============================================================================"

