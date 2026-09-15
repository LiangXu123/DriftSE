#!/bin/bash
export PYTHONWARNINGS="ignore"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 GPU_ID [CONFIG_PATH]"
    exit 1
fi

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$GPU_ID
export WANDB_MODE=dryrun

# --- Load path configuration ---
source "$(dirname "$0")/path_config.sh"

# Default list of configs
if [ "$#" -ge 2 ]; then
    CONFIGS=("$2")
else
    CONFIGS=(       
        "./config/DriftSE/experiments/SE_EARS_distilhubert_ch1282_incond_PANNs.json"
    )
fi
# Assert all config files exist
for CONFIG in "${CONFIGS[@]}"; do
    if [ ! -f "$CONFIG" ]; then
        echo "Error: config file '$CONFIG' does not exist!"
        exit 1
    fi
    echo "Found config: $CONFIG"
done


# ============================================================================
# PHASE 1: Run all enhancement
# ============================================================================

TOTAL=${#CONFIGS[@]}
echo "========================================"
echo "PHASE 1: Running all enhancement"
echo "========================================"
# Loop through configs
for i in "${!CONFIGS[@]}"; do
    CONFIG=${CONFIGS[$i]}
    echo "Running: $CONFIG ($((i+1))/$TOTAL)"
    [ ! -f "$CONFIG" ] && echo "Skipped" && continue
    python enhancement.py --config "$CONFIG"
done

# ============================================================================
# PHASE 2: Run all metrics calculations (CPU)
# ============================================================================
echo "========================================"
echo "PHASE 2: Running all metrics calculations"
echo "========================================"

for index in "${!CONFIGS[@]}"; do
    CONFIG=${CONFIGS[$index]}
    
    # Read test_dir, clean_dir, and enhanced_dir from config JSON
    CONFIG_INFO=$(python3 -c "
import sys, json
cfg = json.load(open(sys.argv[1]))
print(f\"{cfg.get('test_dir', '')}\t{cfg.get('clean_dir', '')}\t{cfg.get('enhanced_dir', '')}\")
" "$CONFIG")

    IFS=$'\t' read -r CONFIG_TEST_DIR CONFIG_CLEAN_DIR ENHANCED_DIR <<< "$CONFIG_INFO"

    # Dynamically resolve clean and noisy directories for this config
    get_dataset_paths "$CONFIG_TEST_DIR" "$CONFIG_CLEAN_DIR"

    echo ""
    echo "--- Metrics: $ENHANCED_DIR ---"
    echo "Config            : $CONFIG"
    echo "Clean directory   : $CLEAN_DIR"
    echo "Noisy directory   : $NOISY_DIR"
    echo "Enhanced directory: $ENHANCED_DIR"

    CMD="python3 util/calc_metrics.py --clean_dir \"$CLEAN_DIR\" --noisy_dir \"$NOISY_DIR\" --enhanced_dir \"$ENHANCED_DIR\""
    echo "Executing command: $CMD"
    if ! eval $CMD; then
        echo "Metrics calculation failed for $ENHANCED_DIR"
        continue
    fi
done