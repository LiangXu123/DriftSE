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
        "./config/with_z/v2_drift2_distillhubert_three_layers.json"
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
    ENHANCED_DIR=$(python -c "import sys, json; print(json.load(open(sys.argv[1]))['enhanced_dir'])" "$CONFIG")

    echo ""
    echo "--- Metrics: $ENHANCED_DIR ---"
    echo "Clean directory   : $CLEAN_DIR_VOICEBANK"
    echo "Noisy directory   : $NOISY_DIR_VOICEBANK"
    echo "Enhanced directory: $ENHANCED_DIR"

    CMD="python3 util/calc_metrics.py --clean_dir \"$CLEAN_DIR_VOICEBANK\" --noisy_dir \"$NOISY_DIR_VOICEBANK\" --enhanced_dir \"$ENHANCED_DIR\""
    echo "Executing command: $CMD"
    if ! eval $CMD; then
        echo "Metrics calculation failed for $ENHANCED_DIR"
        continue
    fi
done