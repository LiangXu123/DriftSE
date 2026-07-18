#!/bin/bash
# Train Speech Enhancement Drifting Model
export PYTHONWARNINGS="ignore"

# Check GPU ID argument
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 GPU_ID"
    echo "GPU_ID is required."
    exit 1
fi
GPU_ID=$1
echo "Using GPU ID: $GPU_ID"
# --- Environment Setup ---
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

# --- Load path configuration (data dirs + encoder checkpoints) ---
source "$(dirname "$0")/path_config.sh"

export WANDB_MODE=dryrun
echo "W&B mode set to: dryrun"

# Optional config argument
CONFIG_PATH=${2:-"./config/with_z/v2_drift2_distillhubert_three_layers.json"}
echo "Using config: $CONFIG_PATH"

python train.py \
    --config "$CONFIG_PATH"