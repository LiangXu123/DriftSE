#!/bin/bash

# =======================
# Default path candidates
# =======================

# --- VoiceBank test set ---
export CLEAN_DIR_VOICEBANK="/vol/liangxu-solar/data/voicebank/VoiceBank-DEMAND-16k-wav/test/clean"
export NOISY_DIR_VOICEBANK="/vol/liangxu-solar/data/voicebank/VoiceBank-DEMAND-16k-wav/test/noisy"

## WSJ0-REVERB
export CLEAN_DIR_REVERB="/vol/liangxu-solar/data/wsj0_reverb/test/anechoic"
export NOISY_DIR_REVERB="/vol/liangxu-solar/data/wsj0_reverb/test/reverb"

## ears_wham for SE
export CLEAN_DIR_EARS_WHAM="/vol/liangxu-solar/data/EARS-WHAM_v2_16k/test/clean/"
export NOISY_DIR_EARS_WHAM="/vol/liangxu-solar/data/EARS-WHAM_v2_16k/test/noisy/"

## ears_reverb for Dereverb
export CLEAN_DIR_EARS_REVERB="/vol/liangxu-solar/data/EARS-Reverb_v2_16k/test/clean/"
export NOISY_DIR_EARS_REVERB="/vol/liangxu-solar/data/EARS-Reverb_v2_16k/test/reverberant/"

# Helper function to dynamically resolve clean & noisy directories for any dataset
get_dataset_paths() {
    local target_dir="$1"
    local config_clean_dir="$2"

    # 1. Direct override if clean_dir is specified in the JSON config
    if [ -n "$config_clean_dir" ] && [ "$config_clean_dir" != "None" ] && [ "$config_clean_dir" != "null" ]; then
        CLEAN_DIR="$config_clean_dir"
        NOISY_DIR="$target_dir"
        return
    fi

    # 2. Match dataset paths based on directory keywords
    case "$target_dir" in
        *EARS-WHAM*|*ears_wham*)
            CLEAN_DIR="$CLEAN_DIR_EARS_WHAM"
            NOISY_DIR="$NOISY_DIR_EARS_WHAM"
            ;;
        *EARS-Reverb*|*ears_reverb*)
            CLEAN_DIR="$CLEAN_DIR_EARS_REVERB"
            NOISY_DIR="$NOISY_DIR_EARS_REVERB"
            ;;
        *wsj0_reverb*)
            CLEAN_DIR="$CLEAN_DIR_REVERB"
            NOISY_DIR="$NOISY_DIR_REVERB"
            ;;
        *voicebank*)
            CLEAN_DIR="$CLEAN_DIR_VOICEBANK"
            NOISY_DIR="$NOISY_DIR_VOICEBANK"
            ;;
        *)
            # Fallback heuristic for custom/unregistered paths
            NOISY_DIR="$target_dir"
            if [[ "$target_dir" == *"/reverberant"* ]]; then
                CLEAN_DIR="${target_dir%/reverberant*}/clean"
            elif [[ "$target_dir" == *"/reverb"* ]]; then
                CLEAN_DIR="${target_dir%/reverb*}/anechoic"
            elif [[ "$target_dir" == *"/noisy"* ]]; then
                CLEAN_DIR="${target_dir%/noisy*}/clean"
            else
                CLEAN_DIR="$target_dir"
            fi
            ;;
    esac
}