#!/bin/bash

# =======================
# Default path candidates
# =======================

# --- VoiceBank test set ---
export CLEAN_DIR_VOICEBANK="./data/voicebank/true_split_train_val_test/16K/test/clean"
export NOISY_DIR_VOICEBANK="./data/voicebank/true_split_train_val_test/16K/test/noisy"
export BASE_DIR_VOICEBANK="./data/voicebank/true_split_train_val_test/16K/"

# --- DNS-Challenge test set ---
export INPUT_DIR_DNS_blind="./data/DNS-Challenge/real_2020_blind_test_set/real/"

# --- SSL Encoder checkpoints ---
export WAVLM_LARGE_PATH="./latent_ckpt/wavlm-large-local"
export HUBERT_LARGE_PATH="./latent_ckpt/hubert-large-local"
export DISTILHUBERT_PATH="./latent_ckpt/distilhubert-local"

