import os
import torch
from transformers import WavLMModel, HubertModel

# Encoder paths — configured in path_config.sh and sourced by train.sh / test.sh
WAVLM_PATH      = os.environ.get("WAVLM_LARGE_PATH",  "./latent_ckpt/wavlm-large-local")
HUBERT_PATH     = os.environ.get("HUBERT_LARGE_PATH",  "./latent_ckpt/hubert-large-local")
DISTILHUBERT_PATH = os.environ.get("DISTILHUBERT_PATH", "./latent_ckpt/distilhubert-local")

audio = torch.randn(1, 16000)
selected_layers = [6, 12, 24]

# ── WavLM-Large ──────────────────────────────────────────────
try:
    print(f"Loading WavLM-Large from {WAVLM_PATH}...")
    wavlm = WavLMModel.from_pretrained(WAVLM_PATH)
    print("Successfully loaded WavLM model.")

    with torch.no_grad():
        wavlm_outputs = wavlm(audio, output_hidden_states=True)

    for layer_idx in selected_layers:
        print(f"[WavLM]    Layer {layer_idx:>2} shape: {wavlm_outputs.hidden_states[layer_idx].shape}")

except Exception as e:
    print(f"WavLM Error: {e}")

print()

# ── HuBERT-Large ─────────────────────────────────────────────
try:
    print(f"Loading HuBERT-Large from {HUBERT_PATH}...")
    hubert = HubertModel.from_pretrained(HUBERT_PATH, ignore_mismatched_sizes=True)
    print("Successfully loaded HuBERT model.")

    with torch.no_grad():
        hubert_outputs = hubert(audio, output_hidden_states=True)

    for layer_idx in selected_layers:
        print(f"[HuBERT]   Layer {layer_idx:>2} shape: {hubert_outputs.hidden_states[layer_idx].shape}")

except Exception as e:
    print(f"HuBERT Error: {e}")

print()

# ── DistilHuBERT ─────────────────────────────────────────────
try:
    print(f"Loading DistilHuBERT from {DISTILHUBERT_PATH}...")
    distilhubert = HubertModel.from_pretrained(DISTILHUBERT_PATH)
    print("Successfully loaded DistilHuBERT model.")

    with torch.no_grad():
        distil_outputs = distilhubert(audio, output_hidden_states=True)

    # DistilHuBERT only has 2 transformer layers (much smaller than HuBERT-Large!)
    print(f"[DistilHuBERT] Total hidden states: {len(distil_outputs.hidden_states)}")
    for layer_idx in range(len(distil_outputs.hidden_states)):
        print(f"[DistilHuBERT] Layer {layer_idx:>2} shape: {distil_outputs.hidden_states[layer_idx].shape}")

except Exception as e:
    print(f"DistilHuBERT Error: {e}")

# Expected output:
# [WavLM]    Layer  6 shape: torch.Size([1, 49, 1024])
# [WavLM]    Layer 12 shape: torch.Size([1, 49, 1024])
# [WavLM]    Layer 24 shape: torch.Size([1, 49, 1024])
#
# [HuBERT]   Layer  6 shape: torch.Size([1, 49, 1024])
# [HuBERT]   Layer 12 shape: torch.Size([1, 49, 1024])
# [HuBERT]   Layer 24 shape: torch.Size([1, 49, 1024])
#
# [DistilHuBERT] Total hidden states: 3   ← CNN + only 2 transformer layers!
# [DistilHuBERT] Layer  0 shape: torch.Size([1, 49, 768])
# [DistilHuBERT] Layer  1 shape: torch.Size([1, 49, 768])
# [DistilHuBERT] Layer  2 shape: torch.Size([1, 49, 768])