import os
import sys
import argparse
import torch
import torchaudio

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FILE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

parser = argparse.ArgumentParser(description="Test all encoders")
parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use (e.g., '0' or '1')")
args, unknown = parser.parse_known_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dummy audio input (2 seconds at 16kHz)
audio_16k = torch.randn(1, 32000).to(device)

print("="*50)
print("1. WavLM-Large")
try:
    from transformers import WavLMModel
    model_id = "./latent_ckpt/wavlm-large-local"
    wavlm = WavLMModel.from_pretrained(model_id).to(device)
    with torch.no_grad():
        out = wavlm(audio_16k, output_hidden_states=True).hidden_states
    print(f"Layer 6 shape: {out[6].shape}")
    print(f"Layer 12 shape: {out[12].shape}")
    print(f"Layer 24 shape: {out[24].shape}")
except Exception as e:
    print(f"Failed: {e}")

print("="*50)
print("2. HuBERT-Large")
try:
    from transformers import HubertModel
    model_id = "./latent_ckpt/hubert-large-local"
    hubert = HubertModel.from_pretrained(model_id, ignore_mismatched_sizes=True).to(device)
    with torch.no_grad():
        out = hubert(audio_16k, output_hidden_states=True).hidden_states
    print(f"Layer 6 shape: {out[6].shape}")
    print(f"Layer 12 shape: {out[12].shape}")
    print(f"Layer 24 shape: {out[24].shape}")
except Exception as e:
    print(f"Failed: {e}")

print("="*50)
print("3. DistilHuBERT")
try:
    from transformers import HubertModel
    model_id = "./latent_ckpt/distilhubert-local"
    distil = HubertModel.from_pretrained(model_id).to(device)
    with torch.no_grad():
        out = distil(audio_16k, output_hidden_states=True).hidden_states
    print(f"Layer 0 shape: {out[0].shape}")
    print(f"Layer 1 shape: {out[1].shape}")
    print(f"Layer 2 shape: {out[2].shape}")
except Exception as e:
    print(f"Failed: {e}")

print("="*50)
print("4. WavCube-pro")
try:
    wavcube_dir = "./latent_ckpt/WavCube"
    if wavcube_dir not in sys.path:
        sys.path.insert(0, wavcube_dir)
    from vocos import Vocos
    cfg_path = os.path.join(wavcube_dir, "configs/WavCube-stage2.yaml")
    ckpt_path = os.path.join(wavcube_dir, "WavCube-pro/checkpoints/vocos_checkpoint_epoch%3D34_step%3D200000_val_loss%3D3.2140.ckpt")
    vocos = Vocos.from_config(cfg_path)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    vocos.load_state_dict(sd, strict=False)
    vocos = vocos.to(device).eval()
    
    # normalize
    mean = audio_16k.mean(dim=-1, keepdim=True)
    std = audio_16k.std(dim=-1, keepdim=True)
    norm_audio = (audio_16k - mean) / (std + 1e-5)
    
    with torch.no_grad():
        feat = vocos.feature_extractor.inf_new(norm_audio)
    print(f"Output shape: {feat.shape}")
except Exception as e:
    print(f"Failed: {e}")

print("="*50)
print("5. BEATs")
try:
    import importlib.util
    beats_dir = "./latent_ckpt/beats-local/"
    if beats_dir not in sys.path:
        sys.path.insert(0, beats_dir)
    beats_ckpt = os.path.join(beats_dir, "BEATs_iter3_plus_AS2M.pt")
    spec = importlib.util.spec_from_file_location("BEATs", os.path.join(beats_dir, "BEATs.py"))
    beats_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(beats_module)
    ckpt = torch.load(beats_ckpt, map_location="cpu")
    cfg = beats_module.BEATsConfig(ckpt["cfg"])
    beats = beats_module.BEATs(cfg)
    beats.load_state_dict(ckpt["model"])
    beats = beats.to(device).eval()
    
    captured = {}
    def hook(name):
        def _h(m, i, o):
            captured[name] = o[0].transpose(0, 1)
        return _h
    beats.encoder.layers[5].register_forward_hook(hook("layer6"))
    beats.encoder.layers[11].register_forward_hook(hook("layer12"))
    
    with torch.no_grad():
        out, _ = beats.extract_features(audio_16k, padding_mask=None)
    print(f"Layer 6 shape: {captured['layer6'].shape}")
    print(f"Layer 12 shape: {captured['layer12'].shape}")
except Exception as e:
    print(f"Failed: {e}")

print("="*50)
print("6. PANNs CNN14")
try:
    from panns_inference import AudioTagging
    panns_dir = "./latent_ckpt/panns-local"
    ckpt_path = os.path.join(panns_dir, "Cnn14_mAP=0.431.pth")
    panns = AudioTagging(checkpoint_path=ckpt_path, device=str(device))
    model = panns.model.module if hasattr(panns.model, "module") else panns.model
    model = model.to(device).eval()
    
    audio_32k = torchaudio.functional.resample(audio_16k, 16000, 32000)
    
    captured = {}
    def hook(name):
        def _h(m, i, o):
            captured[name] = o
        return _h
    
    model.conv_block5.register_forward_hook(hook("layer5"))
    model.conv_block6.register_forward_hook(hook("layer6"))
    
    with torch.no_grad():
        model(audio_32k)
    print(f"Layer 5 shape: {captured['layer5'].shape}")
    print(f"Layer 6 shape: {captured['layer6'].shape}")
except Exception as e:
    print(f"Failed: {e}")


# OUTPUT LOOKS LIKE:
# > python test_encoder.py  --gpu 1
# Using device: cuda
# ==================================================
# 1. WavLM-Large
# Layer 6 shape: torch.Size([1, 99, 1024])
# Layer 12 shape: torch.Size([1, 99, 1024])
# Layer 24 shape: torch.Size([1, 99, 1024])
# ==================================================
# 2. HuBERT-Large
# Layer 6 shape: torch.Size([1, 99, 1024])
# Layer 12 shape: torch.Size([1, 99, 1024])
# Layer 24 shape: torch.Size([1, 99, 1024])
# ==================================================
# 3. DistilHuBERT
# Layer 0 shape: torch.Size([1, 99, 768])
# Layer 1 shape: torch.Size([1, 99, 768])
# Layer 2 shape: torch.Size([1, 99, 768])
# ==================================================
# 4. WavCube-pro
# Output shape: torch.Size([1, 99, 128])
# ==================================================
# 5. BEATs
# Layer 6 shape: torch.Size([1, 96, 768])
# Layer 12 shape: torch.Size([1, 96, 768])
# ==================================================
# 6. PANNs CNN14
# Layer 5 shape: torch.Size([1, 1024, 6, 2])
# Layer 6 shape: torch.Size([1, 2048, 6, 2])
