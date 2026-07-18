
"""
Inference script for Speech Enhancement Drifting Model (NCSN++).
Adapted to match user's preferred I/O format (soundfile, pad_spec).
"""

import argparse
import os
import sys
import glob
from pathlib import Path

import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Add parent directory to path to import sgmse modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from backbones.ncsnpp_v2 import NCSNpp_v2
from backbones.ncsnpp_v2_drift import ncsnpp_v2_drift
from util.other import pad_spec, set_torch_cuda_arch_list

# Setup CUDA
set_torch_cuda_arch_list()

# Default Config (Must match training!)
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_window(window_type, window_length, device):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True, device=device))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True, device=device)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")

def spec_fwd(spec, config):
    e = config["spec_abs_exponent"]
    f = config["spec_factor"]
    spec = spec.abs()**e * torch.exp(1j * spec.angle())
    spec = spec * f
    return spec

def spec_bwd(spec, config):
    e = config["spec_abs_exponent"]
    f = config["spec_factor"]
    spec = spec / f
    mag = spec.abs()
    angle = spec.angle()
    mag = mag**(1.0/e)
    spec = mag * torch.exp(1j * angle)
    return spec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--test_dir", type=str, default=None, help="Path to noisy test files directory")
    parser.add_argument("--enhanced_dir", type=str, default=None, help="Path to save enhanced files")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
    args = parser.parse_args()

    # Config
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    target_sr = 16000 # VoiceBank standard
    
    # Paths
    test_dir = args.test_dir if args.test_dir else config.get("test_dir")
    enhanced_dir = args.enhanced_dir if args.enhanced_dir else config.get("enhanced_dir")
    ckpt_path = args.ckpt if args.ckpt else config.get("ckpt")
    
    if not test_dir or not enhanced_dir or not ckpt_path:
        raise ValueError("Must provide test_dir, enhanced_dir, and ckpt in either config or args.")

    # Load Model (Manual Setup)
    print(f"Loading checkpoint {ckpt_path}...")
    if config['model'].lower() == 'ncsnpp_v2_drift':
        model = ncsnpp_v2_drift(
            nf=config["nf"],
            ch_mult=config["ch_mult"],
            num_res_blocks=config["num_res_blocks"],
            attn_resolutions=config["attn_resolutions"],
            image_size=config["image_size"],
            fourier_scale=config["fourier_scale"],
            resamp_with_conv=config["resamp_with_conv"],
            fir=config["fir"],
            fir_kernel=config["fir_kernel"],
            skip_rescale=config["skip_rescale"],
            resblock_type=config["resblock_type"],
            progressive=config["progressive"],
            progressive_input=config["progressive_input"],
            progressive_combine=config["progressive_combine"],
            init_scale=config["init_scale"],
            embedding_type=config["embedding_type"],
            dropout=config["dropout"],
        ).to(device)        
    else:
        model = NCSNpp_v2(
            nf=config["nf"],
            ch_mult=config["ch_mult"],
            num_res_blocks=config["num_res_blocks"],
            attn_resolutions=config["attn_resolutions"],
            image_size=config["image_size"],
            fourier_scale=config["fourier_scale"],
            resamp_with_conv=config["resamp_with_conv"],
            fir=config["fir"],
            fir_kernel=config["fir_kernel"],
            skip_rescale=config["skip_rescale"],
            resblock_type=config["resblock_type"],
            progressive=config["progressive"],
            progressive_input=config["progressive_input"],
            progressive_combine=config["progressive_combine"],
            init_scale=config["init_scale"],
            embedding_type=config["embedding_type"],
            dropout=config["dropout"],
        ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    # Check if checkpoint is dict or model state
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Get list of noisy files
    noisy_files = []
    noisy_files += sorted(glob.glob(os.path.join(test_dir, '*.wav')))
    noisy_files += sorted(glob.glob(os.path.join(test_dir, '**', '*.wav')))
    # noisy_files += sorted(glob.glob(os.path.join(test_dir, '*.flac')))
    
    print(f"Found {len(noisy_files)} files.")
    
    # Prepare Window
    window = get_window(config["window_type"], config["n_fft"], device)

    # Enhance files
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(test_dir, "")
        filename = filename[1:] if filename.startswith("/") else filename

        # try:
        # Load wav (using torchaudio or librosa)
        y, sr = torchaudio.load(noisy_file)
        y = y[0].cpu().numpy() # Mono numpy array

        # Resample if necessary
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # To Tensor
        y = torch.as_tensor(y).to(device)
        T_orig = len(y)

        # Normalize
        norm_factor = y.abs().max() + 1e-8
        y = y / norm_factor
        
        # STFT
        # Ensure input (T,) or (1, T)
        # torch.stft on (T,) -> (F, T, Complex) or (F, T, 2)
        Y = torch.stft(
            y, 
            n_fft=config["n_fft"], 
            hop_length=config["hop_length"], 
            window=window, 
            center=config["center"], 
            return_complex=True
        )
        
        # Transform
        Y_trans = spec_fwd(Y, config)
        
        # Prepare DNN Input: (1, 1, F, T) Complex - Add Batch/Channel dims
        Y_input = Y_trans.unsqueeze(0).unsqueeze(0)
        
        # Pad Spec (Using sgmse util)
        Y_input = pad_spec(Y_input, mode="zero_pad") 
        
        # Prepare Inputs to match Training
        # Training: model(y + z, y, t)
        batch_size = Y_input.shape[0]
        z = torch.randn_like(Y_input).to(device)
        t = torch.ones(batch_size, device=device) 
        
        with torch.no_grad():
            # Forward Pass
            sample = model(Y_input + 0.05*z, t)
            # sample = model(Y_input, t)
        
        # Debug: Check statistics
        # print(f"Sample Max: {sample.abs().max()}, Min: {sample.abs().min()}")

        # Results
        X_hat_trans = sample.squeeze(0).squeeze(0) # (F, T_padded)
        
        # Inverse Transform
        X_hat = spec_bwd(X_hat_trans, config)
        
        # ISTFT
        x_hat = torch.istft(
            X_hat, 
            n_fft=config["n_fft"], 
            hop_length=config["hop_length"], 
            window=window, 
            center=config["center"],
            length=T_orig
        )
        
        # Renormalize to match input's maximum magnitude
        max_val = x_hat.abs().max()
        if max_val > 1e-8:
            x_hat = x_hat / max_val * norm_factor
        else:
            x_hat = x_hat * norm_factor
        # x_hat = x_hat * norm_factor
        # x_hat = x_hat
        
        # Save
        out_path = os.path.join(enhanced_dir, filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, x_hat.cpu().numpy(), target_sr)
    

    print(f"Enhancement saved: {enhanced_dir}")
