
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


from backbones.ncsnpp_v2_drift_input_condition import ncsnpp_v2_drift_input_condition

from backbones.tfgridnet import TFGridNet_Backbone
from backbones.TFGridNet_Causal import TFGridNet_Causal
from backbones.streaming_unet import CausalNCSNpp
from util.other import pad_spec, set_torch_cuda_arch_list

# Setup CUDA
set_torch_cuda_arch_list()

# Default Config (Must match training!)
import json

from util.config_loader import load_config

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

    model_name = config['model'].lower()
    # Determine if model requires (x, y) conditioning
    _cond_models = {'freq_tcn_se', 'mamba_seunet', 'semambapp', 'tfgridnet', 'tfgridnet_causal', 'streamunet'}
    use_cond_fwd = 'input_condition' in model_name or model_name in _cond_models
    use_strict_gaussian = config.get('use_strict_gaussian', False)

    if model_name == 'tfgridnet':
        model = TFGridNet_Backbone(
            n_layers          = config.get('n_layers', 5),
            emb_dim           = config.get('emb_dim', 32),
            lstm_hidden_units = config.get('lstm_hidden_units', 100),
            attn_n_head       = config.get('attn_n_head', 4),
            causal            = config.get('causal', False),
        ).to(device)
    elif model_name == 'tfgridnet_causal':
        model = TFGridNet_Causal(
            n_srcs                 = config.get('n_srcs', 1),
            n_layers               = config.get('n_layers', 6),
            lstm_hidden_units      = config.get('lstm_hidden_units', 192),
            attn_n_head            = config.get('attn_n_head', 4),
            attn_qk_output_channel = config.get('attn_qk_output_channel', 4),
            emb_dim                = config.get('emb_dim', 48),
            emb_ks                 = config.get('emb_ks', 4),
            emb_hs                 = config.get('emb_hs', 1),
            eps                    = config.get('eps', 1e-5),
        ).to(device)
    elif model_name == 'streamunet':
        model = CausalNCSNpp(
            nf              = config['nf'],
            ch_mult         = config['ch_mult'],
            num_res_blocks  = config['num_res_blocks'],
            input_channels  = config.get('input_channels', 4),
            output_channels = config.get('output_channels', 2),
            input_freqs     = config.get('input_freqs', 256),
            dropout         = config.get('dropout', 0.0),
            norm_type       = config.get('norm_type', 'subband_grouped_batchnorm'),
            no_freq_groups_below = config.get('no_freq_groups_below', 16),
            freq_groups     = config.get('freq_groups', 4),
            down_dilation   = config.get('down_dilation', 2),
            attn_resolutions = tuple(config.get('attn_resolutions', [])),
        ).to(device)
    else:
        # NCSN++ family — requires the full set of architecture keys
        model_kwargs = dict(
            nf=config['nf'],
            ch_mult=config['ch_mult'],
            num_res_blocks=config['num_res_blocks'],
            attn_resolutions=config['attn_resolutions'],
            image_size=config['image_size'],
            fourier_scale=config['fourier_scale'],
            resamp_with_conv=config['resamp_with_conv'],
            fir=config['fir'],
            fir_kernel=config['fir_kernel'],
            skip_rescale=config['skip_rescale'],
            resblock_type=config['resblock_type'],
            progressive=config['progressive'],
            progressive_input=config['progressive_input'],
            progressive_combine=config['progressive_combine'],
            init_scale=config['init_scale'],
            embedding_type=config['embedding_type'],
            dropout=config['dropout'],
            causal=config.get('causal', True),
        )
        if model_name == 'ncsnpp_v2_drift_input_condition':
            model = ncsnpp_v2_drift_input_condition(**model_kwargs).to(device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    # Check if checkpoint is dict or model state
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Get list of noisy files
    # Support both flat directory structures and recursive subfolder layouts (like EARS) safely
    raw_files = glob.glob(os.path.join(test_dir, '*.wav')) + glob.glob(os.path.join(test_dir, '**', '*.wav'), recursive=True)
    noisy_files = sorted(list(set(raw_files)))
    
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
        
        # STFT → (F, n_frames_orig) where F = n_fft//2 + 1
        Y = torch.stft(
            y, 
            n_fft=config["n_fft"], 
            hop_length=config["hop_length"], 
            window=window, 
            center=config["center"], 
            return_complex=True
        )
        n_frames_orig = Y.shape[-1]   # ← save BEFORE pad_spec expands it
        
        # Transform
        Y_trans = spec_fwd(Y, config)
        
        # Prepare DNN Input: (1, 1, F, T) Complex - Add Batch/Channel dims
        Y_input = Y_trans.unsqueeze(0).unsqueeze(0)
        
        # pad_spec zero-pads the time dim to the next multiple of 64
        # so the U-Net downsampling never sees a size it can't halve.
        # The model output will have T_padded >= n_frames_orig frames.
        Y_input = pad_spec(Y_input, mode="zero_pad") 
        
        # Prepare Inputs to match Training
        batch_size = Y_input.shape[0]
        z = torch.randn_like(Y_input).to(device)
        t = torch.ones(batch_size, device=device)

        inference_sigma = config.get("inference_sigma", config.get("sigma_max", 0.15))
        inference_sigma = 0.05

        with torch.no_grad():
            if use_cond_fwd:
                if use_strict_gaussian:
                    sample = model(inference_sigma * z, Y_input, use_strict_gaussian=use_strict_gaussian)
                else:
                    sample = model(inference_sigma * z, Y_input)
            else:
                sample = model(Y_input)

        # Results: (1, 1, F, T_padded) → (F, T_padded)
        X_hat_trans = sample.squeeze(0).squeeze(0)

        # ── KEY FIX: trim the padded time frames back to the original count ──
        # pad_spec added (T_padded - n_frames_orig) zero frames on the right.
        # The model processed those zeros and produced output for them too.
        # If we feed all T_padded frames into ISTFT, the overlap-add extends
        # the waveform by (T_padded - n_frames_orig) * hop_length extra samples.
        # torch.istft(length=T_orig) DOES trim the output, but only AFTER the
        # full overlap-add is computed — so padded frames still add a
        # "reverberant tail" of zeros to the reconstruction.
        # Trimming the spec here avoids that artifact entirely.
        X_hat_trans = X_hat_trans[:, :n_frames_orig]

        # Inverse Transform
        X_hat = spec_bwd(X_hat_trans, config)
        
        # ISTFT — length=T_orig is a safety trim for any remaining sample-level mismatch
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
        
        # Save - flatten the output folder by saving directly in enhanced_dir
        out_filename = os.path.basename(filename)
        out_path = os.path.join(enhanced_dir, out_filename)
        os.makedirs(enhanced_dir, exist_ok=True)
        sf.write(out_path, x_hat.cpu().numpy(), target_sr)
    

    print(f"Enhancement saved: {enhanced_dir}")
