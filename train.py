
"""
Training script for Drifting Models on Speech Enhancement (SE).
Uses NCSN++ v2 (from sgmse) as the generator backbone.
Implements Coupled Normalization to fix magnitude learning issues.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FF
from torch.utils.data import DataLoader
from torchinfo import summary
# Add parent directory to path to import sgmse modules
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

from backbones.ncsnpp_v2 import NCSNpp_v2
from backbones.ncsnpp_v2_drift import ncsnpp_v2_drift
from util.drifting import compute_V, normalize_features, normalize_drift
from torch_pesq import PesqLoss
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
from util.utils import (
    EMA,
    WarmupLRScheduler,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    set_seed,
)
from util.speech_dataset import SpeechDataset
import wandb
from tqdm import tqdm
import json
from scipy.stats import truncnorm
from transformers import WavLMModel, HubertModel
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(
            f"Window type {window_type} not implemented!")

def to_audio(spec, config):
    """
    Reverse the spectrogram transformation and ISTFT.
    spec: (B, F, T) complex tensor
    """
    # 1. Reverse Spec Transform
    # spec = spec.abs()**0.5 * exp(j * angle) * 0.15
    # So: spec / 0.15 = abs**0.5 * exp(j * angle)
    # abs_orig = (abs / 0.15)**2
    
    spec_factor = config["spec_factor"]
    
    mag = spec.abs()
    phase = spec.angle()
    
    mag_orig = (mag / spec_factor) ** 2
    spec_orig = mag_orig * torch.exp(1j * phase)
    
    # 2. ISTFT
    # Matching SpeechDataset defaults: n_fft=510, hop_length=128, window=hann
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    window = get_window(config["window_type"], n_fft).to(spec.device)
    
    # istft requires (B, F, T) complex
    # train_step gives (B, F, T) complex directly if we use x_gen_complex
    
    wav = torch.istft(
        spec_orig,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
    )
    return wav

def compute_ccmse_loss(gen_wav, clean_wav, fft_sizes=(512, 1024, 2048), eps=1e-8):
    """
    MultiResolution Complex Compressed MSE (CCMSE) loss.

    Computes a normalised complex-valued MSE between generated and clean waveforms
    simultaneously across multiple STFT resolutions, then averages:

        L_CCMSE = (1 / |R|) * sum_r  ||STFT_r(gen) - STFT_r(clean)||_F^2
                                       ─────────────────────────────────────
                                          ||STFT_r(clean)||_F^2  +  eps

    Args:
        gen_wav   : (B, T) generated waveform  — original amplitude domain
        clean_wav : (B, T) clean reference     — original amplitude domain
        fft_sizes : iterable of FFT sizes to use (default: 512, 1024, 2048)
        eps       : stability constant for the denominator

    Returns:
        Scalar loss tensor.
    """
    total = 0.0
    device = gen_wav.device

    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, periodic=True).to(device)

        # STFT: returns (B, F, T_frames) complex
        S_gen   = torch.stft(gen_wav,   n_fft=n_fft, hop_length=hop,
                             window=win, center=True, return_complex=True)
        S_clean = torch.stft(clean_wav, n_fft=n_fft, hop_length=hop,
                             window=win, center=True, return_complex=True)

        # Frobenius norm squared over (F, T_frames) per sample, then mean over batch
        diff_sq   = (S_gen - S_clean).abs().pow(2).sum(dim=(-2, -1)).mean()   # scalar
        denom     = S_clean.abs().pow(2).sum(dim=(-2, -1)).mean() + eps        # scalar

        total = total + diff_sq / denom

    return total / len(fft_sizes)

def get_noise_schedule(config, batch_size, device):

    """
    Returns the noise (sigma) based on the schedule.
    """
    mean = config.get('mean', -3.0)
    std = config.get('std', 1.2)
    sigma_max = config.get('sigma_max', 0.35)
    sigma_min = 0.01
    noise_schedule = config.get('noise_schedule', 'log').lower()

    if noise_schedule == 'log':
        # Log-Normal (Proposed)
        # Biases sampling toward smaller sigmas while still occasionally hitting large ones.
        # log_sigma = torch.randn(batch_size, device=device) * std + mean
        # t_noise = torch.exp(log_sigma)
        
        # 1. Calculate bounds in Z-score space (Standard Normal)
        # We want log_sigma in [ln(sigma_min), ln(sigma_max)]
        a = (np.log(sigma_min) - mean) / std
        b = (np.log(sigma_max) - mean) / std
        
        # 2. Sample from Truncated Normal on CPU
        # This guarantees the distribution shape is preserved without a hard clip spike.
        log_sigma = truncnorm.rvs(a, b, loc=mean, scale=std, size=batch_size)
        
        log_sigma_tensor = torch.from_numpy(log_sigma).float().to(device)
        t_noise = torch.exp(log_sigma_tensor)
        
    elif noise_schedule == 'cosine':
        # Cosine (Ablation A)
        u = torch.rand(batch_size, device=device)
        s = 0.008
        t_noise = (torch.cos(((u + s) / (1 + s)) * np.pi / 2)) ** 2 * sigma_max
    elif noise_schedule == 'linear': # 'linear' 
        # Linear (Ablation B)
        t_noise = torch.rand(batch_size, device=device) * sigma_max
    else:
        raise ValueError(f"Unknown noise schedule: {noise_schedule}")

    # Clamp to avoid extreme values
    t_noise = t_noise.clamp(min=1e-5, max=sigma_max)
    return t_noise

def normalize_data(gen_x, clean_x, config):
    """
    Normalize generated and clean features based on config['norm_method'].
    """
    B, F, T, C = gen_x.shape
    norm_method = config.get("norm_method", "none") 
    
    # helper to flatten/reshape
    def to_frame_flat(x): return x.permute(0, 2, 1, 3).reshape(B * T, F * C)
    def to_utt_flat(x): return x.reshape(B, F * T * C)
    
    feat_gen_norm = None
    feat_pos_norm = None
    global_scale = 1.0

    if norm_method == 'utterance_level':
        # Flatten to (B, D_utt)
        f_gen = to_utt_flat(gen_x)
        f_pos = to_utt_flat(clean_x) # (B, D)
        
        with torch.no_grad():
            # Norm per utterance
            utt_scales = torch.norm(f_pos, p=2, dim=1, keepdim=True) + 1e-8
            global_scale = utt_scales # Keep shape (B, 1) effectively
        
        feat_gen_norm = f_gen / global_scale
        feat_pos_norm = f_pos / global_scale
        
    elif norm_method == 'frame_level':
        # Flatten to (B*T, D_frame)
        f_gen = to_frame_flat(gen_x)
        f_pos = to_frame_flat(clean_x)
        
        with torch.no_grad():
            # Global scale based on average frame energy
            frame_norms = torch.norm(f_pos, p=2, dim=1, keepdim=True)
            global_scale = frame_norms.mean().clamp(min=1e-5)
            
        feat_gen_norm = f_gen / global_scale
        feat_pos_norm = f_pos / global_scale
        
    elif norm_method == 'none':
        # No normalization, just use raw data. 
        # Default to frame_level shape for consistency
        feat_gen_norm = to_frame_flat(gen_x)
        feat_pos_norm = to_frame_flat(clean_x)
        global_scale = 1.0
        
    else:
        raise ValueError(f"Unknown norm_method: {norm_method}")
        
    return feat_gen_norm, feat_pos_norm, global_scale


def compute_drift_term(feat_gen_norm, feat_pos_norm, x_shape, config):
    """
    Compute the drift term V based on config['drift_method'].
    Input features are expected to be in the shape dictated by norm_method.
    This function handles reshaping if drift_method != norm_method.
    """
    B, F, T, C = x_shape
    norm_method = config.get("norm_method", "none")
    drift_method = config.get("drift_method", "none")
    temperatures = config["temperatures"]
    
    target_gen = feat_gen_norm
    target_pos = feat_pos_norm
    
    # 1. Reshape if needed
    if drift_method == 'utterance_level':
        # We need (B, D_utt)
        if norm_method == 'frame_level' or norm_method == 'none':
            # Currently (B*T, D_frame), reshape to (B, D_utt)
            target_gen = feat_gen_norm.reshape(B, -1)
            target_pos = feat_pos_norm.reshape(B, -1)
            
    elif drift_method == 'frame_level':
        # We need (B*T, D_frame)
        if norm_method == 'utterance_level':
            # Currently (B, D_utt), reshape to (B*T, D_frame)
            target_gen = feat_gen_norm.reshape(B*T, -1)
            target_pos = feat_pos_norm.reshape(B*T, -1)
            
    else:
        raise ValueError(f"Unknown drift_method: {drift_method}")

    # Set up History/Negatives
    y_pos_all = target_pos
    y_neg_all = target_gen
    mask_self = True

    # Compute V
    V_total = torch.zeros_like(target_gen)
    
    positive_drift_weight = config.get("Positive_drift_weight", 1.0)
    negative_drift_weight = config.get("Negative_drift_weight", 1.0)
    
    for tau in temperatures:
        V_tau = compute_V(
            target_gen,
            y_pos_all,
            y_neg_all,
            tau,
            mask_self=mask_self,
            positive_drift_weight=positive_drift_weight,
            negative_drift_weight=negative_drift_weight
        )
        # v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
        # V_tau = V_tau / (v_norm + 1e-8)
        V_total += V_tau

    V_total /= len(temperatures)
    
    # Return inputs used for drift (target_gen) so we can compute loss against them + V
    return V_total, target_gen, target_pos

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple,
    batch_idx: int,
    config: dict,
    device: torch.device,
    pesq_loss_fn: Optional[nn.Module] = None,
    sisdr_loss_fn: Optional[nn.Module] = None,
    wavlm_model: Optional[nn.Module] = None,
) -> dict:
    """
    Single training step for Speech Enhancement Drifting (NCSN++).
    """
    model.train()
    
    # Unpack batch: Clean (X Target), Noisy (Y Condition)
    # Shapes: (B, 2, F, T) Real Tensor
    clean_speech, noisy_speech, clean_audio_wav, normfac = batch
    clean_speech = clean_speech.to(device)
    noisy_speech = noisy_speech.to(device)
    clean_audio_wav = clean_audio_wav.to(device)
    normfac = normfac.to(device)  # (B,) scalar per sample

    batch_size = clean_speech.shape[0]
    temperatures = config["temperatures"]
    smooth_drift = config.get("smooth_drift", False) 

    # 1. Prepare Inputs for NCSN++ (Complex Tensors)
    # (B, 2, F, T) Real -> Permute -> (B, F, T, 2) -> Complex (B, F, T) -> Unsqueeze -> (B, 1, F, T)
    x_pos_complex = torch.view_as_complex(clean_speech.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)
    y_cond_complex = torch.view_as_complex(noisy_speech.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)

    # Sample Output Noise z ~ N(0, I) (Complex Standard Normal)
    # 
    
    # Time Embedding t (Fixed at 1.0 for One-Step Drifting, based on ouve/sub-vp logic)
    # NCSN++ usually expects t in [0, 1] or continuous.
    # t = torch.ones(batch_size, device=device)

    # 2. Generate Samples (Predict Clean Speech)
    # NCSN++ Forward: (z, cond, t) -> x_gen_complex
    # Output shape: (B, 1, F, T) Complex
    # x_gen_complex = model(z_complex, y_cond_complex, t)
    train_add_gaussian = config.get('train_add_gaussian', False)
    if str(train_add_gaussian).lower() == 'true':
        # Noise Schedule
        # print('train_add_gaussian:',train_add_gaussian)
        z_complex = torch.randn_like(x_pos_complex)
        t_noise = get_noise_schedule(config, batch_size, device)
        sigma_broadcast = t_noise.view(batch_size, 1, 1, 1)
        noisy_input = y_cond_complex + (sigma_broadcast * z_complex)        
        x_gen_complex = model(noisy_input, torch.ones(batch_size, device=device))
    else:
        # print('train_add_gaussian:11111',train_add_gaussian)
        x_gen_complex = model(y_cond_complex, torch.ones(batch_size, device=device))
    # Convert back to Real representation for Loss Calculation (Flattened)
    # (B, 1, F, T) Complex -> (B, F, T) -> (B, F, T, 2)
    x_gen_real = torch.view_as_real(x_gen_complex.squeeze(1)) # (B, F, T, 2)
    clean_speech_flat = torch.view_as_real(x_pos_complex.squeeze(1)) # (B, F, T, 2)

    # 3. Compute Drifting Loss
    # Flatten features (Treat each time frame as a sample)
    # x_gen_real: (B, F, T, 2)
    # Permute to (B, T, F, 2) -> Reshape to (B * T, F * 2)
    # This reduces dimensionality from 131,072 (entire spec) to 512 (one frame)
    # 3. Compute Drifting Loss
    # 3. Compute Drifting Loss
    
    # helper to flatten/reshape
    B, F, T, C = x_gen_real.shape
    
    # 3a. Normalize Data
    feat_gen_norm, feat_pos_norm, global_scale = normalize_data(x_gen_real, clean_speech_flat, config)
        
    # 3b. Compute Drift Term
    V_total, target_gen, target_pos = compute_drift_term(feat_gen_norm, feat_pos_norm, x_gen_real.shape, config)

    
    # Apply drift to the CURRENT normalized features for Loss
    target = (target_gen + V_total).detach()
    
    drift_loss = FF.mse_loss(target_gen, target) * drift_weight if drift_weight > 0 else 0.0
    loss = drift_loss
    # 2. Temporal Smoothing (The fix for Frame-wise)
    drift_smooth = 0.0
    if str(smooth_drift).lower() == 'true':
        V_seq = V_total.reshape(B, T, -1) # Put the Time dimension back
        # This penalizes sharp "jumps" in drift direction between frame t and t+1
        drift_smooth = FF.mse_loss(V_seq[:, 1:], V_seq[:, :-1])

        # 3. Combined Loss
        loss += 0.1 * drift_smooth    

    # --- Add PESQ / SISDR Loss ---
    pesq_loss_val = 0.0
    sisdr_loss_val = 0.0

    # --- Add PESQ / SISDR / Latent Drift / CCMSE Loss ---
    pesq_loss_val = 0.0
    sisdr_loss_val = 0.0
    latent_drift_loss_val = 0.0
    ccmse_loss_val = 0.0
    
    latent_drift_weight = config.get("latent_drift_weight", 0.0)
    ccmse_weight = config.get("ccmse_weight", 0.0)

    if config["pesq_weight"] > 0 or config["sisdr_weight"] > 0 or latent_drift_weight > 0 or ccmse_weight > 0:
        # Reconstruct Audio
        # x_gen_complex: (B, 1, F, T) -> (B, F, T)
        gen_audio = to_audio(x_gen_complex.squeeze(1), config)
        
        # Rescale gen_audio from normalized domain back to original amplitude.
        # to_audio inverts the spec_transform but NOT the waveform normfac
        # that was applied in the dataset. Multiply by normfac to align scales.
        # normfac: (B,) -> broadcast to (B, T)
        gen_audio = gen_audio * normfac.unsqueeze(1)
        
        # Clean Audio (already in original amplitude — raw waveform before /normfac)
        clean_audio = clean_audio_wav

        # ====================================================================
        # Renormalize to match input's maximum magnitude
        # ====================================================================
        # We process each sample in the batch individually
        # gen_audio: (B, T)
        # clean_audio: (B, T)

        
        # PESQ
        if config["pesq_weight"] > 0 and pesq_loss_fn is not None:
             p_loss = pesq_loss_fn(clean_audio, gen_audio)
             pesq_loss_val = config["pesq_weight"] * torch.mean(p_loss)
             loss = loss + pesq_loss_val

        # SISDR
        if config["sisdr_weight"] > 0 and sisdr_loss_fn is not None:
            # gen_audio: (B, T) -> (B, 1, T)
            # clean_audio: (B, T) -> (B, 1, T)
            s_loss = sisdr_loss_fn(gen_audio.unsqueeze(1), clean_audio.unsqueeze(1))
            sisdr_loss_val = config["sisdr_weight"] * torch.mean(s_loss)
            loss = loss + sisdr_loss_val

        # CCMSE — MultiResolution Complex Compressed MSE
        # Both gen_audio and clean_audio are in original amplitude domain here.
        if ccmse_weight > 0:
            ccmse_raw = compute_ccmse_loss(gen_audio, clean_audio)
            ccmse_loss_val = ccmse_weight * ccmse_raw
            loss = loss + ccmse_loss_val

        def normalize_audio(wav_tensor):
            """Normalizes audio to zero mean and unit variance per sample.
            Input: (B, T)
            Output: (B, T)
            """
            mean = wav_tensor.mean(dim=-1, keepdim=True)
            std = wav_tensor.std(dim=-1, keepdim=True)
            return (wav_tensor - mean) / (std + 1e-5)   
        # Latent Drift
        if latent_drift_weight > 0 and wavlm_model is not None:
             latent_temps = config.get("latent_temperatures", [0.01, 0.05, 0.1])            
             # WavLM expects (B, T) input
             # gen_audio: (B, T), clean_audio: (B, T)
             with torch.no_grad():
                 # Extract features (Clean)
                 # WavLM/Hubert output: (B, T_frames, D)
                 # WavLM/Hubert output: (B, T_frames, D)
                 latent_outputs_clean = wavlm_model(normalize_audio(clean_audio), output_hidden_states=True)
                 # feat_clean = latent_outputs_clean.last_hidden_state
             
             # For generated, we need gradients
             latent_outputs_gen = wavlm_model(normalize_audio(gen_audio), output_hidden_states=True)
             # feat_gen = latent_outputs_gen.last_hidden_state
             
             # Get layers to use
             wavlm_layers = config.get("feature_layers", [24]) # Default to last layer if not specified
            #  print('wavlm_layers:{}'.format(wavlm_layers))
             total_latent_loss_accum = 0.0
             total_latent_total_norm_accum = 0.0
             total_latent_pos_norm_accum = 0.0
             total_latent_neg_norm_accum = 0.0
             
             for layer_idx in wavlm_layers:
                 # Get features for the target layer
                 feat_clean = latent_outputs_clean.hidden_states[layer_idx]
                 feat_gen = latent_outputs_gen.hidden_states[layer_idx]
             
                 # Reuse normalization logic?
                 # WavLM features are (B, T, D). 
                 # We can treat T dimension as 'frames' similar to spectrogram frames, 
                 # but they are much fewer (20ms stride).
                 
                 # Need to normalize inputs to compute_drift_term or compute_V
                 # compute_drift_term expects (B, F, T, C) or similar structure if we use it.
                 # But here we have (B, T, D). Let's use drifting.normalize_features directly 
                 # or adapt to compute_drift_term's expected shape.
                 
                 # Latent Drift Method
                 latent_drift_method = config.get("latent_drift_method", "frame_level")
                 B_size, T_frames, D_dim = feat_gen.shape
    
                 f_gen_norm = None
                 f_pos_norm = None
                 
                 if latent_drift_method == "frame_level":
                     # Flatten to (B*T, D)
                     f_gen = feat_gen.reshape(B_size * T_frames, D_dim)
                     f_pos = feat_clean.reshape(B_size * T_frames, D_dim)
                     
                     # Normalize (Frame-level Global Scale)
                     with torch.no_grad():
                         frame_norms = torch.norm(f_pos, p=2, dim=1, keepdim=True)
                         global_scale_lat = frame_norms.mean().clamp(min=1e-5)
                         
                     f_gen_norm = f_gen / global_scale_lat
                     f_pos_norm = f_pos / global_scale_lat
    
                 elif latent_drift_method == "utterance_level":
                     # Flatten to (B, T*D)
                     f_gen = feat_gen.reshape(B_size, T_frames * D_dim)
                     f_pos = feat_clean.reshape(B_size, T_frames * D_dim)
                     
                     # Normalize (Utterance-level Per-Sample Scale)
                     with torch.no_grad():
                         utt_norms = torch.norm(f_pos, p=2, dim=1, keepdim=True)
                         global_scale_lat = utt_norms.clamp(min=1e-8)
                     
                     f_gen_norm = f_gen / global_scale_lat
                     f_pos_norm = f_pos / global_scale_lat
                     
                 else:
                     raise ValueError(f"Unknown latent_drift_method: {latent_drift_method}")
             
                 
                 # Call compute_V loop
                 V_lat_total = torch.zeros_like(f_gen_norm)
                 V_pos_total = torch.zeros_like(f_gen_norm)
                 V_neg_total = torch.zeros_like(f_gen_norm)
                 
                 # Positives = Clean, Negatives = Gen
                 y_pos_lat = f_pos_norm
                 y_neg_lat = f_gen_norm
                 
                 positive_drift_weight = config.get("Positive_drift_weight", 1.0)
                 negative_drift_weight = config.get("Negative_drift_weight", 1.0)

                 for tau in latent_temps:
                     V_tau, V_pos_tau, V_neg_tau = compute_V(
                         f_gen_norm,
                         y_pos_lat,
                         y_neg_lat,
                         tau,
                         mask_self=True,
                         return_components=True,
                         positive_drift_weight=positive_drift_weight,
                         negative_drift_weight=negative_drift_weight
                     )
                    #  v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
                    #  V_tau = V_tau / (v_norm + 1e-8)
                     V_lat_total += V_tau
                     V_pos_total += V_pos_tau
                     V_neg_total += V_neg_tau
                     
                 V_lat_total /= len(latent_temps)
                 V_pos_total /= len(latent_temps)
                 V_neg_total /= len(latent_temps)
                 
                 # Compute Loss
                 target_lat = (f_gen_norm + V_lat_total).detach()
                 l_drift = FF.mse_loss(f_gen_norm, target_lat)
                 
                 total_latent_loss_accum += l_drift
                 
                 total_latent_total_norm_accum += torch.sqrt(torch.mean(V_lat_total ** 2) + 1e-8).item()
                 total_latent_pos_norm_accum += torch.sqrt(torch.mean(V_pos_total ** 2) + 1e-8).item()
                 total_latent_neg_norm_accum += torch.sqrt(torch.mean(V_neg_total ** 2) + 1e-8).item()
             
             # Average over layers or Sum? "sum them" says the user prompt.
             # "extract multi-layer feature, and for each layer, comput the latent drift loss, and sum them"
             latent_drift_loss_val = latent_drift_weight * total_latent_loss_accum
             latent_total_norm_val = total_latent_total_norm_accum / len(wavlm_layers)
             latent_pos_norm_val = total_latent_pos_norm_accum / len(wavlm_layers)
             latent_neg_norm_val = total_latent_neg_norm_accum / len(wavlm_layers)
             loss += latent_drift_loss_val


    # 4. Update Model
    accumulate_grad_batches = config.get("accumulate_grad_batches", 1)
    
    # Scale loss
    loss = loss / accumulate_grad_batches
    loss.backward()

    # Step Optimizer
    if (batch_idx + 1) % accumulate_grad_batches == 0:
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["grad_clip"]
        )
        optimizer.step()
        optimizer.zero_grad()
    else:
        grad_norm = torch.tensor(0.0)

    # Metrics
    drift_norm = (V_total ** 2).mean().item() ** 0.5
    
    # Unscale loss for logging
    loss_val = loss.item() * accumulate_grad_batches
    
    return {
        "loss": loss_val,
        "drift_loss": drift_loss.item() if isinstance(drift_loss, torch.Tensor) else drift_loss,
        "drift_norm": drift_norm,
        "drift_smooth": drift_smooth.item() if isinstance(drift_smooth, torch.Tensor) else drift_smooth,
        "pesq": pesq_loss_val.item() if isinstance(pesq_loss_val, torch.Tensor) else pesq_loss_val,
        "sisdr": sisdr_loss_val.item() if isinstance(sisdr_loss_val, torch.Tensor) else sisdr_loss_val,
        "ccmse": ccmse_loss_val.item() if isinstance(ccmse_loss_val, torch.Tensor) else ccmse_loss_val,
        "latent_drift": latent_drift_loss_val.item() if isinstance(latent_drift_loss_val, torch.Tensor) else latent_drift_loss_val,
        "latent_total_norm": latent_total_norm_val if latent_drift_weight > 0 else 0.0,
        "latent_pos_norm": latent_pos_norm_val if latent_drift_weight > 0 else 0.0,
        "latent_neg_norm": latent_neg_norm_val if latent_drift_weight > 0 else 0.0,
        "grad_norm": grad_norm.item()
    }


def train(
    config_path: str,
    seed: int = 42,
    resume: Optional[str] = None,
    num_workers: int = 8,
    log_interval: int = 50,
    save_interval: int = 1,
):
    """Main training loop."""
    set_seed(seed)
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    wandb.init(
        project="drifting_se",
        config=config,
        dir=str(output_dir),
        name=f"se_drift_e{config['epochs']}" 
    )

    # 1. Dataset
    print(f"Loading dataset from {config['data_dir']}...")
    train_dataset = SpeechDataset(
        data_dir=config['data_dir'], 
        subset="train", 
        dummy=False, 
        shuffle_spec=True,
        num_frames=config["image_size"], # T=256
        return_waveform=True,
        noise_dir=config.get('noise_dir', "/home/liangxu/data/DEMAND_16k/"),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True, # Keeps workers alive between epochs (saves startup time)
        prefetch_factor=4,     # Buffer more batches        
    )

    # 1.5 Loss Functions
    pesq_loss_fn = None
    sisdr_loss_fn = None
    
    if config["pesq_weight"] > 0:
        print("Initializing PESQ Loss...")
        # Assuming 16kHz
        pesq_loss_fn = PesqLoss(1.0, sample_rate=16000).to(device).eval()
    
    if config["sisdr_weight"] > 0:
        print("Initializing SISDR Loss...")
        sisdr_loss_fn = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')

    # 1.6 Latent Model (WavLM or Hubert)
    wavlm_model = None
    if config.get("latent_drift_weight", 0) > 0:
        model_type = config.get("latent_model_type", "wavlm").lower()
        model_path = os.environ.get("WAVLM_LARGE_PATH", "./latent_ckpt/wavlm-large-local")
        
        try:
            if model_type == "hubert":
                model_path = os.environ.get("HUBERT_LARGE_PATH", "./latent_ckpt/hubert-large-local")
                wavlm_model = HubertModel.from_pretrained(model_path)
            elif model_type == "distillhubert":
                model_path = os.environ.get("DISTILHUBERT_PATH", "./latent_ckpt/distilhubert-local")
                wavlm_model = HubertModel.from_pretrained(model_path)
            else:
                # Default to WavLM
                wavlm_model = WavLMModel.from_pretrained(model_path)
            print(f"Initializing Latent Model: {model_type} from {model_path}...")
            wavlm_model.to(device)
            wavlm_model.eval()
            # Freeze 
            for param in wavlm_model.parameters():
                param.requires_grad = False
            print(f"{model_type} loaded and frozen.")
        except Exception as e:
            print(f"Failed to load Latent Model ({model_type}): {e}")
            print("Continuing without latent drift...")
            raise ValueError('Latent model load failed!')


    # 2. Model (NCSN++ v2)
    print(f"Creating model {config['model']} (NCSN++)...")

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

    # Initialize output layer to zero for stable potential initialization
    # Ensures gen starts near 0, making initial distance to target ~1.0 (Unit Sphere)
    # nn.init.xavier_uniform_(model.output_layer.weight) # Let the model start with some signal
    # nn.init.zeros_(model.output_layer.weight)
    # if model.output_layer.bias is not None:
        # nn.init.zeros_(model.output_layer.bias)

    print("\n" + "="*100)
    print("MODEL SUMMARY")
    print("="*100)

    # Create dummy input matching your data format
    # Input: x (noisy+noise), y (noisy condition), t (time)
    batch_size = config["batch_size"]
    dummy_x = torch.randn(batch_size, 1, config["image_size"], config["image_size"], 
                        dtype=torch.complex64, device=device)
    dummy_t = torch.ones(batch_size, device=device)

    summary(
        model, 
        input_data=[dummy_x, dummy_t],
        # col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=0,
        verbose=0
    )

    print("="*100 + "\n")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # EMA
    ema = EMA(model, decay=config["ema_decay"])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = WarmupLRScheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        base_lr=config["lr"],
    )

    # Resume
    start_epoch = 0
    global_step = 0
    if resume:
        checkpoint = load_checkpoint(resume, model, ema, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["step"]
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Training Loop
    print(f"\nStarting training for {config['epochs']} epochs...")

    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            info = train_step(model, optimizer, batch, batch_idx, config, device, pesq_loss_fn, sisdr_loss_fn, wavlm_model)
            
            
            ema.update(model)
            scheduler.step()
            
            epoch_loss += info["loss"]
            num_batches += 1
            global_step += 1
            
            # Log to WandB
            lr = scheduler.get_lr()
            wandb.log({
                "loss": info['loss'],
                "drift_loss": info['drift_loss'],
                "latent_drift": info['latent_drift'],                
                "latent_total_norm": info.get('latent_total_norm', 0.0),
                "latent_pos_norm": info.get('latent_pos_norm', 0.0),
                "latent_neg_norm": info.get('latent_neg_norm', 0.0),
                "drift_norm": info['drift_norm'],
                "drift_smooth": info['drift_smooth'],
                "ccmse": info['ccmse'],                
                "pesq": info['pesq'],
                "sisdr": info['sisdr'],
                "grad_norm": info['grad_norm'],
                "lr": lr,
                "epoch": epoch
            }, step=global_step)
            # Update pbar
            pbar.set_postfix({
                "Loss": f"{info['loss']:.6f}",
                "drift": f"{info['drift_loss']:.6f}",
                "latent": f"{info['latent_drift']:.6f}",                
                "ccmse": f"{info['ccmse']:.6f}",                
                "L_norm": f"{info.get('latent_total_norm', 0.0):.4f}",
                "P_norm": f"{info.get('latent_pos_norm', 0.0):.4f}",
                "N_norm": f"{info.get('latent_neg_norm', 0.0):.4f}",
                "d_norm": f"{info['drift_norm']:.6f}",
                "d_smt": f"{info['drift_smooth']:.6f}",
                "sisdr": f"{info['sisdr']:.6f}",                
                "pesq": f"{info['pesq']:.6f}"
            })

        # Handle last batch if not perfectly divisible
        if (batch_idx + 1) % config.get("accumulate_grad_batches", 1) != 0:
             optimizer.step()
             optimizer.zero_grad()
        avg_loss = epoch_loss / max(num_batches, 1)
        
        print(f"Epoch {epoch+1} | {time.time()-epoch_start:.1f}s | Loss: {avg_loss:.8f} | L: {info['latent_drift']:.8f} | CCMSE: {info['ccmse']:.6f} | P: {info['pesq']:.5f} | S: {info['sisdr']:.5f}")

        wandb.log({"epoch_loss": avg_loss, 
        "drift_norm": info['drift_norm'],
        "ccmse": info['ccmse'],        
        "pesq": info['pesq'],
        "sisdr": info['sisdr'],
        "latent_drift": info['latent_drift'],
        "latent_total_norm": info.get('latent_total_norm', 0.0),
        "latent_pos_norm": info.get('latent_pos_norm', 0.0),
        "latent_neg_norm": info.get('latent_neg_norm', 0.0),
        "grad_norm": info['grad_norm'],
        "lr": lr,
        "epoch": epoch
        }, step=global_step)

        # Always save last.ckpt (every epoch)
        last_ckpt_path = output_dir / "last.ckpt"
        save_checkpoint(
            str(last_ckpt_path),
            model,
            ema,
            optimizer,
            scheduler,
            epoch,
            global_step,
            config,
        )
        # print(f"Saved checkpoint to {last_ckpt_path}")

        # Save numbered checkpoint at intervals
        if (epoch + 1) % save_interval == 0 or epoch + 1 == config["epochs"]:
            ckpt_path = output_dir / f"epoch{epoch+1}.ckpt"
            save_checkpoint(
                str(ckpt_path),
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
            )
            print(f"Saved milestone checkpoint to {ckpt_path}")


    print("Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
    args = parser.parse_args()
    
    config = load_config(args.config)

    train(
        config_path=args.config,
        resume=args.resume,
    )
