
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
import pytorch_optimizer as optim 
import torch.nn.functional as FF
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import WavLMModel, HubertModel
from backbones.ncsnpp_v2_drift_input_condition import ncsnpp_v2_drift_input_condition
from backbones.tfgridnet import TFGridNet_Backbone
from backbones.TFGridNet_Causal import TFGridNet_Causal
from backbones.streaming_unet import CausalNCSNpp
from util.drifting import compute_V, compute_V_paired

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
import sys

# 1. Force Python's package system to pretend protobuf version is 5.29.6
try:
    import google.protobuf
    # Override the version attributes before wandb reads them
    google.protobuf.__version__ = "5.29.6"
    sys.modules['google.protobuf'].__version__ = "5.29.6"
except ImportError:
    pass

# 2. Now import wandb (it will see "5.29.6" and bypass the check)
import wandb
from tqdm import tqdm
import json
from scipy.stats import truncnorm
from util.config_loader import load_config

from util.latent_drifting import (
    get_window,
    to_audio,
    compute_ccmse_loss,
    get_noise_schedule,
    compute_latent_drift_loss,
    load_aux_encoder,
    compute_aux_encoder_drift_loss,
    _aux_extract_features
)


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
    latent_memory_banks: Optional[dict] = None,
    manifolds_dict: Optional[dict] = None,
    mcd_codebooks: Optional[dict] = None,
    global_step: int = 0,
    norm_input_audio: bool = True,
    aux_encoder_model: Optional[nn.Module] = None,
    aux_encoder_cfg: Optional[dict] = None,
) -> dict:
    """
    Single training step for Speech Enhancement Drifting (NCSN++).
    """
    model.train()
    
    # Unpack batch: Clean (X Target), Noisy (Y Condition)
    # Shapes: (B, 2, F, T) Real Tensor
    clean_speech, noisy_speech, clean_audio_wav, noisy_audio_wav, normfac = batch
    # clean_speech = clean_speech.to(device)
    noisy_speech = noisy_speech.to(device)
    clean_audio_wav = clean_audio_wav.to(device)
    noisy_audio_wav = noisy_audio_wav.to(device)
    normfac = normfac.to(device)  # (B,) scalar per sample

    batch_size = noisy_speech.shape[0]

    # 1. Prepare Inputs for NCSN++ (Complex Tensors)
    # (B, 2, F, T) Real -> Permute -> (B, F, T, 2) -> Complex (B, F, T) -> Unsqueeze -> (B, 1, F, T)
    # x_pos_complex = torch.view_as_complex(clean_speech.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)
    y_cond_complex = torch.view_as_complex(noisy_speech.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)

    # NCSN++ usually expects t in [0, 1] or continuous.

    # 2. Generate Samples (Predict Clean Speech)
    # NCSN++ Forward: (z, cond, t) -> x_gen_complex
    # Output shape: (B, 1, F, T) Complex
    # x_gen_complex = model(z_complex, y_cond_complex, t)
    train_add_gaussian = config.get('train_add_gaussian', False)
    use_strict_gaussian = config.get('use_strict_gaussian', False)
    # Either flag enables the two-input (x, y, t) forward signature
    use_cond_fwd = (
        config.get('use_conditional_backbone', False)
        or 'input_condition' in config['model'].lower()
        or config['model'].lower() == 'streamunet'
    )
    if str(train_add_gaussian).lower() == 'true':
        # Noise Schedule
        z_complex = torch.randn_like(y_cond_complex)
        t_noise = get_noise_schedule(config, batch_size, device)
        sigma_broadcast = t_noise.view(batch_size, 1, 1, 1)
        noisy_input = y_cond_complex + (sigma_broadcast * z_complex)
        if use_cond_fwd:
            # input_condition models forward(x, y, t): noise input + conditioner
            if use_strict_gaussian:
                x_gen_complex = model(sigma_broadcast * z_complex, y_cond_complex, torch.ones(batch_size, device=device), use_strict_gaussian=use_strict_gaussian)
            else:
                x_gen_complex = model(sigma_broadcast * z_complex, y_cond_complex, torch.ones(batch_size, device=device))
        else:
            x_gen_complex = model(noisy_input, torch.ones(batch_size, device=device))
    else:
        if use_cond_fwd:
            if use_strict_gaussian:
                x_gen_complex = model(y_cond_complex, y_cond_complex, torch.ones(batch_size, device=device), use_strict_gaussian=use_strict_gaussian)
            else:
                x_gen_complex = model(y_cond_complex, y_cond_complex, torch.ones(batch_size, device=device))
        else:
            x_gen_complex = model(y_cond_complex, torch.ones(batch_size, device=device))
        
    # Convert back to Real representation for Loss Calculation (Flattened)
    # (B, 1, F, T) Complex -> (B, F, T) -> (B, F, T, 2)
    x_gen_real = torch.view_as_real(x_gen_complex.squeeze(1)) # (B, F, T, 2)
    # clean_speech_flat = torch.view_as_real(x_pos_complex.squeeze(1)) # (B, F, T, 2)

    # Initialize loss as building block for other losses
    loss = 0.0

    # NaN Checks for base tensor
    if torch.isnan(x_gen_real).any():
        print("[ERROR] x_gen_real contains NaNs!")

    # --- Add MSE Loss on output spectrum ---
    mse_weight = config.get("MSE_weight", 0.0)
    mse_loss_val = 0.0
    if mse_weight > 0:
        x_pos_real = clean_speech.to(device).permute(0, 2, 3, 1).contiguous()
        mse_raw = torch.nn.functional.mse_loss(x_gen_real, x_pos_real)
        mse_loss_val = mse_weight * mse_raw
        loss += mse_loss_val

    # --- Add PESQ / SISDR / Latent Drift / CCMSE Loss ---
    pesq_loss_val = 0.0
    sisdr_loss_val = 0.0
    latent_drift_loss_val = 0.0
    aux_drift_loss_val = 0.0
    ccmse_loss_val = 0.0
    
    latent_drift_weight = config.get("latent_drift_weight", 0.0)
    ccmse_weight = config.get("ccmse_weight", 0.0)

    # Determine if auxiliary encoder or auxiliary FD requires audio reconstruction
    aux_drift_weight = 0.0
    if aux_encoder_cfg is not None:
        aux_drift_weight = aux_encoder_cfg.get("latent_drift_weight", 0.0)

    if (config.get("pesq_weight", 0) > 0 or 
        config.get("sisdr_weight", 0) > 0 or 
        latent_drift_weight > 0 or 
        ccmse_weight > 0 or 
        aux_drift_weight > 0):
        # Reconstruct Audio
        # x_gen_complex: (B, 1, F, T) -> (B, F, T)
        gen_audio = to_audio(x_gen_complex.squeeze(1), config)
        
        # Rescale gen_audio from normalized domain back to original amplitude.
        # to_audio inverts the spec_transform but NOT the waveform normfac
        # normfac: (B,) -> broadcast to (B, T)
        gen_audio = gen_audio * normfac.unsqueeze(1)
        # ====================================================================
        # Renormalize to match input's maximum magnitude
        # ====================================================================

        # PESQ
        if config["pesq_weight"] > 0 and pesq_loss_fn is not None:
             p_loss = pesq_loss_fn(clean_audio_wav, gen_audio)
             pesq_loss_val = config["pesq_weight"] * torch.mean(p_loss)
             loss += pesq_loss_val

        # SISDR
        if config["sisdr_weight"] > 0 and sisdr_loss_fn is not None:
            # gen_audio: (B, T) -> (B, 1, T)
            # clean_audio_wav: (B, T) -> (B, 1, T)
            s_loss = sisdr_loss_fn(gen_audio.unsqueeze(1), clean_audio_wav.unsqueeze(1))
            sisdr_loss_val = config["sisdr_weight"] * torch.mean(s_loss)
            loss += sisdr_loss_val

        # CCMSE — MultiResolution Complex Compressed MSE
        # Both gen_audio and clean_audio_wav are in original amplitude domain here.
        if ccmse_weight > 0:
            ccmse_raw = compute_ccmse_loss(gen_audio, clean_audio_wav)
            ccmse_loss_val = ccmse_weight * ccmse_raw
            loss += ccmse_loss_val


        # Latent Drift — dispatch to conditional or legacy path
        latent_drift_loss_val, latent_total_norm_val, latent_pos_norm_val = compute_latent_drift_loss(
            gen_audio=gen_audio,
            clean_audio_wav=clean_audio_wav,
            wavlm_model=wavlm_model,
            config=config,
            latent_memory_banks=latent_memory_banks,
            batch_idx=batch_idx,
            norm_input_audio=norm_input_audio,
            gen_audio_neg=None,
        )
        if isinstance(latent_drift_loss_val, torch.Tensor):
            loss += latent_drift_loss_val

        # Auxiliary encoder dual-branch drift loss
        if aux_encoder_model is not None and aux_encoder_cfg is not None:
            aux_drift_loss_val, aux_tnorm, aux_pnorm = compute_aux_encoder_drift_loss(
                gen_audio=gen_audio,
                clean_audio_wav=clean_audio_wav,
                aux_model=aux_encoder_model,
                aux_cfg=aux_encoder_cfg,
                global_step=global_step,
                batch_idx=batch_idx,
                latent_memory_banks=latent_memory_banks,
                gen_audio_neg=None,
                config=config,
            )
            if isinstance(aux_drift_loss_val, torch.Tensor):
                loss += aux_drift_loss_val

    # 4. Update Model
    accumulate_grad_batches = config.get("accumulate_grad_batches", 1)
    
    # Scale loss and backward if a tensor loss is active
    if isinstance(loss, torch.Tensor):
        loss = loss / accumulate_grad_batches
        loss.backward()
    else:
        # Avoid AttributeError: 'float' object has no attribute 'backward' if all loss weights are zero
        pass

    # Step Optimizer
    if isinstance(loss, torch.Tensor) and (batch_idx + 1) % accumulate_grad_batches == 0:
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["grad_clip"]
        )
        optimizer.step()
        optimizer.zero_grad()
    else:
        grad_norm = torch.tensor(0.0)

    # Unscale loss for logging
    loss_val = loss.item() * accumulate_grad_batches if isinstance(loss, torch.Tensor) else loss * accumulate_grad_batches
    
    return {
        "loss": loss_val,
        "mse": mse_loss_val.item() if isinstance(mse_loss_val, torch.Tensor) else mse_loss_val,
        "pesq": pesq_loss_val.item() if isinstance(pesq_loss_val, torch.Tensor) else pesq_loss_val,
        "sisdr": sisdr_loss_val.item() if isinstance(sisdr_loss_val, torch.Tensor) else sisdr_loss_val,
        "ccmse": ccmse_loss_val.item() if isinstance(ccmse_loss_val, torch.Tensor) else ccmse_loss_val,
        "latent_drift": latent_drift_loss_val.item() if isinstance(latent_drift_loss_val, torch.Tensor) else latent_drift_loss_val,
        "aux_drift": aux_drift_loss_val.item() if isinstance(aux_drift_loss_val, torch.Tensor) else aux_drift_loss_val
    }


def train(
    config_path: str,
    seed: int = 42,
    resume: Optional[str] = None,
    num_workers: int = 8,
    log_interval: int = 50,
    save_interval: int = 5,
):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    config = load_config(config_path)

    # print(f"Using device: {device}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    wandb.init(
        project="DriftSE",
        config=config,
        dir=str(output_dir),
        name=f"drift_sr_{os.path.basename(config['enhanced_dir'])}" 
    )

    # 1. Dataset
    dataset_kwargs = {
        'data_dir': None, 
        'subset': "train", 
        'dummy': False, 
        'shuffle_spec': True,
        'num_frames': config["image_size"], # T=256
        'return_waveform': True,
        'task': config.get('task', 'se')
    }
    
    # Only pass overrides if explicitly provided in the JSON
    for k in ['noise_dir', 'clean_dir', 'mix_clean_dir', 'use_paird_training', 'clean_data_source', 'noisy_data_source', 'mixture_clean_source', 'mix_noisy_on_the_fly', 'normalize']:
        if k in config:
            dataset_kwargs[k] = config[k]
            
    # Explicitly configure stft_kwargs from the JSON config to align hop_length and n_fft
    dataset_kwargs['stft_kwargs'] = {
        "n_fft": config.get("n_fft", 510),
        "hop_length": config.get("hop_length", 128),
        "window": config.get("window_type", "hann"),
        "center": config.get("center", True),
        "return_complex": True
    }
            
    train_dataset = SpeechDataset(**dataset_kwargs)
    
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

    # 1.6 Latent Model (WavLM, HuBERT, or WavCube-pro)
    wavlm_model = None
    if config.get("latent_drift_weight", 0) > 0:
        model_type = config.get("latent_model_type", "wavlm").lower()
        
        if "wavcubepro" in model_type:
            wavcube_dir = "/vol/liangxu-solar/exp_code/latent_ckpt/WavCube"
            if wavcube_dir not in sys.path:
                sys.path.insert(0, wavcube_dir)
            try:
                from vocos import Vocos
            except ImportError as e:
                print(f"Failed to import vocos: {e}")
                raise ValueError("vocos package is not installed or not in sys.path.")
                
            wavcube_config_path = config.get("wavcubepro_config", os.path.join(wavcube_dir, "configs/WavCube-stage2.yaml"))
            wavcube_ckpt_path = config.get("wavcubepro_ckpt", os.path.join(wavcube_dir, "WavCube-pro/checkpoints/vocos_checkpoint_epoch%3D34_step%3D200000_val_loss%3D3.2140.ckpt"))
            model_path = wavcube_ckpt_path
            
            try:
                print(f"Initializing WavCube-pro model from config: {wavcube_config_path} and ckpt: {wavcube_ckpt_path}...")
                vocos_model = Vocos.from_config(wavcube_config_path)
                state_dict = torch.load(wavcube_ckpt_path, map_location="cpu")["state_dict"]
                vocos_model.load_state_dict(state_dict, strict=False)
                vocos_model = vocos_model.to(device)
                vocos_model.eval()
                
                # Freeze parameters
                for param in vocos_model.parameters():
                    param.requires_grad = False
                print(f"WavCube-pro loaded and frozen.")
                
                # Wrap in compatibility class
                class WavCubeProWrapper(nn.Module):
                    def __init__(self, vocos):
                        super().__init__()
                        self.vocos = vocos
                        # WavCube-pro features dimension is 128
                        self.config = argparse.Namespace(hidden_size=128)
                        
                    def forward(self, audio, output_hidden_states=True):
                        fe = self.vocos.feature_extractor
                        fe.eval()
                        
                        z_hat = fe.inf_new(audio)
                        
                        class ModelOutput:
                            def __init__(self, hidden_states, last_hidden_state):
                                self.hidden_states = hidden_states
                                self.last_hidden_state = last_hidden_state
                                
                        return ModelOutput(hidden_states=[z_hat], last_hidden_state=z_hat)
                
                wavlm_model = WavCubeProWrapper(vocos_model)
            except Exception as e:
                print(f"Failed to load WavCube-pro: {e}")
                raise ValueError("WavCube-pro load failed!")
        elif model_type in ("panns", "beats"):
            # ── PANNs / BEATs as the main backbone ──────────────────────────
            # Hardcoded default paths (same pattern as wavcubepro above).
            # Any key can be overridden via config["latent_ckpt_dir"] /
            # config["latent_ckpt_file"] / config["latent_sample_rate"], or
            # via a nested config["main_encoder"] dict.
            _PANNS_DIR  = "/vol/liangxu-solar/exp_code/latent_ckpt/panns-local"
            _PANNS_FILE = "Cnn14_mAP=0.431.pth"
            _BEATS_DIR  = "/vol/liangxu-solar/exp_code/latent_ckpt/beats-local/"
            _BEATS_FILE = "BEATs_iter3_plus_AS2M.pt"

            if model_type == "panns":
                _default_dir  = _PANNS_DIR
                _default_file = _PANNS_FILE
                _default_sr   = 32000
            else:  # beats
                _default_dir  = _BEATS_DIR
                _default_file = _BEATS_FILE
                _default_sr   = 16000

            main_enc_cfg = config.get("main_encoder", {})
            if not main_enc_cfg:
                main_enc_cfg = {
                    "type":        model_type,
                    "ckpt_dir":    config.get("latent_ckpt_dir",    _default_dir),
                    "ckpt_file":   config.get("latent_ckpt_file",   _default_file),
                    "sample_rate": config.get("latent_sample_rate", _default_sr),
                }
            else:
                main_enc_cfg.setdefault("type",        model_type)
                main_enc_cfg.setdefault("ckpt_dir",    _default_dir)
                main_enc_cfg.setdefault("ckpt_file",   _default_file)
                main_enc_cfg.setdefault("sample_rate", _default_sr)

            layer_indices = {sc["layer"] for sc in config.get("scales", [])}
            sr_main = main_enc_cfg["sample_rate"]

            print(f"Initializing Latent Model: {model_type} "
                  f"from {main_enc_cfg['ckpt_dir']}/{main_enc_cfg['ckpt_file']} ...")
            _raw_main_enc = load_aux_encoder(main_enc_cfg, device)

            class AuxEncoderWrapper(nn.Module):
                """Wraps a PANNs-CNN14 or BEATs model into the backbone interface.

                Satisfies: model(audio, output_hidden_states=True) → obj with
                    obj.hidden_states  : list[Tensor(B, T, D)]  indexed by layer
                The list is sparse — only the indices requested in `layer_indices`
                are filled; all others are None.  compute_conditional_drift_loss
                only accesses indices listed in config["scales"], so this is safe.
                """
                def __init__(self, raw_model, enc_type, layer_indices, sr):
                    super().__init__()
                    self.raw_model     = raw_model
                    self.enc_type      = enc_type
                    self.layer_indices = layer_indices
                    self.sr            = sr
                    # Expose a dummy .config.hidden_size for any callers that inspect it
                    max_dim = 2048  # PANNs block6; BEATs is 768
                    self.config = argparse.Namespace(hidden_size=max_dim)

                def forward(self, audio, output_hidden_states=True):
                    feats = _aux_extract_features(
                        audio, self.raw_model, self.enc_type,
                        self.layer_indices, self.sr,
                        no_grad=not torch.is_grad_enabled(),
                    )
                    # Build a hidden_states list up to max(layer_indices)+1
                    max_idx = max(self.layer_indices) + 1
                    hs = [feats.get(i, None) for i in range(max_idx)]

                    class ModelOutput:
                        def __init__(self, hidden_states):
                            self.hidden_states    = hidden_states
                            self.last_hidden_state = hidden_states[-1] if hidden_states else None

                    return ModelOutput(hidden_states=hs)

            wavlm_model = AuxEncoderWrapper(_raw_main_enc, model_type, layer_indices, sr_main)
            print(f"[MainEncoder] {model_type.upper()} wrapped and ready (layers={sorted(layer_indices)}, sr={sr_main}).")

        else:
            # Determine Hugging Face model ID based on model_type
            if "distill" in model_type.lower() or "distil" in model_type.lower():
                model_path = "./latent_ckpt/distilhubert-local"
                model_cls = HubertModel
            elif "hubert" in model_type.lower():
                model_path = "./latent_ckpt/hubert-large-local"
                model_cls = HubertModel
            else:
                model_path = "./latent_ckpt/wavlm-large-local"
                model_cls = WavLMModel

            try:
                print(f"Initializing Latent Model: {model_type} from {model_path}...")
                wavlm_model = model_cls.from_pretrained(model_path)
                wavlm_model.to(device)
                wavlm_model.eval()
                # Freeze
                for param in wavlm_model.parameters():
                    param.requires_grad = False
                print(f"{model_type} loaded and frozen.")
            except Exception as e:
                print(f"Failed to load Latent Model ({model_type}) from {model_path}: {e}")
                raise ValueError('Latent model load failed!')

    # Determine if normalization is needed
    norm_input_audio = True
    if wavlm_model is not None:
        if model_type in ("panns", "beats"):
            # PANNs / BEATs handle their own resampling inside _aux_extract_features;
            # the wrapper should receive raw (possibly already-16kHz) audio — no
            # global normalisation needed here (PANNs expects raw waveform).
            norm_input_audio = False
            print(f"[*] Note: {model_type} manages its own pre-processing. Audio normalization is DISABLED.")
        elif 'wavlm-base' in model_type.lower() or 'distil' in model_type.lower():
            norm_input_audio = False
            print(f"[*] Note: {model_type} expects UNNORMALIZED audio inputs. Audio normalization is DISABLED.")
        else:
            print(f"[*] Note: {model_type} expects NORMALIZED audio inputs. Audio normalization is ENABLED.")

    # 2. Model (NCSN++ v2)
    print(f"Creating model {config['model']}...")

    if config['model'].lower() == 'ncsnpp_v2_drift_input_condition':
        model = ncsnpp_v2_drift_input_condition(
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
    elif config['model'].lower() == 'tfgridnet':
        model = TFGridNet_Backbone(
            n_layers          = config.get('n_layers', 5),
            emb_dim           = config.get('emb_dim', 32),
            lstm_hidden_units = config.get('lstm_hidden_units', 100),
            attn_n_head       = config.get('attn_n_head', 4),
            causal            = config.get('causal', False),
        ).to(device)
    elif config['model'].lower() == 'tfgridnet_causal':
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
    elif config['model'].lower() == 'streamunet':
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
        raise ValueError(f"Unknown model name: {config['model']}")

    # Create dummy input matching your data format
    # Input: x (noisy+noise), y (noisy condition), t (time)
    batch_size = config["batch_size"]
    dummy_x = torch.randn(batch_size, 1, config["image_size"], config["image_size"], 
                        dtype=torch.complex64, device=device)
    dummy_t = torch.ones(batch_size, device=device)


    # print("="*100 + "\n")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    if 'causal' in config['model'].lower() or config['model'].lower() == 'streamunet':
        print("\n" + "="*50)
        print("Executing Time Axis Causal Check...")
        model.eval()
        with torch.no_grad():
            B, C, F, T_steps = 4, 1, config["image_size"], 64
            dummy_x1 = torch.randn(B, C, F, T_steps, dtype=torch.complex64, device=device)
            dummy_y1 = torch.randn(B, C, F, T_steps, dtype=torch.complex64, device=device)
            dummy_t = torch.ones(B, device=device)

            # Use appropriate forward depending on model type
            use_cond_fwd = config.get('use_conditional_backbone', False) or 'input_condition' in config['model'].lower()
            if use_cond_fwd:
                out1 = model(dummy_x1, dummy_y1, dummy_t)
            else:
                out1 = model(dummy_x1, dummy_t)
                
            t_change = (T_steps // 2) + 1
            dummy_x2 = dummy_x1.clone()
            dummy_y2 = dummy_y1.clone()
            
            # Perturb future
            dummy_x2[:, :, :, t_change:] += torch.randn_like(dummy_x2[:, :, :, t_change:]) * 10.0
            dummy_y2[:, :, :, t_change:] += torch.randn_like(dummy_y2[:, :, :, t_change:]) * 10.0
            
            if use_cond_fwd:
                out2 = model(dummy_x2, dummy_y2, dummy_t)
            else:
                out2 = model(dummy_x2, dummy_t)
                
            diff_past = torch.mean(torch.abs(out1[:, :, :, :t_change] - out2[:, :, :, :t_change])).item()
            diff_future = torch.mean(torch.abs(out1[:, :, :, t_change:] - out2[:, :, :, t_change:])).item()
            
            if diff_past < 1e-5 and diff_future > 1e-5:
                print(f"Causality Check PASSED. (Diff past: {diff_past:.8f}, Diff future: {diff_future:.8f})")
            elif diff_past >= 1e-5:
                print(f"Causality Check FAILED! Information leaked to the past. (Diff past: {diff_past:.8f}, Diff future: {diff_future:.8f})")
            else:
                print(f"Causality Check FAILED! Network seems to ignore the input (Diff future < 1e-5). (Diff past: {diff_past:.8f}, Diff future: {diff_future:.8f})")
                
        # Re-enter train mode
        model.train()
        print("="*50 + "\n")

    # EMA
    ema = EMA(model, decay=config["ema_decay"])

    # Optimizer
    if config.get("SOAP", False):
        optimizer = optim.SOAP(
            model.parameters(),
            lr=config["lr"],
            betas=(0.95, 0.95),
            weight_decay=config["weight_decay"],
            precondition_frequency=5
        )
    else:
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

    # Initialize Memory Banks for Latent Drift
    latent_memory_banks = {
        'pos': {},
        'neg': {}
    }

    # Load Offline Codebooks (shared by manifold guidance and MCD)
    offline_codebooks = {}
    codebook_path = config.get("mcd_codebook_path", "./reverb/manifold_codebooks.pt")
    if config.get("use_manifold_guidance", False) or config.get("use_mcd", False):
        if os.path.exists(codebook_path):
            print(f"Loading Offline Codebooks from {codebook_path}...")
            offline_codebooks = torch.load(codebook_path, map_location=device)
            # Report which layers are available
            print(f"  -> Available layers in codebook: {sorted(offline_codebooks.keys())}")
        else:
            print(f"Warning: Codebook not found at {codebook_path}. Disabling manifold guidance and MCD.")
            config["use_manifold_guidance"] = False
            config["use_mcd"] = False

    # For backward compatibility, alias both as the same dict
    manifolds_dict = offline_codebooks  # keyed by layer_idx (int)
    mcd_codebooks = offline_codebooks

    # 1.7 Auxiliary encoder (BEATs / PANNs) for dual-branch drifting
    aux_encoder_model = None
    aux_encoder_cfg   = config.get("aux_encoder", None)
    if aux_encoder_cfg is not None:
        try:
            aux_encoder_model = load_aux_encoder(aux_encoder_cfg, device)
        except Exception as e:
            print(f"[AuxEncoder] WARNING: failed to load aux encoder: {e}")
            aux_encoder_model = None


    # Training Loop
    print(f"\nStarting training for {config['epochs']} epochs...")

    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            info = train_step(model, optimizer, batch, batch_idx, config, device, pesq_loss_fn, sisdr_loss_fn, wavlm_model, latent_memory_banks, manifolds_dict, mcd_codebooks, global_step=global_step, norm_input_audio=norm_input_audio, aux_encoder_model=aux_encoder_model, aux_encoder_cfg=aux_encoder_cfg)
            
            
            ema.update(model)
            scheduler.step()
            
            epoch_loss += info["loss"]
            num_batches += 1
            global_step += 1
            
            # Log to WandB
            lr = scheduler.get_lr()
            _postfix = {"Loss": f"{info['loss']:.8f}", "latent_drift": f"{info['latent_drift']:.8f}"}
            if config.get("MSE_weight", 0.0) > 0:
                _postfix["mse"] = f"{info['mse']:.8f}"
            if config.get("pesq_weight", 0.0) > 0:
                _postfix["pesq"] = f"{info['pesq']:.8f}"
            if config.get("sisdr_weight", 0.0) > 0:
                _postfix["sisdr"] = f"{info['sisdr']:.8f}"
            if config.get("ccmse_weight", 0.0) > 0:
                _postfix["ccmse"] = f"{info['ccmse']:.8f}"
            if aux_encoder_cfg is not None:
                _postfix["aux_drift"] = f"{info['aux_drift']:.8f}"   
            pbar.set_postfix(_postfix)

        # Handle last batch if not perfectly divisible
        if (batch_idx + 1) % config.get("accumulate_grad_batches", 1) != 0:
             optimizer.step()
             optimizer.zero_grad()
        avg_loss = epoch_loss / max(num_batches, 1)
        
        print_str = f"Epoch {epoch+1} | {time.time()-epoch_start:.1f}s | Loss: {avg_loss:.8f} | L: {info['latent_drift']:.8f}"
        if config.get("MSE_weight", 0.0) > 0:
            print_str += f" | MSE: {info['mse']:.8f}"
        if config.get("pesq_weight", 0.0) > 0:
            print_str += f" | PESQ: {info['pesq']:.8f}"
        if config.get("sisdr_weight", 0.0) > 0:
            print_str += f" | SISDR: {info['sisdr']:.8f}"
        if config.get("ccmse_weight", 0.0) > 0:
            print_str += f" | CCMSE: {info['ccmse']:.8f}"
        if aux_encoder_cfg is not None:
            print_str += f" | AuxDrift: {info['aux_drift']:.8f}"    
        print(print_str)

        wandb.log({"epoch_loss": avg_loss, 
        "pesq": info['pesq'],
        "sisdr": info['sisdr'],
        "ccmse": info.get('ccmse', 0.0),
        "latent_drift": info['latent_drift'],
        "aux_drift": info.get('aux_drift', 0.0),
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
