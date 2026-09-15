import os
import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np
from scipy.stats import truncnorm
from typing import Optional
from util.drifting import compute_V, compute_V_paired

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
    spec_factor = config["spec_factor"]
    spec_abs_exponent = config.get("spec_abs_exponent", 0.5)
    
    mag = spec.abs()
    phase = spec.angle()
    
    mag_orig = (mag / spec_factor) ** (1.0 / spec_abs_exponent)
    spec_orig = mag_orig * torch.exp(1j * phase)
    
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    window = get_window(config["window_type"], n_fft).to(spec.device)
    
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
    """
    total = 0.0
    device = gen_wav.device

    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, periodic=True).to(device)

        S_gen   = torch.stft(gen_wav,   n_fft=n_fft, hop_length=hop,
                             window=win, center=True, return_complex=True)
        S_clean = torch.stft(clean_wav, n_fft=n_fft, hop_length=hop,
                             window=win, center=True, return_complex=True)

        diff_sq   = (S_gen - S_clean).abs().pow(2).sum(dim=(-2, -1)).mean()
        denom     = S_clean.abs().pow(2).sum(dim=(-2, -1)).mean() + eps

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
        a = (np.log(sigma_min) - mean) / std
        b = (np.log(sigma_max) - mean) / std
        log_sigma = truncnorm.rvs(a, b, loc=mean, scale=std, size=batch_size)
        log_sigma_tensor = torch.from_numpy(log_sigma).float().to(device)
        t_noise = torch.exp(log_sigma_tensor)
        
    elif noise_schedule == 'cosine':
        u = torch.rand(batch_size, device=device)
        s = 0.008
        t_noise = (torch.cos(((u + s) / (1 + s)) * np.pi / 2)) ** 2 * sigma_max
    elif noise_schedule == 'linear':
        t_noise = torch.rand(batch_size, device=device) * sigma_max
    else:
        raise ValueError(f"Unknown noise schedule: {noise_schedule}")

    t_noise = t_noise.clamp(min=1e-5, max=sigma_max)
    return t_noise

def compute_latent_drift_loss(
    gen_audio: torch.Tensor,
    clean_audio_wav: torch.Tensor,
    wavlm_model: nn.Module,
    config: dict,
    latent_memory_banks: Optional[dict] = None,
    batch_idx: int = 0,
    norm_input_audio: bool = True,
    gen_audio_neg: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes latent drift loss using WavLM features.
    Returns:
        latent_drift_loss, latent_total_norm, latent_pos_norm
    """
    latent_drift_weight = config.get("latent_drift_weight", 0.0)
    if latent_drift_weight <= 0 or wavlm_model is None:
        return torch.tensor(0.0, device=gen_audio.device), 0.0, 0.0
        
    def normalize_audio(wav_tensor):
        mean = wav_tensor.mean(dim=-1, keepdim=True)
        std = wav_tensor.std(dim=-1, keepdim=True)
        return (wav_tensor - mean) / (std + 1e-5)   
    
    latent_temps = config.get("latent_temperatures", [0.01, 0.05, 0.1])
    if "feature_layers" in config:
        wavlm_layers = config["feature_layers"]
        wavlm_layer_weights = {layer: 1.0 for layer in wavlm_layers}
    elif "scales" in config:
        wavlm_layers = [sc["layer"] for sc in config["scales"]]
        wavlm_layer_weights = {sc["layer"]: sc.get("weight", 1.0) for sc in config["scales"]}
    else:
        wavlm_layers = [24]
        wavlm_layer_weights = {24: 1.0}
    
    compute_1_on_1_drift = config.get("compute_1_on_1_drift", False)
    latent_drift_method = config.get("latent_drift_method", "frame_level")
    buffer_length = config.get("buffer_length", 0)

    with torch.no_grad():
        clean_in = normalize_audio(clean_audio_wav) if norm_input_audio else clean_audio_wav
        latent_outputs_clean = wavlm_model(clean_in, output_hidden_states=True)
        
    gen_in = normalize_audio(gen_audio) if norm_input_audio else gen_audio
    latent_outputs_gen = wavlm_model(gen_in, output_hidden_states=True)
    
    if gen_audio_neg is not None:
        with torch.no_grad():
            gen_in_neg = normalize_audio(gen_audio_neg) if norm_input_audio else gen_audio_neg
            latent_outputs_gen_neg = wavlm_model(gen_in_neg, output_hidden_states=True)
    else:
        latent_outputs_gen_neg = None
    
    total_latent_loss_accum = 0.0
    total_latent_total_norm_accum = 0.0
    total_latent_pos_norm_accum = 0.0
    
    for layer_idx in wavlm_layers:
        layer_weight = wavlm_layer_weights.get(layer_idx, 1.0)
        if layer_idx >= len(latent_outputs_clean.hidden_states):
            max_idx = len(latent_outputs_clean.hidden_states) - 1
            print(f"Warning: Layer {layer_idx} out of range for model (max {max_idx}). Falling back to layer {max_idx}.")
            layer_idx = max_idx
        feat_clean = latent_outputs_clean.hidden_states[layer_idx]
        feat_gen = latent_outputs_gen.hidden_states[layer_idx]
        if latent_outputs_gen_neg is not None:
            feat_gen_neg = latent_outputs_gen_neg.hidden_states[layer_idx]
        else:
            feat_gen_neg = None
        
        B_size, T_frames, D_dim = feat_gen.shape
        
        if latent_drift_method == "frame_level":
            chunk_sizes = config.get("chunk_sizes", None)
            if chunk_sizes is None:
                chunk_sizes = [config.get("chunk_size", 1)]
            elif not isinstance(chunk_sizes, list):
                chunk_sizes = [chunk_sizes]
        elif latent_drift_method == "utterance_level":
            chunk_sizes = ["utt"]
        else:
            raise ValueError(f"Unknown latent_drift_method: {latent_drift_method}")

        layer_latent_loss_accum = 0.0
        layer_total_norm_accum = 0.0
        layer_pos_norm_accum = 0.0
            
        for chunk_size in chunk_sizes:
            f_gen_norm = None
            f_pos_norm = None
            
            if latent_drift_method == "frame_level":
                if chunk_size > 1:
                    num_chunks = T_frames // chunk_size
                    # Truncate time dimension to fit full chunks
                    f_gen_trunc = feat_gen[:, :num_chunks * chunk_size, :]
                    f_pos_trunc = feat_clean[:, :num_chunks * chunk_size, :]
                    
                    f_gen = f_gen_trunc.reshape(B_size * num_chunks, chunk_size * D_dim)
                    f_pos = f_pos_trunc.reshape(B_size * num_chunks, chunk_size * D_dim)
                else:
                    f_gen = feat_gen.reshape(B_size * T_frames, D_dim)
                    f_pos = feat_clean.reshape(B_size * T_frames, D_dim)
                
                with torch.no_grad():
                    frame_norms = torch.norm(f_pos, p=2, dim=1, keepdim=True)
                    global_scale_lat = frame_norms.mean().clamp(min=1e-5)
                    
                f_gen_norm = f_gen / global_scale_lat
                f_pos_norm = f_pos / global_scale_lat

            elif latent_drift_method == "utterance_level":
                f_gen = feat_gen.reshape(B_size, T_frames * D_dim)
                f_pos = feat_clean.reshape(B_size, T_frames * D_dim)
                
                with torch.no_grad():
                    utt_norms = torch.norm(f_pos, p=2, dim=1, keepdim=True)
                    global_scale_lat = utt_norms.clamp(min=1e-8)
                
                f_gen_norm = f_gen / global_scale_lat
                f_pos_norm = f_pos / global_scale_lat
                
            V_lat_total = torch.zeros_like(f_gen_norm)
            V_pos_total = torch.zeros_like(f_gen_norm)
            V_neg_total = torch.zeros_like(f_gen_norm)
            
            mb_key = f"main_L{layer_idx}_C{chunk_size}"
            
            # Determine negative features for this batch
            if feat_gen_neg is not None:
                if latent_drift_method == "frame_level":
                    f_gen_neg = feat_gen_neg.reshape(B_size * T_frames, D_dim)
                else:
                    f_gen_neg = feat_gen_neg.reshape(B_size, T_frames * D_dim)
                batch_neg_lat_detached = (f_gen_neg / global_scale_lat).detach()
                batch_neg_lat = batch_neg_lat_detached
            else:
                batch_neg_lat_detached = f_gen_norm.detach()
                batch_neg_lat = f_gen_norm

            if latent_memory_banks is not None and buffer_length > 0:
                if mb_key not in latent_memory_banks["pos"]:
                    latent_memory_banks["pos"][mb_key] = f_pos_norm.detach()
                    latent_memory_banks["neg"][mb_key] = batch_neg_lat_detached
                else:
                    latent_memory_banks["pos"][mb_key] = torch.cat(
                        [latent_memory_banks["pos"][mb_key], f_pos_norm.detach()], dim=0
                    )[-buffer_length:]
                    latent_memory_banks["neg"][mb_key] = torch.cat(
                        [latent_memory_banks["neg"][mb_key], batch_neg_lat_detached], dim=0
                    )[-buffer_length:]
                y_pos_lat = latent_memory_banks["pos"][mb_key]
                y_neg_lat = latent_memory_banks["neg"][mb_key]
            else:
                y_pos_lat = f_pos_norm
                y_neg_lat = batch_neg_lat
                
            if batch_idx % 2000 == 0 and layer_idx == wavlm_layers[0] and (chunk_size == chunk_sizes[0]):
                with torch.no_grad():
                    subset_gen_lat = f_gen_norm[:100]
                    subset_pos_lat = f_pos_norm[:100]
                    dists_lat = torch.cdist(subset_gen_lat, subset_pos_lat, p=2)
                    
                    log_strs = []
                    for t in latent_temps:
                        logits = -dists_lat / (t + 1e-8)
                        probs = torch.softmax(logits, dim=1)
                        
                        max_p = probs.max(dim=1)[0].mean().item()
                        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10), dim=1)).mean().item()
                        
                        status = "Good"
                        if max_p > 0.9: status = "Too Cold"
                        elif max_p < (1.5 / subset_pos_lat.shape[0]): status = "Too Hot"
                        
                        log_strs.append(f"T={t}: MaxP={max_p:.4f}, Pplx={perplexity:.1f}/{subset_pos_lat.shape[0]} ({status})")
                    print(f"[DEBUG LATENT L{layer_idx} C{chunk_size}] " + " | ".join(log_strs))
                
            for tau in latent_temps:
                if compute_1_on_1_drift:
                    V_tau, V_pos_tau, V_neg_tau = compute_V_paired(
                        f_gen_norm,
                        y_pos_lat,
                        y_neg_lat,
                        tau,
                        mask_self=True,
                        return_components=True,
                        chunksize=chunk_size,
                    )
                else:
                    V_tau, V_pos_tau, V_neg_tau = compute_V(
                        f_gen_norm,
                        y_pos_lat,
                        y_neg_lat,
                        tau,
                        mask_self=True,
                        return_components=True,
                        chunksize=chunk_size,
                    )
                V_lat_total += V_tau
                V_pos_total += V_pos_tau
                V_neg_total += V_neg_tau
                
            V_lat_total /= len(latent_temps)
            V_pos_total /= len(latent_temps)
            V_neg_total /= len(latent_temps)
            
            target_lat = (f_gen_norm + V_lat_total).detach()
            l_drift = FF.mse_loss(f_gen_norm, target_lat)
            
            layer_latent_loss_accum += layer_weight * l_drift
            layer_total_norm_accum += torch.sqrt(torch.mean(V_lat_total ** 2) + 1e-8).item()
            layer_pos_norm_accum += torch.sqrt(torch.mean(V_pos_total ** 2) + 1e-8).item()

        # Average over chunk sizes
        num_scales = len(chunk_sizes)
        total_latent_loss_accum += layer_latent_loss_accum / num_scales
        total_latent_total_norm_accum += layer_total_norm_accum / num_scales
        total_latent_pos_norm_accum += layer_pos_norm_accum / num_scales
    
    latent_drift_loss_val = latent_drift_weight * (total_latent_loss_accum / len(wavlm_layers))
    latent_total_norm_val = total_latent_total_norm_accum / len(wavlm_layers)
    latent_pos_norm_val = total_latent_pos_norm_accum / len(wavlm_layers)
    
    return latent_drift_loss_val, latent_total_norm_val, latent_pos_norm_val

# ═══════════════════════════════════════════════════════════════════════════
# Auxiliary Encoder (BEATs / PANNs) — dual-branch drifting
# ═══════════════════════════════════════════════════════════════════════════

def load_aux_encoder(aux_cfg: dict, device: torch.device) -> nn.Module:
    """
    Load and freeze the auxiliary encoder specified in aux_cfg.
    Supported types: "beats", "panns", "wavcubepro".
    Returns the raw PyTorch module (frozen).
    """
    import sys, importlib.util
    enc_type = aux_cfg.get("type", "").lower()

    if enc_type == "beats":
        beats_dir  = aux_cfg["ckpt_dir"]
        beats_ckpt = os.path.join(beats_dir, aux_cfg["ckpt_file"])
        if beats_dir not in sys.path:
            sys.path.insert(0, beats_dir)
        spec = importlib.util.spec_from_file_location(
            "BEATs", os.path.join(beats_dir, "BEATs.py")
        )
        beats_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(beats_module)
        ckpt = torch.load(beats_ckpt, map_location="cpu")
        cfg  = beats_module.BEATsConfig(ckpt["cfg"])
        model = beats_module.BEATs(cfg)
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        print(f"[AuxEncoder] BEATs loaded from {beats_ckpt}")
        return model

    elif enc_type == "panns":
        from panns_inference import AudioTagging
        ckpt_path = os.path.join(aux_cfg["ckpt_dir"], aux_cfg["ckpt_file"])
        at   = AudioTagging(checkpoint_path=ckpt_path, device=str(device))
        cnn14 = at.model.module if hasattr(at.model, "module") else at.model
        cnn14 = cnn14.to(device).eval()
        for p in cnn14.parameters():
            p.requires_grad = False
        print(f"[AuxEncoder] PANNs CNN14 loaded from {ckpt_path}")
        return cnn14

    elif enc_type == "wavcubepro":
        wavcube_dir = "/vol/liangxu-solar/exp_code/latent_ckpt/WavCube"
        if wavcube_dir not in sys.path:
            sys.path.insert(0, wavcube_dir)
        try:
            from vocos import Vocos
        except ImportError as e:
            print(f"Failed to import vocos: {e}")
            raise ValueError("vocos package is not installed or not in sys.path.")
            
        wavcube_config_path = aux_cfg.get("wavcubepro_config", os.path.join(wavcube_dir, "configs/WavCube-stage2.yaml"))
        wavcube_ckpt_path = aux_cfg.get("wavcubepro_ckpt", os.path.join(wavcube_dir, "WavCube-pro/checkpoints/vocos_checkpoint_epoch%3D34_step%3D200000_val_loss%3D3.2140.ckpt"))
        
        try:
            print(f"Initializing WavCube-pro model from config: {wavcube_config_path} and ckpt: {wavcube_ckpt_path}...")
            vocos_model = Vocos.from_config(wavcube_config_path)
            state_dict = torch.load(wavcube_ckpt_path, map_location="cpu")["state_dict"]
            vocos_model.load_state_dict(state_dict, strict=False)
            vocos_model = vocos_model.to(device).eval()
            
            # Freeze parameters
            for param in vocos_model.parameters():
                param.requires_grad = False
            print(f"[AuxEncoder] WavCube-pro loaded and frozen.")
            return vocos_model
        except Exception as e:
            print(f"Failed to load WavCube-pro: {e}")
            raise ValueError("WavCube-pro load failed!")

    else:
        raise ValueError(f"[AuxEncoder] Unknown type: '{enc_type}'. Use 'beats', 'panns', or 'wavcubepro'.")


def _aux_extract_features(
    audio: torch.Tensor,       # (B, T_samples) at 16 kHz
    aux_model: nn.Module,
    enc_type: str,             # "beats", "panns", or "wavcubepro"
    layer_indices: set,        # {6, 12}, {5, 6}, or {0}
    sr_aux: int,               # 16000 or 32000
    no_grad: bool = False,
) -> dict:
    """
    Run one forward pass of the aux encoder and return
    {layer_idx: Tensor[B, T_feat, D]}  via registered hooks/direct extraction.
    """
    import torchaudio
    captured = {}
    hooks    = []

    if enc_type == "beats":
        for li in layer_indices:
            mod_idx = li - 1  # config is 1-based, encoder.layers is 0-based
            def _make_hook(key):
                def _h(m, inp, out):
                    # out = (x, attn, pos_bias); x: [T, B, C]
                    captured[key] = out[0].transpose(0, 1)   # → [B, T, C]
                return _h
            hooks.append(
                aux_model.encoder.layers[mod_idx].register_forward_hook(_make_hook(li))
            )
        pad = torch.zeros(audio.shape[0], audio.shape[1],
                          dtype=torch.bool, device=audio.device)
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            aux_model.extract_features(audio, padding_mask=pad)

    elif enc_type == "panns":
        # PANNs layer 5 → conv_block5 (1024-d), layer 6 → conv_block6 (2048-d)
        block_map = {5: "conv_block5", 6: "conv_block6"}
        for li in layer_indices:
            blk = getattr(aux_model, block_map[li])
            def _make_panns_hook(key):
                def _h(m, inp, out):
                    # out: [B, C, T', F'] — average over freq axis → [B, C, T'] → [B, T', C]
                    captured[key] = out.mean(dim=-1).permute(0, 2, 1)
                return _h
            hooks.append(blk.register_forward_hook(_make_panns_hook(li)))
        # Resample 16 kHz → sr_aux (32 kHz for PANNs)
        audio_in = torchaudio.functional.resample(audio, 16000, sr_aux) \
                   if sr_aux != 16000 else audio
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            aux_model(audio_in)

    elif enc_type == "wavcubepro":
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            # Normalize audio (as WavCube-pro expects normalized inputs)
            mean = audio.mean(dim=-1, keepdim=True)
            std = audio.std(dim=-1, keepdim=True)
            audio_norm = (audio - mean) / (std + 1e-5)
            z_hat = aux_model.feature_extractor.inf_new(audio_norm)
            for li in layer_indices:
                captured[li] = z_hat

    for h in hooks:
        h.remove()
    return captured


def compute_aux_encoder_drift_loss(
    gen_audio: torch.Tensor,
    clean_audio_wav: torch.Tensor,
    aux_model: nn.Module,
    aux_cfg: dict,
    global_step: int = 0,
    batch_idx: int = 0,
    latent_memory_banks: Optional[dict] = None,
    gen_audio_neg: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, float, float]:
    """
    Dual-branch auxiliary encoder drifting loss.
    """
    enc_type  = aux_cfg.get("type", "").lower()
    drift_mode = aux_cfg.get("drift_mode", "frame")
    aux_weight = aux_cfg.get("latent_drift_weight", 0.5)
    if aux_weight <= 0 or aux_model is None:
        return torch.tensor(0.0, device=gen_audio.device), 0.0, 0.0

    scales = aux_cfg.get("scales", [])
    if not scales:
        return torch.tensor(0.0, device=gen_audio.device), 0.0, 0.0

    latent_temps  = aux_cfg.get("latent_temperatures", [0.005, 0.01])
    main_buffer_length = config.get("buffer_length", 0) if config else 0
    buffer_length = aux_cfg.get("buffer_length", main_buffer_length)
    sr_aux        = aux_cfg.get("sample_rate", 16000)
    layer_indices = {sc["layer"] for sc in scales}

    B, T_samples = gen_audio.shape

    if drift_mode == "frame":
        with torch.no_grad():
            fc = _aux_extract_features(clean_audio_wav, aux_model, enc_type,
                                       layer_indices, sr_aux, no_grad=True)
            if gen_audio_neg is not None:
                fg_neg_raw = _aux_extract_features(gen_audio_neg, aux_model, enc_type,
                                                   layer_indices, sr_aux, no_grad=True)
            else:
                fg_neg_raw = None
        fg = _aux_extract_features(gen_audio, aux_model, enc_type,
                                   layer_indices, sr_aux, no_grad=False)
        feats_clean = fc
        feats_gen   = fg
        feats_gen_neg = fg_neg_raw

    elif drift_mode == "utter":
        with torch.no_grad():
            fc = _aux_extract_features(clean_audio_wav, aux_model, enc_type,
                                       layer_indices, sr_aux, no_grad=True)
            if gen_audio_neg is not None:
                fg_neg_raw = _aux_extract_features(gen_audio_neg, aux_model, enc_type,
                                                   layer_indices, sr_aux, no_grad=True)
            else:
                fg_neg_raw = None
        fg = _aux_extract_features(gen_audio, aux_model, enc_type,
                                   layer_indices, sr_aux, no_grad=False)
        feats_clean = {li: f.mean(dim=1, keepdim=True) for li, f in fc.items()}
        feats_gen   = {li: f.mean(dim=1, keepdim=True) for li, f in fg.items()}
        if fg_neg_raw is not None:
            feats_gen_neg = {li: f.mean(dim=1, keepdim=True) for li, f in fg_neg_raw.items()}
        else:
            feats_gen_neg = None

    elif drift_mode == "chunk_1s":
        chunk_dur_s = aux_cfg.get("chunk_duration_s", 1.0)
        hop_dur_s   = chunk_dur_s - aux_cfg.get("chunk_overlap_s", 0.5)
        chunk_len   = int(chunk_dur_s * 16000)
        hop_len     = int(hop_dur_s   * 16000)
        feats_clean = {li: [] for li in layer_indices}
        feats_gen   = {li: [] for li in layer_indices}
        feats_gen_neg = {li: [] for li in layer_indices} if gen_audio_neg is not None else None
        start = 0
        while start + chunk_len <= T_samples:
            c_chunk = clean_audio_wav[:, start:start + chunk_len]
            g_chunk = gen_audio[:,        start:start + chunk_len]
            if gen_audio_neg is not None:
                gn_chunk = gen_audio_neg[:, start:start + chunk_len]
            with torch.no_grad():
                fc_c = _aux_extract_features(c_chunk, aux_model, enc_type,
                                             layer_indices, sr_aux, no_grad=True)
                if gen_audio_neg is not None:
                    fgn_c = _aux_extract_features(gn_chunk, aux_model, enc_type,
                                                  layer_indices, sr_aux, no_grad=True)
            fg_c = _aux_extract_features(g_chunk, aux_model, enc_type,
                                         layer_indices, sr_aux, no_grad=False)
            for li in layer_indices:
                feats_clean[li].append(fc_c[li].mean(dim=1, keepdim=True))
                feats_gen[li].append(fg_c[li].mean(dim=1, keepdim=True))
                if gen_audio_neg is not None:
                    feats_gen_neg[li].append(fgn_c[li].mean(dim=1, keepdim=True))
            start += hop_len
        if not feats_clean.get(next(iter(layer_indices)), []):
            return torch.tensor(0.0, device=gen_audio.device), 0.0, 0.0
        feats_clean = {li: torch.cat(v, dim=1) for li, v in feats_clean.items()}
        feats_gen   = {li: torch.cat(v, dim=1) for li, v in feats_gen.items()}
        if feats_gen_neg is not None:
            feats_gen_neg = {li: torch.cat(v, dim=1) for li, v in feats_gen_neg.items()}
    else:
        raise ValueError(f"[AuxEncoder] Unknown drift_mode: '{drift_mode}'")

    total_loss  = 0.0
    total_tnorm = 0.0
    total_pnorm = 0.0
    num_active  = 0
    debug_aux_strs = []

    for sc in scales:
        gate     = sc.get("gate_steps", 0)
        if global_step < gate:
            continue
        li   = sc["layer"]
        w_j  = sc.get("weight", 1.0)
        name = sc.get("name", f"layer{li}")
        if li not in feats_gen:
            continue

        feat_c = feats_clean[li]
        feat_g = feats_gen[li]
        Bs, T_eff, D = feat_g.shape

        f_pos_raw = feat_c.reshape(Bs * T_eff, D)
        f_gen_raw = feat_g.reshape(Bs * T_eff, D)
        
        with torch.no_grad():
            frame_norms = torch.norm(f_pos_raw, p=2, dim=1, keepdim=True)
            global_scale_lat = frame_norms.mean().clamp(min=1e-5)
            
        f_pos = f_pos_raw / global_scale_lat
        f_gen = f_gen_raw / global_scale_lat
        
        if feats_gen_neg is not None:
            feat_gn = feats_gen_neg[li]
            f_gen_neg_raw = feat_gn.reshape(Bs * T_eff, D)
            f_gen_neg = f_gen_neg_raw / global_scale_lat
        else:
            f_gen_neg = None

        mb_key = f"aux_{enc_type}_{name}"
        batch_neg_detached = f_gen_neg.detach() if f_gen_neg is not None else f_gen.detach()
        
        if latent_memory_banks is not None and buffer_length > 0:
            if mb_key not in latent_memory_banks["pos"]:
                latent_memory_banks["pos"][mb_key] = f_pos.detach()
                latent_memory_banks["neg"][mb_key] = batch_neg_detached
            else:
                latent_memory_banks["pos"][mb_key] = torch.cat(
                    [latent_memory_banks["pos"][mb_key], f_pos.detach()], dim=0
                )[-buffer_length:]
                latent_memory_banks["neg"][mb_key] = torch.cat(
                    [latent_memory_banks["neg"][mb_key], batch_neg_detached], dim=0
                )[-buffer_length:]
            y_pos = latent_memory_banks["pos"][mb_key]
            y_neg = latent_memory_banks["neg"][mb_key]
        else:
            y_pos = f_pos
            y_neg = f_gen_neg.detach() if f_gen_neg is not None else f_gen

        if batch_idx % 2000 == 0:
            with torch.no_grad():
                sub_g = f_gen[:100];  sub_p = f_pos[:100]
                dists = torch.cdist(sub_g, sub_p, p=2)
                log_strs = []
                for t in latent_temps:
                    probs  = torch.softmax(-dists / (t + 1e-8), dim=1)
                    max_p  = probs.max(dim=1)[0].mean().item()
                    pplx   = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10), dim=1)).mean().item()
                    status = "Good"
                    if max_p > 0.9: status = "Too Cold"
                    elif max_p < 1.5 / sub_p.shape[0]: status = "Too Hot"
                    log_strs.append(f"T={t}: MaxP={max_p:.4f} Pplx={pplx:.1f} ({status})")
                debug_aux_strs.append(f"L{li}: [" + " | ".join(log_strs) + "]")

        V_total = torch.zeros_like(f_gen)
        V_pos   = torch.zeros_like(f_gen)
        for tau in latent_temps:
            V_tau, V_pos_tau, V_neg_tau = compute_V(
                f_gen, y_pos, y_neg, tau,
                mask_self=True, return_components=True,
                chunksize=1,
            )
            V_total += V_tau
            V_pos   += V_pos_tau
        V_total /= len(latent_temps)
        V_pos   /= len(latent_temps)

        target     = (f_gen + V_total).detach()
        scale_loss = w_j * FF.mse_loss(f_gen, target)
        total_loss  += scale_loss
        total_tnorm += torch.sqrt(torch.mean(V_total ** 2) + 1e-8).item()
        total_pnorm += torch.sqrt(torch.mean(V_pos   ** 2) + 1e-8).item()
        num_active  += 1

    if debug_aux_strs:
        print(f"[DEBUG AUX {enc_type.upper()} {drift_mode}] " + " || ".join(debug_aux_strs))

    if num_active == 0:
        return torch.tensor(0.0, device=gen_audio.device), 0.0, 0.0

    return (
        aux_weight * (total_loss / num_active),
        total_tnorm / num_active,
        total_pnorm / num_active,
    )
