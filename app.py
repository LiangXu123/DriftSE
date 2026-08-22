import os
import sys
import json
import torch
import torchaudio
import librosa
import soundfile as sf
import gradio as gr
from huggingface_hub import hf_hub_download

# Attempt to import spaces for ZeroGPU support on HF
try:
    import spaces
    USING_ZERO_GPU = True
except ImportError:
    USING_ZERO_GPU = False

from backbones.ncsnpp_v2 import NCSNpp_v2
from backbones.ncsnpp_v2_drift import ncsnpp_v2_drift
from util.other import pad_spec, set_torch_cuda_arch_list
from util.calc_metrics import process_file_parallel

set_torch_cuda_arch_list()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_sr = 16000

# Model configurations mapping
MODELS = {
    "WavLM Drifting (3 Layers)": {
        "repo_id": "LIANGXU123/DriftSE",
        "ckpt_path": "logs/wavlm_three_layers_with_z/last.ckpt",
        "config_path": "config/with_z/v2_drift2_wavlm_three_layers.json"
    },
    "HuBERT Drifting (3 Layers)": {
        "repo_id": "LIANGXU123/DriftSE",
        "ckpt_path": "logs/hubert_three_layers_with_z/last.ckpt",
        "config_path": "config/with_z/v2_drift2_hubert_three_layers.json"
    },
    "DistillHuBERT Drifting (3 Layers)": {
        "repo_id": "LIANGXU123/DriftSE",
        "ckpt_path": "logs/distillhubert_three_layers_pesq_sisdr_ccmse_with_z/last.ckpt",
        "config_path": "config/with_z/v2_drift2_distillhubert_three_layers_pesq_sisdr_ccmse.json"
    }
}

# Cache for loaded models
loaded_models = {}

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

def get_model(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    print(f"Loading {model_name}...")
    info = MODELS[model_name]
    
    # Download checkpoint if not exists
    if not os.path.exists(info["ckpt_path"]):
        print(f"Downloading checkpoint for {model_name}...")
        hf_hub_download(repo_id=info["repo_id"], filename=info["ckpt_path"], local_dir=".")
        
    config = load_config(info["config_path"])
    
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

    checkpoint = torch.load(info["ckpt_path"], map_location=device)
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    loaded_models[model_name] = (model, config)
    return loaded_models[model_name]

def core_inference(noisy_audio_path, model_choice):
    model, config = get_model(model_choice)
    window = get_window(config["window_type"], config["n_fft"], device)
    
    # Load and resample noisy audio
    y, sr = torchaudio.load(noisy_audio_path)
    y = y[0].cpu().numpy()
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    y_tensor = torch.as_tensor(y).to(device)
    T_orig = len(y_tensor)
    
    norm_factor = y_tensor.abs().max() + 1e-8
    y_tensor = y_tensor / norm_factor
    
    Y = torch.stft(
        y_tensor, 
        n_fft=config["n_fft"], 
        hop_length=config["hop_length"], 
        window=window, 
        center=config["center"], 
        return_complex=True
    )
    
    Y_trans = spec_fwd(Y, config)
    Y_input = Y_trans.unsqueeze(0).unsqueeze(0)
    Y_input = pad_spec(Y_input, mode="zero_pad")
    
    batch_size = Y_input.shape[0]
    z = torch.randn_like(Y_input).to(device)
    t = torch.ones(batch_size, device=device)
    
    with torch.no_grad():
        train_add_gaussian = config.get('train_add_gaussian', True)
        if str(train_add_gaussian).lower() == 'true':
            sample = model(Y_input + 0.01*z, t)
        else:
            sample = model(Y_input, t)
            
    X_hat_trans = sample.squeeze(0).squeeze(0)
    X_hat = spec_bwd(X_hat_trans, config)
    
    x_hat = torch.istft(
        X_hat, 
        n_fft=config["n_fft"], 
        hop_length=config["hop_length"], 
        window=window, 
        center=config["center"],
        length=T_orig
    )
    
    max_val = x_hat.abs().max()
    if max_val > 1e-8:
        x_hat = x_hat / max_val * norm_factor
    else:
        x_hat = x_hat * norm_factor
        
    out_path = "enhanced_output.wav"
    sf.write(out_path, x_hat.cpu().numpy(), target_sr)
    return out_path

if USING_ZERO_GPU:
    core_inference = spaces.GPU(core_inference)

def process_audio(noisy_audio_path, clean_audio_path, model_choice):
    if noisy_audio_path is None:
        return None, "Please upload a noisy audio file."
    
    # Run the core inference (on GPU if available/configured)
    out_path = core_inference(noisy_audio_path, model_choice)
    
    metrics_out = ""
    if clean_audio_path is not None:
        args_tuple = ("eval", clean_audio_path, noisy_audio_path, out_path)
        # Process metrics on CPU
        row, err = process_file_parallel(args_tuple)
        if err:
            metrics_out = f"Error computing metrics: {err}"
        elif row:
            metrics_out = (
                f"**PESQ:** {row['pesq']:.3f}\n\n"
                f"**STOI:** {row['stoi']:.3f}\n\n"
                f"**ESTOI:** {row['estoi']:.3f}\n\n"
                f"**SI-SDR:** {row['si_sdr']:.3f}"
            )
    else:
        metrics_out = "No clean audio provided. Metrics were not computed."
        
    return out_path, metrics_out

# Gradio Interface
with gr.Blocks(title="DriftSE Speech Enhancement") as app:
    gr.Markdown("# DriftSE Speech Enhancement")
    gr.Markdown("Upload a noisy audio file to enhance it using latent drifting speech enhancement models. Optionally, upload the clean reference to compute evaluation metrics (PESQ, STOI, ESTOI, SI-SDR).")
    
    with gr.Row():
        with gr.Column():
            noisy_in = gr.Audio(type="filepath", label="Noisy Audio (Required)")
            clean_in = gr.Audio(type="filepath", label="Clean Reference Audio (Optional)")
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()), 
                value=list(MODELS.keys())[0], 
                label="Enhancement Model"
            )
            enhance_btn = gr.Button("Enhance", variant="primary")
            
        with gr.Column():
            enhanced_out = gr.Audio(label="Enhanced Audio")
            metrics_out = gr.Markdown(label="Evaluation Metrics")

    enhance_btn.click(
        fn=process_audio,
        inputs=[noisy_in, clean_in, model_choice],
        outputs=[enhanced_out, metrics_out]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
