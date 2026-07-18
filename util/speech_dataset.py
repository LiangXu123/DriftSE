import os
from glob import glob
from os.path import join
import torch
from torch.utils.data import Dataset
from torchaudio import load
import numpy as np
import torch.nn.functional as F
import random

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(
            f"Window type {window_type} not implemented!")


class SpeechDataset(Dataset):
    def __init__(self, data_dir, subset, dummy=False, shuffle_spec=False, num_frames=256,
                 format='default', normalize="noisy", spec_transform=None,
                 stft_kwargs=None, return_waveform=False, **kwargs):

        self.clean_dir = kwargs.get('clean_dir', "/home/liangxu/data/voicebank/true_split_train_val_test/16K/train/clean/")
        self.noise_dir = kwargs.get('noise_dir', "/home/liangxu/data/DEMAND_16k/")
        
        # Read file paths
        # Clean speech: VoiceBank
        self.clean_files = sorted(glob(join(self.clean_dir, "*.wav")))
        
        # Noisy speech: DEMAND (Recursive search)
        self.noise_files = sorted(glob(join(self.noise_dir, "**", "*.wav"), recursive=True))

        if len(self.clean_files) == 0:
             print(f"WARNING: No clean files found in {self.clean_dir}")
        if len(self.noise_files) == 0:
             print(f"WARNING: No noise files found in {self.noise_dir}")

        if dummy:
            self.clean_files = self.clean_files[:100]
            self.noise_files = self.noise_files[:100]

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.return_waveform = return_waveform
        
        # Default spec transform: Power compression + Scaling
        # To match sgmse/data_module.py logic
        self.spec_abs_exponent = 0.5
        self.spec_factor = 0.15
        
        # Default stft kwargs
        if stft_kwargs is None:
            # Matches sgmse/data_module.py Defaults
            # n_fft=510 gives 256 bins
            self.stft_kwargs = {
                "n_fft": 510,
                "hop_length": 128,
                "window": get_window("hann", 510),
                "center": True,
                "return_complex": True
            }
        else:
            self.stft_kwargs = stft_kwargs
            # Ensure window is tensor if string
            if isinstance(self.stft_kwargs.get("window"), str):
                 self.stft_kwargs["window"] = get_window(
                     self.stft_kwargs["window"], self.stft_kwargs["n_fft"]
                 )

    @staticmethod
    def mix_noise(clean, noise, snr_db):
        """
        clean: (T,) tensor
        noise: (T,) tensor  — trim/tile to match clean length
        snr_db: target SNR in dB (e.g. 0, 5, 10, 15, 20)
        """
        # Match lengths
        if noise.shape[0] < clean.shape[0]:
            # Tile noise if too short
            repeats = (clean.shape[0] // noise.shape[0]) + 1
            noise = noise.repeat(repeats)
            noise = noise[:clean.shape[0]]
        elif noise.shape[0] > clean.shape[0]:
            # Randomly crop noise if too long
            start = np.random.randint(0, noise.shape[0] - clean.shape[0] + 1)
            noise = noise[start:start+clean.shape[0]]
        else:
            # Exact match
            noise = noise 


        # Compute power
        clean_power = clean.pow(2).mean()
        noise_power = noise.pow(2).mean()

        # Scale noise to achieve target SNR
        # SNR = 10 * log10(P_clean / P_noise)
        # => P_noise_target = P_clean / 10^(SNR/10)
        target_noise_power = clean_power / (10 ** (snr_db / 10))
        noise_scale = torch.sqrt(target_noise_power / (noise_power + 1e-8))
        
        noisy = clean + noise_scale * noise
        return noisy

    def spec_transform_fn(self, spec):
         # Power compression (0.5) and scaling (0.15)
         spec = spec.abs()**self.spec_abs_exponent * torch.exp(1j * spec.angle())
         spec = spec * self.spec_factor
         return spec


    def __getitem__(self, i):
        # 1. Load Clean Speech
        x, _ = load(self.clean_files[i])
        
        # 2. Load Random Noise
        noise_idx = np.random.randint(0, len(self.noise_files))
        n, _ = load(self.noise_files[noise_idx])

        # Ensure single channel (C=1) - Squeeze or Select
        if x.dim() == 2:
            x = x[0]
        if n.dim() == 2:
            n = n[0]
            
        # 3. Dynamic Mixing
        # SNRs {0, 5, 10, 15} dB as requested or 0-20 random?
        # User said: "Example: mix at random SNR between 0–20 dB" AND "mixed with VoiceBank clean speech at SNRs {0, 5, 10, 15} dB".
        # I will use the code snippet they gave: snr = torch.randint(0, 21, (1,)).item()
        # snr = torch.randint(0, 21, (1,)).item()
        snr = random.choice([0, 5, 10, 15])
        
        # We need to mix BEFORE padding/cutting to ensure consistency?
        # Actually mixing assumes full length usually. 
        # But for training we usually cut to a segment.
        # If we cut first, we might cut silent part of noise.
        # It is better to mix then cut or cut then mix?
        # The provided mix_noise tiles noise to match clean.
        # Let's use the provided logic: mix_noise(clean, noise).
        y = self.mix_noise(x, n, snr)

        # 4. Cut/Pad to fixed segment
        # Formula applies for center=True
        target_len = (self.num_frames - 1) * self.stft_kwargs["hop_length"]
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        
        if pad == 0:
            # Extract random part
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            x = x[..., start:start+target_len]
            y = y[..., start:start+target_len]
        else:
            # Pad
            x = F.pad(x, (pad//2, pad//2+(pad % 2)), mode='constant')
            y = F.pad(y, (pad//2, pad//2+(pad % 2)), mode='constant')

        # Normalize waveform
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        else:
            normfac = 1.0
        
        xx = x
        normfac_tensor = torch.tensor(normfac if isinstance(normfac, float) else normfac.item(), dtype=torch.float32)
        x = x / (normfac + 1e-8)
        y = y / (normfac + 1e-8)

        # STFT
        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)

        # Transform
        X = self.spec_transform_fn(X)
        Y = self.spec_transform_fn(Y)
        
        # Convert Complex to Real (C, H, W) -> (2, F, T)
        if X.is_complex():
            X = torch.view_as_real(X)
            Y = torch.view_as_real(Y)
            
        # X: (F, T, 2)
        X = X.permute(2, 0, 1) # (2, F, T)
        Y = Y.permute(2, 0, 1) # (2, F, T)

        # X is Clean (Target), Y is Noisy (Condition)
        if self.return_waveform:
            return X, Y, xx, normfac_tensor
        return X, Y

    def __len__(self):
        if self.dummy:
            return min(len(self.clean_files), 100)
        return len(self.clean_files)
