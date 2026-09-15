import os
from glob import glob
from os.path import join
import torch
from torch.utils.data import Dataset
from torchaudio import load
import numpy as np
import torch.nn.functional as F
import random
import pyroomacoustics as pra
from scipy.signal import fftconvolve

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(
            f"Window type {window_type} not implemented!")

def generate_random_rir(target_fs=16000):
    """Generates a random room impulse response using PyRoomAcoustics."""
    room_dim = np.array([
        np.random.uniform(5.0, 15.0), 
        np.random.uniform(5.0, 15.0), 
        np.random.uniform(2.0, 6.0)
    ])
    
    t60 = np.random.uniform(0.4, 1.0)
    e_absorption, max_order = pra.inverse_sabine(t60, room_dim)
    
    room = pra.ShoeBox(room_dim, fs=target_fs, materials=pra.Material(e_absorption), max_order=min(3, max_order))
    
    source_pos = np.array([
        np.random.uniform(1.0, room_dim[0] - 1.0),
        np.random.uniform(1.0, room_dim[1] - 1.0),
        np.random.uniform(1.0, room_dim[2] - 1.0)
    ])
    
    mic_pos = np.array([
        np.random.uniform(1.0, room_dim[0] - 1.0),
        np.random.uniform(1.0, room_dim[1] - 1.0),
        np.random.uniform(1.0, room_dim[2] - 1.0)
    ])
    
    room.add_source(source_pos)
    room.add_microphone(mic_pos)
    room.compute_rir()
    
    return room.rir[0][0]

class SpeechDataset(Dataset):
    def __init__(self, data_dir, subset, dummy=False, shuffle_spec=False, num_frames=256,
                 task='se', normalize="noisy", spec_transform=None,
                 stft_kwargs=None, return_waveform=False, **kwargs):
        self.task = task
        # --- Task-based auto-configuration ---
        if self.task == 'se':
            self.use_paird_training = True
            self.clean_data_source = 'voicebank'
            self.mixture_clean_source = 'voicebank'
            self.noisy_data_source = 'demand'
            self.mix_noisy_on_the_fly = True
        elif self.task == 'se_ears_wham':
            self.use_paird_training = True
            self.clean_data_source = 'ears_wham'
            self.mixture_clean_source = None
            self.noisy_data_source = 'ears_wham'
            self.mix_noisy_on_the_fly = False               
        elif self.task == 'DNS_ALL_TASK':
            self.use_paird_training = True
            self.clean_data_source = 'dns'
            self.mixture_clean_source = 'dns'
            self.noisy_data_source = 'dns'
            self.mix_noisy_on_the_fly = True            
        elif self.task == 'se_unpaired':
            self.use_paird_training = False
            self.clean_data_source = 'voicebank'
            self.mixture_clean_source = 'voicebank'
            self.noisy_data_source = 'demand'
            self.mix_noisy_on_the_fly = True
        elif self.task == 'se_unpaired_wsj0':
            self.use_paird_training = False
            self.clean_data_source = 'wsj0'
            self.mixture_clean_source = 'voicebank'
            self.noisy_data_source = 'demand'
            self.mix_noisy_on_the_fly = True
        elif self.task == 'se_unpaired_trump':
            self.use_paird_training = False
            self.clean_data_source = 'trump'
            self.mixture_clean_source = 'voicebank'
            self.noisy_data_source = 'demand'
            self.mix_noisy_on_the_fly = True            
        elif self.task == 'se_unpaired_dns':
            self.use_paird_training = False
            self.clean_data_source = 'dns'
            self.mixture_clean_source = 'voicebank'
            self.noisy_data_source = 'demand'
            self.mix_noisy_on_the_fly = True
        elif self.task == 'reverb':
            self.use_paird_training = True
            self.clean_data_source = 'wsj0-reverb'
            self.mixture_clean_source = 'wsj0-reverb'
            self.noisy_data_source = 'wsj0-reverb'
            self.mix_noisy_on_the_fly = False
        elif self.task == 'reverb_ears':
            self.use_paird_training = True
            self.clean_data_source = 'ears_reverb'
            self.mixture_clean_source = None
            self.noisy_data_source = 'ears_reverb'
            self.mix_noisy_on_the_fly = False                
        elif self.task == 'reverb_unpaired':
            self.use_paird_training = False
            self.clean_data_source = 'wsj0-reverb'
            self.mixture_clean_source = 'wsj0-reverb'
            self.noisy_data_source = 'wsj0-reverb'
            self.mix_noisy_on_the_fly = False
        elif self.task == 'reverb_dns':
            self.use_paird_training = False
            self.clean_data_source = 'dns'
            self.mixture_clean_source = 'wsj0-reverb'
            self.noisy_data_source = 'wsj0-reverb'
            self.mix_noisy_on_the_fly = False
        else:
            raise ValueError(f"Unknown task configuration: {self.task}")

        # Optional Kwarg overrides
        self.use_paird_training = kwargs.get('use_paird_training', self.use_paird_training)
        self.clean_data_source = kwargs.get('clean_data_source', self.clean_data_source)
        self.mixture_clean_source = kwargs.get('mixture_clean_source', self.mixture_clean_source)
        self.noisy_data_source = kwargs.get('noisy_data_source', self.noisy_data_source)
        self.mix_noisy_on_the_fly = kwargs.get('mix_noisy_on_the_fly', self.mix_noisy_on_the_fly)

        # 1. Clean Files (x)
        self.clean_files = []
        if self.clean_data_source == 'wsj0-reverb':
            self.clean_dir = kwargs.get('clean_dir', f"/vol/liangxu-solar/data/wsj0_reverb/{subset}/anechoic")
            self.clean_files = sorted(glob(join(self.clean_dir, "**", "*.wav"), recursive=True))
        elif self.clean_data_source == 'wsj0':
            self.clean_files = sorted(
                glob("/vol/liangxu-solar/data/wsj0_reverb/train/anechoic/*.wav") +
                glob("/vol/liangxu-solar/data/wsj0_reverb/valid/anechoic/*.wav") +
                glob("/vol/liangxu-solar/data/wsj0_reverb/test/anechoic/*.wav")
            )
        elif self.clean_data_source == 'voicebank':
            self.clean_dir = kwargs.get('clean_dir', "/vol/liangxu-solar/data/voicebank/VoiceBank-DEMAND-16k-wav/train/clean/")
            self.clean_files = sorted(glob(join(self.clean_dir, "*.wav")))            
        elif self.clean_data_source == 'ears_wham':
            self.clean_dir = kwargs.get('clean_dir', "/vol/liangxu-solar/data/EARS-WHAM_v2_16k/train/clean/")
            self.clean_files = sorted(glob(join(self.clean_dir, "**", "*.wav"), recursive=True))
        elif self.clean_data_source == 'ears_reverb':
            self.clean_dir = kwargs.get('clean_dir', "/vol/liangxu-solar/data/EARS-Reverb_v2_16k/train/clean/")
            self.clean_files = sorted(glob(join(self.clean_dir, "**", "*.wav"), recursive=True))            
        elif self.clean_data_source == 'trump':
            self.clean_dir = kwargs.get('clean_dir', "/vol/liangxu-solar/data/trump_audio/sgmse_clean_audios/N_30/")
            self.clean_files = sorted(glob(join(self.clean_dir, "*.wav")))
        elif self.clean_data_source == 'dns':
            # Default DNS clean directory can be overridden via `clean_dir` kwarg
            self.clean_dir = kwargs.get('clean_dir', "/vol/liangxu-solar/data/DNS-Challenge/datasets/clean/")
            self.clean_files = sorted(glob(join(self.clean_dir, "*.wav")))

        # 1.5 Mixture Clean Files (for constructing y when on-the-fly unpaired mixing)
        self.mixture_clean_files = []
        if self.mix_noisy_on_the_fly and not self.use_paird_training:
            if self.mixture_clean_source == 'voicebank':
                self.mix_clean_dir = kwargs.get('mix_clean_dir', "/vol/liangxu-solar/data/voicebank/VoiceBank-DEMAND-16k-wav/train/clean/")
                self.mixture_clean_files = sorted(glob(join(self.mix_clean_dir, "*.wav")))
            elif self.mixture_clean_source == 'wsj0-reverb':
                mix_clean_dir = kwargs.get('mix_clean_dir', f"/vol/liangxu-solar/data/wsj0_reverb/{subset}/anechoic")
                self.mixture_clean_files = sorted(glob(join(mix_clean_dir, "**", "*.wav"), recursive=True))

        # 2. Noise Files (y)
        self.noise_files = []
        if self.noisy_data_source == "wsj0-reverb":
            self.noise_dir = kwargs.get('noise_dir', f"/vol/liangxu-solar/data/wsj0_reverb/{subset}/reverb")
            self.noise_files = sorted(glob(join(self.noise_dir, "**", "*.wav"), recursive=True))
        elif self.noisy_data_source == "demand":
            self.noise_dir = kwargs.get('noise_dir', "/vol/liangxu-solar/data/DEMAND_16k/")
            self.noise_files = sorted(glob(join(self.noise_dir, "**", "*.wav"), recursive=True))
        elif self.noisy_data_source == "ears_wham":
            self.noise_dir = kwargs.get('noise_dir', "/vol/liangxu-solar/data/EARS-WHAM_v2_16k/train/noisy/")
            self.noise_files = sorted(glob(join(self.noise_dir, "**", "*.wav"), recursive=True))    
        elif self.noisy_data_source == "ears_reverb":
            self.noise_dir = kwargs.get('noise_dir', "/vol/liangxu-solar/data/EARS-Reverb_v2_16k/train/reverberant/")
            self.noise_files = sorted(glob(join(self.noise_dir, "**", "*.wav"), recursive=True))          
        elif self.noisy_data_source == "dns":
            self.noise_dir = kwargs.get('noise_dir', "/vol/liangxu-solar/data/DNS-Challenge/datasets/noise/")
            self.noise_files = sorted(glob(join(self.noise_dir, "*.wav")))            

        if len(self.clean_files) == 0:
             print(f"WARNING: No clean files found for {self.clean_data_source}")
        if len(self.noise_files) == 0:
             print(f"WARNING: No noise files found for form {self.noisy_data_source}")

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

        # Print out the dataset configuration summary once loaded
        print("\n" + "="*60)
        print(f"🚀 SpeechDataset initialized with task: '{self.task}'")
        print(f"  - Normalize: {self.normalize}")
        print(f"  - Target Clean (x_gt) Source: {self.clean_data_source} [{len(self.clean_files)} files]")
        if self.mix_noisy_on_the_fly:
            if self.use_paird_training:
                print(f"  - Noisy Input (y) Source: Synthesized on-the-fly [PAIRED]")
                print(f"    * Noise: {self.noisy_data_source} [{len(self.noise_files)} files]")
            else:
                print(f"  - Noisy Input (y) Source: Synthesized on-the-fly [UNPAIRED]")
                print(f"    * Base Speech:  {self.mixture_clean_source} [{len(self.mixture_clean_files)} files]")
                print(f"    * Noise:        {self.noisy_data_source} [{len(self.noise_files)} files]")
        else:
            if self.use_paird_training:
                print(f"  - Noisy Input (y) Source: Pre-mixed environmental [PAIRED]")
                print(f"    * Noisy Files:  {self.noisy_data_source} [{len(self.noise_files)} files]")
            else:
                print(f"  - Noisy Input (y) Source: Pre-mixed environmental [UNPAIRED]")
                print(f"    * Noisy Files:  {self.noisy_data_source} [{len(self.noise_files)} files]")
        print("="*60 + "\n")

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
        # 1. Load Ground Truth Clean Speech (x_gt)
        x_gt, _ = load(self.clean_files[i%len(self.clean_files)])

        # Ensure single channel (C=1)
        if x_gt.dim() == 2:
            x_gt = x_gt[0]

        # 2. Extract or Synthesize Noisy Observation (y)
        if self.mix_noisy_on_the_fly:    
            if self.use_paird_training:
                # Use the exact same physical baseline for both x,y
                x_ready_add_noise = x_gt.clone()
            else:
                # Load a DIFFERENT Clean Speech item to construct y (strictly unpaired)
                j = np.random.randint(0, len(self.mixture_clean_files))
                
                # If target and mixture come from same source, prevent exact same file collision.
                # If target is wsj0 and mixture is voicebank, there is exactly 0% chance of identical speaker/file sampling, so we do nothing!
                if self.clean_data_source == self.mixture_clean_source:
                    while j == i:
                        j = np.random.randint(0, len(self.mixture_clean_files))

                x_ready_add_noise, _ = load(self.mixture_clean_files[j])
                if x_ready_add_noise.dim() == 2:
                    x_ready_add_noise = x_ready_add_noise[0]

            # Load Random Noise for y
            noise_idx = np.random.randint(0, len(self.noise_files))
            n, _ = load(self.noise_files[noise_idx])
            if n.dim() == 2:
                n = n[0]
            if self.task == 'DNS_ALL_TASK':
                # use dns mix strategy ( reverb first and then do dynamic mixing)
                clean_np = x_ready_add_noise.numpy()
                rir = generate_random_rir(target_fs=16000)
                reverb_clean_np = fftconvolve(clean_np, rir, mode='full')[:len(clean_np)]
                
                max_reverb = np.max(np.abs(reverb_clean_np))
                max_clean = np.max(np.abs(clean_np))
                reverb_clean_np = reverb_clean_np / (max_reverb + 1e-8) * max_clean
                
                x_ready_add_noise = torch.from_numpy(reverb_clean_np).float()
                
                # snr = random.uniform(0, 20.0)
                snr = random.uniform(-10, 20.0)
                y = self.mix_noise(x_ready_add_noise, n, snr)
            else:
                # Dynamic Mixing (y)
                snr = random.choice([0, 5, 10, 15])
                y = self.mix_noise(x_ready_add_noise, n, snr)

        else:
            # Load Pre-Mixed/Reverb Noisy File
            if self.use_paird_training:
                # Strictly paired evaluation (e.g., standard wsj0-reverb paired)
                y, _ = load(self.noise_files[i])
            else:
                # If unpaired (e.g., DNS clean, WSJ0 reverb env), randomly select a noise record
                noise_idx = np.random.randint(0, len(self.noise_files))
                # Add strict check to avoid sampling the exact same corresponding index if the datasets match
                if self.clean_data_source == self.noisy_data_source:
                    mapped_i = i % len(self.clean_files)
                    while noise_idx == mapped_i:
                        noise_idx = np.random.randint(0, len(self.noise_files))
                        
                y, _ = load(self.noise_files[noise_idx])
                
            if y.dim() == 2:
                y = y[0]

        # 5. Cut/Pad to fixed segment
        target_len = (self.num_frames - 1) * self.stft_kwargs["hop_length"]
        
        def pad_or_cut(tensor, force_start=None):
            current_len = tensor.size(-1)
            pad_needed = max(target_len - current_len, 0)
            
            if pad_needed == 0:
                if force_start is not None:
                    # Use the provided synchronized start offset
                    start = min(force_start, current_len - target_len)
                else:
                    # Generate an independent random start for this specific tensor
                    if self.shuffle_spec:
                        start = int(np.random.uniform(0, current_len - target_len))
                    else:
                        start = int((current_len - target_len) / 2)
                return tensor[..., start:start+target_len], start
            else:
                return F.pad(tensor, (pad_needed//2, pad_needed//2+(pad_needed % 2)), mode='constant'), 0

        if self.use_paird_training:
            # PAIRED: Crop x_gt first, then force y to use the exact same start offset.
            x_gt, sync_start = pad_or_cut(x_gt)
            y, _ = pad_or_cut(y, force_start=sync_start)
        else:
            # UNPAIRED: Crop x_gt and y completely independently to fully explore both files.
            x_gt, _ = pad_or_cut(x_gt)
            y, _ = pad_or_cut(y)

        # Normalize waveform
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean" or self.normalize == "separate":
            normfac = x_gt.abs().max()
        else:
            normfac = 1.0
        
        x_gt_raw_wav = x_gt
        y_raw_wav = y
        normfac_tensor = torch.tensor(normfac if isinstance(normfac, float) else normfac.item(), dtype=torch.float32)
        x_gt = x_gt / (normfac + 1e-8)
        if self.normalize == "separate":
            y = y / (y.abs().max() + 1e-8)
        else:
            y = y / (normfac + 1e-8)

        # STFT
        X = torch.stft(x_gt, **self.stft_kwargs)
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
            return X, Y, x_gt_raw_wav, y_raw_wav, normfac_tensor
        return X, Y

    def __len__(self):
        if self.use_paird_training or not self.mix_noisy_on_the_fly:
            # For paired training, length is determined by clean files
            n = len(self.clean_files)
        else:
            # For unpaired on-the-fly mixing, length is determined by mixture clean files
            n = len(self.mixture_clean_files)
        if self.dummy:
            return min(n, 100)
        return n
