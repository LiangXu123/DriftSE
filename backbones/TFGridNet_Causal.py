import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from collections import OrderedDict
from typing import Tuple

# -----------------------------------------------------------------------------
#                               TFGridNet_Causal
# -----------------------------------------------------------------------------

class TFGridNet_Causal(nn.Module):
    """
    Fully causal TF-GridNet for streaming speech enhancement / dereverberation.
    Mathematically guaranteed zero temporal look-ahead.
    """

    def __init__(
        self,
        n_srcs: int = 1,
        n_layers: int = 6,
        lstm_hidden_units: int = 192,
        attn_n_head: int = 4,
        attn_qk_output_channel: int = 4,    # E in the original paper
        emb_dim: int = 48,
        emb_ks: int = 4,
        emb_hs: int = 1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.t_ksize = 3                     # temporal kernel size

        # Input: 4 channels (x.real, x.imag, y.real, y.imag)
        in_ch = 4
        ks = (self.t_ksize, 3)               # (time, freq)

        # Causal Conv2D + Causal LayerNorm (no temporal pooling)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, emb_dim, ks, padding=(0, 1)),
            LayerNormalization(emb_dim, dim=1, total_dim=4, eps=eps),
        )

        # Stack of causal GridNet blocks
        self.blocks = nn.ModuleList([
            CausalGridNetBlock(
                emb_dim, emb_ks, emb_hs, lstm_hidden_units,
                n_head=attn_n_head,
                qk_output_channel=attn_qk_output_channel,
                eps=eps,
            )
            for _ in range(n_layers)
        ])

        # Causal Deconv2D → left-pad time axis only
        self.deconv = nn.ConvTranspose2d(
            emb_dim, n_srcs * 2, ks, padding=(0, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor = None):
        """
        x : [B, 1, F, T] complex
        y : [B, 1, F, T] complex
        """
        input_concat = torch.cat((x.real, x.imag, y.real, y.imag), dim=1)  # [B,4,F,T]
        batch = input_concat.permute(0, 1, 3, 2)                           # [B,4,T,F]

        # Causal time padding: t_ksize-1 frames of past context on the left
        batch = F.pad(batch, (0, 0, self.t_ksize - 1, 0))

        batch = self.conv(batch)          # [B, emb_dim, T, F]

        for block in self.blocks:
            batch = block(batch)          # shape preserved

        old_T = batch.shape[2]
        batch = self.deconv(batch)        # [B, n_srcs*2, T+t_ksize-1, F]
        batch = batch[..., :old_T, :]     # trim future frames introduced by the kernel

        batch = batch.reshape(
            batch.shape[0], self.n_srcs, 2, batch.shape[2], batch.shape[3]
        )                                # [B, n_srcs, 2, T, F]
        batch = torch.view_as_complex(
            batch.permute(0, 1, 4, 3, 2).contiguous()
        )                                # [B, n_srcs, F, T]
        return batch

    @property
    def num_spk(self):
        return self.n_srcs


# -----------------------------------------------------------------------------
#                           Causal GridNet Block
# -----------------------------------------------------------------------------

class CausalGridNetBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        emb_ks: int,
        emb_hs: int,
        hidden_channels: int,
        n_head: int = 4,
        qk_output_channel: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

        in_channels = emb_dim * emb_ks
        olp = emb_ks - emb_hs
        self.olp = olp

        # ---- Intra-RNN (frequency axis) ------------------------------------
        # Bidirectional across frequency within each single independent frame
        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1,
            batch_first=True, bidirectional=True,
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        # ---- Inter-RNN (time axis) -----------------------------------------
        # Strictly causal: unidirectional LSTM with direct frame-wise linear projection
        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1,
            batch_first=True, bidirectional=False,
        )
        # Bypasses ConvTranspose1d completely to prevent temporal leakage
        self.inter_linear = nn.Linear(hidden_channels, emb_dim)

        # ---- Full-band self-attention --------------------------------------
        E = qk_output_channel
        assert emb_dim % n_head == 0

        self.attn_conv_Q = nn.Conv2d(emb_dim, n_head * E, 1)
        self.attn_norm_Q = AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps)
        self.attn_conv_K = nn.Conv2d(emb_dim, n_head * E, 1)
        self.attn_norm_K = AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps)
        self.attn_conv_V = nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        self.attn_norm_V = AllHeadPReLULayerNormalization4DC(
            (n_head, emb_dim // n_head), eps=eps
        )
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            nn.PReLU(),
            LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, old_T, old_Q = x.shape
        olp = self.olp

        # Pad symmetrically in frequency to size Q
        Q = (
            math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )

        x = x.permute(0, 2, 3, 1)          # [B, old_T, old_Q, C]
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, 0, 0))  # Pad frequency only

        # ---- Intra-RNN -----------------------------------------------------
        input_ = x
        intra_rnn = self.intra_norm(input_)

        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view(B * old_T, -1, self.emb_ks * C)
            intra_rnn, _ = self.intra_rnn(intra_rnn)
            intra_rnn = self.intra_linear(intra_rnn)
            intra_rnn = intra_rnn.view(B, old_T, Q, C)
        else:
            intra_rnn = intra_rnn.view(B * old_T, Q, C).transpose(1, 2)
            intra_rnn = F.unfold(
                intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn, _ = self.intra_rnn(intra_rnn)
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn = self.intra_linear(intra_rnn)
            intra_rnn = intra_rnn.view(B, old_T, C, Q).transpose(-2, -1)
        intra_rnn = intra_rnn + input_                                  # residual
        intra_rnn = intra_rnn.transpose(1, 2)                           # [B, Q, old_T, C]

        # ---- Inter-RNN (Strictly Causal) -----------------------------------
        inter_rnn_residual = intra_rnn
        inter_rnn = self.inter_norm(intra_rnn)                         # [B, Q, old_T, C]

        # Reshape to [B * Q, C, old_T]
        inter_rnn = inter_rnn.permute(0, 1, 3, 2).contiguous().view(B * Q, C, old_T)

        # Causal left-only padding: (emb_ks - 1) frames of past context
        inter_rnn = F.pad(inter_rnn, (self.emb_ks - 1, 0))             # [B * Q, C, old_T + emb_ks - 1]

        # Unfold along time axis (stride = 1)
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(1, 1)
        )                                                               # [B * Q, C * emb_ks, old_T]
        inter_rnn = inter_rnn.transpose(1, 2)                           # [B * Q, old_T, C * emb_ks]

        # Run unidirectional LSTM
        inter_rnn, _ = self.inter_rnn(inter_rnn)                        # [B * Q, old_T, hidden_channels]

        # Project directly back to channels causally
        inter_rnn = self.inter_linear(inter_rnn)                        # [B * Q, old_T, C]

        # Reshape back to [B, C, old_T, Q]
        inter_rnn = inter_rnn.view(B, Q, old_T, C).permute(0, 3, 2, 1).contiguous()
        
        # Trim frequency padding back to old_Q
        inter_rnn = inter_rnn[..., olp : olp + old_Q]
        
        # Permute residual from [B, Q, old_T, C] to [B, C, old_T, Q] to match layout
        res = inter_rnn_residual.permute(0, 3, 2, 1).contiguous()
        res = res[..., olp : olp + old_Q]
        
        inter_rnn = inter_rnn + res

        # ---- Causal full-band self-attention -------------------------------
        batch = inter_rnn                                               # [B,C,old_T,old_Q]

        Q_a = self.attn_norm_Q(self.attn_conv_Q(batch))
        K_a = self.attn_norm_K(self.attn_conv_K(batch))
        V_a = self.attn_norm_V(self.attn_conv_V(batch))

        Q_a = Q_a.view(-1, *Q_a.shape[2:])        # [B*n_head, E, old_T, old_Q]
        K_a = K_a.view(-1, *K_a.shape[2:])
        V_a = V_a.view(-1, *V_a.shape[2:])

        Q_a = Q_a.transpose(1, 2).flatten(start_dim=2)       # [B', old_T, E*old_Q]
        K_a = K_a.transpose(2, 3).contiguous().view(
            -1, K_a.shape[1] * K_a.shape[3], old_T
        )                                                    # [B', E*old_Q, old_T]
        V_a = V_a.transpose(1, 2)                            # [B', old_T, C', old_Q]
        old_shape = V_a.shape
        V_a = V_a.flatten(start_dim=2)                       # [B', old_T, C'*old_Q]

        emb_dim_q = Q_a.shape[-1]
        attn_mat = torch.matmul(Q_a, K_a) / (emb_dim_q ** 0.5)

        # CAUSAL MASK: upper-triangular
        causal_mask = torch.triu(
            torch.ones(old_T, old_T, device=attn_mat.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_mat = attn_mat.masked_fill(causal_mask, float("-inf"))
        attn_mat = F.softmax(attn_mat, dim=2)

        V_a = torch.matmul(attn_mat, V_a)                    # [B', old_T, C'*old_Q]
        V_a = V_a.reshape(old_shape).transpose(1, 2)         # [B', C', old_T, old_Q]

        emb_dim_v = V_a.shape[1]
        batch = V_a.contiguous().view(
            B, self.n_head * emb_dim_v, old_T, old_Q
        )                                                    # [B, C, old_T, old_Q]
        batch = self.attn_concat_proj(batch)                  # [B, C, old_T, old_Q]

        return batch + inter_rnn


# -----------------------------------------------------------------------------
#                         Normalisation helpers
# -----------------------------------------------------------------------------

HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16) if hasattr(torch, "bfloat16") else (torch.float16,)


class LayerNormalization(nn.Module):
    def __init__(self, input_dim, dim=1, total_dim=4, eps=1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta  = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        if x.dim() - 1 < self.dim:
            raise ValueError(f"Expect x to have {self.dim + 1} dimensions, got {x.dim()}")
        dtype = x.dtype if x.dtype in HALF_PRECISION_DTYPES else None
        if dtype is not None:
            x = x.float()
        mu  = x.mean(dim=self.dim, keepdim=True)
        std = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu) / std) * self.gamma + self.beta
        return x_hat.to(dtype) if dtype else x_hat


class AllHeadPReLULayerNormalization4DC(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        H, E = input_dimension
        param_size = [1, H, E, 1, 1]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta  = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E

    def forward(self, x):
        B, _, T, F = x.shape
        x = x.view(B, self.H, self.E, T, F)
        x = self.act(x)
        mu  = x.mean(dim=(2,), keepdim=True)
        std = torch.sqrt(x.var(dim=(2,), unbiased=False, keepdim=True) + self.eps)
        x = ((x - mu) / std) * self.gamma + self.beta
        return x


# -----------------------------------------------------------------------------
#                               Standalone test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Install thop first if you haven't: pip install thop
    from thop import profile
    import torch

    # Comprehensive causality test suite
    print("=== TFGridNet_Causal: Rigorous Causality Test ===\n")
    
    F_bins, T_steps = 256, 256  # changed from 32, 64
    model = TFGridNet_Causal(
        n_srcs=1,
        n_layers=2,
        lstm_hidden_units=64,
        emb_dim=16,
        attn_n_head=1,
    ).eval()

    torch.manual_seed(42)
    
    # Test 1: Single trial with max_diff check (basic sanity)
    x = torch.randn(1, 1, F_bins, T_steps, dtype=torch.cfloat)
    y = torch.randn(1, 1, F_bins, T_steps, dtype=torch.cfloat)
    split = T_steps // 2
    out_full = model(x, y)
    x_masked, y_masked = x.clone(), y.clone()
    x_masked[..., split:] = 0.0
    y_masked[..., split:] = 0.0
    out_masked = model(x_masked, y_masked)
    diff = (out_full[..., :split] - out_masked[..., :split]).abs().max().item()
    print(f"Basic test: split={split}, max_diff={diff:.2e}  {'✓' if diff < 1e-6 else '✗'}")

    # --- Added Complexity Profiling Section ---
    configs = [
        {"name": "Base", "n_layers": 4, "emb_dim": 32, "lstm_hidden_units": 100, "attn_n_head": 4},
        {"name": "Tiny", "n_layers": 2, "emb_dim": 16, "lstm_hidden_units": 64, "attn_n_head": 1},
    ]

    for cfg in configs:
        model_prof = TFGridNet_Causal(
            n_srcs=1,
            n_layers=cfg["n_layers"],
            emb_dim=cfg["emb_dim"],
            lstm_hidden_units=cfg["lstm_hidden_units"],
            attn_n_head=cfg["attn_n_head"],
        ).eval()

        total_params = sum(p.numel() for p in model_prof.parameters())
        trainable_params = sum(p.numel() for p in model_prof.parameters() if p.requires_grad)
        total_params_m = total_params / 1e6
        trainable_params_m = trainable_params / 1e6

        x_mac = torch.randn(1, 1, 256, 128, dtype=torch.cfloat)
        y_mac = torch.randn(1, 1, 256, 128, dtype=torch.cfloat)
        try:
            macs, _ = profile(model_prof, inputs=(x_mac, y_mac), verbose=False)
            mac_string = f"{macs / 1e9:.2f} G MACs  ({macs:,})"
        except Exception as e:
            mac_string = "N/A (Complex float profiling fallback required)"

        print("\n" + "-" * 50)
        print(f">>> Model Complexity Profile: TFGridNet_Causal ({cfg['name']}) <<<")
        print(f"Total Parameters:     {total_params_m:.2f} M  ({total_params:,} elements)")
        print(f"Trainable Parameters: {trainable_params_m:.2f} M")
        print(f"Total MACs (1s audio): {mac_string}")
        print("-" * 50)
    print("")
    # ------------------------------------------

    # Test 2: Multiple random trials with varying split points
    num_trials = 20
    all_pass = True
    for trial in range(num_trials):
        # Vary length between 8 and 64
        T_rand = torch.randint(8, 65, (1,)).item()
        F_rand = torch.randint(16, 48, (1,)).item()
        x = torch.randn(1, 1, F_rand, T_rand, dtype=torch.cfloat)
        y = torch.randn(1, 1, F_rand, T_rand, dtype=torch.cfloat)
        # Choose split between 2 and T_rand-1
        split = torch.randint(2, T_rand-1, (1,)).item() if T_rand > 3 else 1
        out_full = model(x, y)
        x_masked, y_masked = x.clone(), y.clone()
        x_masked[..., split:] = 0.0
        y_masked[..., split:] = 0.0
        out_masked = model(x_masked, y_masked)
        diff = (out_full[..., :split] - out_masked[..., :split]).abs().max().item()
        ok = diff < 1e-5
        if not ok:
            print(f"Trial {trial}: FAIL with T={T_rand}, F={F_rand}, split={split}, max_diff={diff:.3e}")
            all_pass = False
            break
    print(f"Multiple random trials ({num_trials}): {'ALL PASS ✓' if all_pass else 'FAIL ✗'}")

    # Test 3: Verify that future inputs never affect any past output for all t in one sequence
    # We zero out all future frames from each time step and check all past outputs.
    T = 32
    x = torch.randn(1, 1, F_bins, T, dtype=torch.cfloat)
    y = torch.randn(1, 1, F_bins, T, dtype=torch.cfloat)
    out_full = model(x, y)
    all_steps_pass = True
    for t in range(1, T-1):
        x_masked = x.clone()
        y_masked = y.clone()
        x_masked[..., t:] = 0.0
        y_masked[..., t:] = 0.0
        out_masked = model(x_masked, y_masked)
        diff = (out_full[..., :t] - out_masked[..., :t]).abs().max().item()
        if diff >= 1e-5:
            print(f"  Leakage at t={t}: max_diff={diff:.3e}")
            all_steps_pass = False
            break
    print(f"All time-step causality check (T={T}): {'PASS ✓' if all_steps_pass else 'FAIL ✗'}")

    # Test 4: Batch test – causality should hold independently per sample
    batch_size = 4
    T = 16
    x_batch = torch.randn(batch_size, 1, F_bins, T, dtype=torch.cfloat)
    y_batch = torch.randn(batch_size, 1, F_bins, T, dtype=torch.cfloat)
    out_full_batch = model(x_batch, y_batch)
    # Mask future for first sample only
    x_masked_batch = x_batch.clone()
    y_masked_batch = y_batch.clone()
    x_masked_batch[0, ..., 8:] = 0.0
    y_masked_batch[0, ..., 8:] = 0.0
    out_masked_batch = model(x_masked_batch, y_masked_batch)
    # Past of first sample must match
    diff = (out_full_batch[0, ..., :8] - out_masked_batch[0, ..., :8]).abs().max().item()
    # Other samples should not match (since they still see full future)
    other_diff = (out_full_batch[1:, ..., :] - out_masked_batch[1:, ..., :]).abs().max().item()
    print(f"Batch test (sample 0 past): diff={diff:.2e}  {'✓' if diff < 1e-6 else '✗'}")
    print(f"Batch test (other samples): diff={other_diff:.3e} (expected >0, since future not zeroed)")

    print("\n=== Causality verification complete ===")

# > python -m backbones.TFGridNet_Causal
# === TFGridNet_Causal: Rigorous Causality Test ===

# Basic test: split=128, max_diff=0.00e+00  ✓
# /home/liangxu/.local/lib/python3.12/site-packages/thop/vision/calc_func.py:53: UserWarning: This API is being deprecated
#   warnings.warn("This API is being deprecated")

# --------------------------------------------------
# >>> Model Complexity Profile: TFGridNet_Causal (Base) <<<
# Total Parameters:     1.24 M  (1,235,286 elements)
# Trainable Parameters: 1.24 M
# Total MACs (1s audio): 41.43 G MACs  (41,430,450,176.0)
# --------------------------------------------------

# --------------------------------------------------
# >>> Model Complexity Profile: TFGridNet_Causal (Tiny) <<<
# Total Parameters:     0.22 M  (220,746 elements)
# Trainable Parameters: 0.22 M
# Total MACs (1s audio): 7.45 G MACs  (7,447,330,816.0)
# --------------------------------------------------
