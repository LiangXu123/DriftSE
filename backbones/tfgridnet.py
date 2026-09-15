import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple
import difflib

try:
    from .shared import BackboneRegistry
except ImportError:
    # Fallback for standalone __main__ execution
    class _Registry:
        def register(self, name):
            def decorator(cls):
                return cls
            return decorator
    BackboneRegistry = _Registry()

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn"""
    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matches for layer with name {} in {}.\n All matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    return getattr(library, match[0])


class AbsSeparator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_spk(self):
        raise NotImplementedError


class TFGridNet(AbsSeparator):
    def __init__(
        self,
        n_srcs=1,
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=200,
        attn_n_head=4,
        attn_qk_output_channel=2,
        emb_dim=48,
        emb_ks=4, # Kernel size I, Kernel size for Unfold and Deconv1D
        emb_hs=1, # Stride size J, Stride size for Unfold and Deconv1D
        activation="prelu",
        eps=1.0e-5,
        causal=False,
        **kwargs
    ):
        super().__init__()
        if causal:
            raise NotImplementedError(
                "Causal mode is not supported in this model. "
                "Please use TFGridNet_Causal for causal streaming processing."
            )

        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        self.emb_dim = emb_dim

        in_ch = 4  # x.real, x.imag, y.real, y.imag
        t_ksize = 3
        self.t_ksize = t_ksize
        ks = (t_ksize, 3)

        # Symmetric padding in both time and frequency axes
        conv_pad   = (t_ksize // 2, 1)
        deconv_pad = (t_ksize // 2, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, emb_dim, ks, padding=conv_pad),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([
            GridNetV3Block(
                emb_dim, emb_ks, emb_hs, lstm_hidden_units,
                n_head=attn_n_head,
                qk_output_channel=attn_qk_output_channel,
                activation=activation,
                eps=eps,
            )
            for _ in range(n_layers)
        ])

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=deconv_pad)

    def forward(self, x, y, t=None):
        """
        x, y : [B, 1, F, T] complex
        t    : [B,]  (unused here, kept for API compatibility)
        out  : [B, 1, F, T] complex
        """
        input_concat = torch.cat((x.real, x.imag, y.real, y.imag), dim=1)  # [B, 4, F, T]
        batch = input_concat.permute(0, 1, 3, 2)                            # [B, 4, T, F]

        batch = self.conv(batch)   # [B, emb_dim, T, F]

        for block in self.blocks:
            batch = block(batch)   # [B, emb_dim, T, F]  — shape preserved

        old_T = batch.shape[2]
        batch = self.deconv(batch) # [B, n_srcs*2, T, F]

        batch = batch.reshape(
            batch.shape[0], self.n_srcs, 2, batch.shape[2], batch.shape[3]
        )  # [B, n_srcs, 2, T, F]
        batch = torch.view_as_complex(
            batch.permute(0, 1, 4, 3, 2).contiguous()
        )  # [B, n_srcs, F, T]
        return batch

    @property
    def num_spk(self):
        return self.n_srcs


class GridNetV3Block(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        hidden_channels,
        n_head=4,
        qk_output_channel=4,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        assert activation == "prelu"

        in_channels = emb_dim * emb_ks

        # ── Intra-RNN (frequency axis) ───────────────────────────────────────
        # Always bidirectional
        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        # ── Inter-RNN (time axis) ────────────────────────────────────────────
        # Always bidirectional
        inter_out_ch = hidden_channels * 2
        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1,
            batch_first=True,
            bidirectional=True,
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(inter_out_ch, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(
                inter_out_ch, emb_dim, emb_ks, stride=emb_hs
            )

        E = qk_output_channel
        assert emb_dim % n_head == 0

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )
        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )
        self.add_module(
            "attn_conv_V", nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        )
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DC((n_head, emb_dim // n_head), eps=eps),
        )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks  = emb_ks
        self.emb_hs  = emb_hs
        self.n_head  = n_head

    def forward(self, x):
        B, C, old_T, old_Q = x.shape

        olp = self.emb_ks - self.emb_hs
        T = (
            math.ceil((old_T + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        Q = (
            math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )

        x = x.permute(0, 2, 3, 1)  # [B, old_T, old_Q, C]

        # Symmetric padding in both time and frequency axes
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))

        # ── Intra RNN ────────────────────────────────────────────────────────
        input_ = x
        intra_rnn = self.intra_norm(input_)   # [B, T, Q, C]
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self.emb_ks * C])
            intra_rnn, _ = self.intra_rnn(intra_rnn)
            intra_rnn = self.intra_linear(intra_rnn)
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B * T, Q, C])
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn = F.unfold(
                intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn, _ = self.intra_rnn(intra_rnn)
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn = self.intra_linear(intra_rnn)
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)   # [B, T, Q, C]
        intra_rnn = intra_rnn + input_          # residual
        intra_rnn = intra_rnn.transpose(1, 2)   # [B, Q, T, C]

        # ── Inter RNN ────────────────────────────────────────────────────────
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)     # [B, Q, T, C]
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.emb_ks * C])
            inter_rnn, _ = self.inter_rnn(inter_rnn)
            inter_rnn = self.inter_linear(inter_rnn)
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B * Q, T, C)
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn, _ = self.inter_rnn(inter_rnn)
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn = self.inter_linear(inter_rnn)
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)   # [B, Q, T, C]
        inter_rnn = inter_rnn + input_          # residual
        inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # [B, C, T, Q]

        # Trim symmetric padding back to (old_T, old_Q)
        t_start = olp
        inter_rnn = inter_rnn[..., t_start : t_start + old_T, olp : olp + old_Q]
        batch = inter_rnn   # [B, C, old_T, old_Q]

        # ── Full-sequence attention ───────────────────────────────────────────
        Q_a = self["attn_norm_Q"](self["attn_conv_Q"](batch))   # [B, n_head*E, old_T, old_Q]
        K_a = self["attn_norm_K"](self["attn_conv_K"](batch))
        V_a = self["attn_norm_V"](self["attn_conv_V"](batch))

        Q_a = Q_a.view(-1, *Q_a.shape[2:])   # [B*n_head, E, old_T, old_Q]
        K_a = K_a.view(-1, *K_a.shape[2:])
        V_a = V_a.view(-1, *V_a.shape[2:])

        Q_a = Q_a.transpose(1, 2).flatten(start_dim=2)          # [B', old_T, E*old_Q]
        K_a = K_a.transpose(2, 3).contiguous().view(
            [B * self.n_head, -1, old_T]
        )                                                         # [B', E*old_Q, old_T]
        V_a = V_a.transpose(1, 2)                                # [B', old_T, C, old_Q]
        old_shape = V_a.shape
        V_a = V_a.flatten(start_dim=2)                           # [B', old_T, C*old_Q]

        emb_dim = Q_a.shape[-1]
        attn_mat = torch.matmul(Q_a, K_a) / (emb_dim ** 0.5)    # [B', old_T, old_T]

        # Symmetric full temporal attention (No upper-triangular masking)
        attn_mat = F.softmax(attn_mat, dim=2)
        V_a = torch.matmul(attn_mat, V_a)                        # [B', old_T, C*old_Q]

        V_a = V_a.reshape(old_shape).transpose(1, 2)             # [B', C, old_T, old_Q]
        emb_dim = V_a.shape[1]
        batch = V_a.contiguous().view(
            [B, self.n_head * emb_dim, old_T, old_Q]
        )
        batch = self["attn_concat_proj"](batch)                   # [B, C, old_T, old_Q]

        return batch + inter_rnn


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

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if x.ndim - 1 < self.dim:
            raise ValueError(
                f"Expect x to have {self.dim + 1} dimensions, but got {x.ndim}"
            )
        if x.dtype in HALF_PRECISION_DTYPES:
            dtype = x.dtype
            x = x.float()
        else:
            dtype = None
        mu_  = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat.to(dtype=dtype) if dtype else x_hat


class AllHeadPReLULayerNormalization4DC(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2, input_dimension
        H, E = input_dimension
        param_size = [1, H, E, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta  = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, F = x.shape
        x = x.view([B, self.H, self.E, T, F])
        x = self.act(x)
        stat_dim = (2,)
        mu_  = x.mean(dim=stat_dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)
        x = ((x - mu_) / std_) * self.gamma + self.beta
        return x


@BackboneRegistry.register("tfgridnet")
class TFGridNet_Backbone(TFGridNet):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--n_layers",           type=int,  default=5)
        parser.add_argument("--emb_dim",            type=int,  default=32)
        parser.add_argument("--lstm_hidden_units",  type=int,  default=100)
        return parser

    def __init__(self, **kwargs):
        valid_keys = {
            "n_srcs", "n_imics", "n_layers", "lstm_hidden_units", "attn_n_head",
            "attn_qk_output_channel", "emb_dim", "emb_ks", "emb_hs",
            "activation", "eps",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        filtered_kwargs.setdefault("n_layers",          5)
        filtered_kwargs.setdefault("emb_dim",           32)
        filtered_kwargs.setdefault("lstm_hidden_units", 100)
        super().__init__(**filtered_kwargs)

    def forward(self, x, y, t=None):
        return super().forward(x, y, t)


# =============================================================================
# Standalone test helpers
# =============================================================================

def _make_model(F: int, n_layers: int = 4,
                emb_dim: int = 32, lstm_hidden_units: int = 100,
                attn_n_head: int = 4) -> TFGridNet_Backbone:
    return TFGridNet_Backbone(
        n_layers=n_layers,
        emb_dim=emb_dim,
        lstm_hidden_units=lstm_hidden_units,
        attn_n_head=attn_n_head,
    ).eval()


if __name__ == "__main__":
    # 1. Install thop first if you haven't: pip install thop
    from thop import profile
    import torch

    F_bins, T_steps = 256, 128  # 128 frames = 1s audio at 16kHz (hop=128)
    
    configs = [
        {"name": "Base", "n_layers": 4, "emb_dim": 32, "lstm_hidden_units": 100, "attn_n_head": 4},
        {"name": "Tiny", "n_layers": 2, "emb_dim": 16, "lstm_hidden_units": 64, "attn_n_head": 1},
    ]

    for cfg in configs:
        model = _make_model(
            F=F_bins,
            n_layers=cfg["n_layers"],
            emb_dim=cfg["emb_dim"],
            lstm_hidden_units=cfg["lstm_hidden_units"],
            attn_n_head=cfg["attn_n_head"]
        )

        # Calculate Total and Trainable Parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Convert to Millions (M)
        total_params_m = total_params / 1e6
        trainable_params_m = trainable_params / 1e6

        # Initialize Dummy Inputs
        x = torch.randn(1, 1, F_bins, T_steps, dtype=torch.cfloat)
        y = torch.randn(1, 1, F_bins, T_steps, dtype=torch.cfloat)

        # Calculate MACs via Profile
        try:
            macs, _ = profile(model, inputs=(x, y), verbose=False)
            mac_string = f"{macs / 1e9:.2f} G MACs  ({macs:,})"
        except Exception as e:
            mac_string = "N/A (Complex float profiling fallback required)"
            
        print("-" * 50)
        print(f">>> Model Complexity Profile: TFGridNet ({cfg['name']}) <<<")
        print(f"Total Parameters:     {total_params_m:.2f} M  ({total_params:,} elements)")
        print(f"Trainable Parameters: {trainable_params_m:.2f} M")
        print(f"Total MACs (1s audio): {mac_string}")
        print("-" * 50)

    # Basic shape check for last model
    print(f"Input shape:  {x.shape}")
    with torch.no_grad():
        out = model(x, y)
    print(f"Output shape: {out.shape}")
    print("-" * 50)
    print(">>> TFGridNet (Strictly Non-Causal) Initialized Successfully! <<<")

# > python -m backbones.tfgridnet
# /vol/liangxu-solar/exp_code/Drifting_SE/DriftReverb_FD_EMA/backbones/tfgridnet.py:352: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   @torch.cuda.amp.autocast(enabled=False)         
# /home/liangxu/.local/lib/python3.12/site-packages/thop/vision/calc_func.py:53: UserWarning: This API is being deprecated
#   warnings.warn("This API is being deprecated")
# --------------------------------------------------
# >>> Model Complexity Profile: TFGridNet (Base) <<<                                                                    
# Total Parameters:     1.69 M  (1,690,646 elements)   
# Trainable Parameters: 1.69 M
# Total MACs (1s audio): 58.88 G MACs  (58,875,109,376.0)
# --------------------------------------------------
# --------------------------------------------------  
# >>> Model Complexity Profile: TFGridNet (Tiny) <<<
# Total Parameters:     0.30 M  (301,490 elements)       
# Trainable Parameters: 0.30 M                      
# Total MACs (1s audio): 10.56 G MACs  (10,558,202,880.0)
# --------------------------------------------------
# Input shape:  torch.Size([1, 1, 256, 128])      
# Output shape: torch.Size([1, 1, 256, 128])          
# --------------------------------------------------
# >>> TFGridNet (Strictly Non-Causal) Initialized Successfully! <<<
