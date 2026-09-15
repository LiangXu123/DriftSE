from typing import Optional, List, Tuple, Mapping, Sequence, Callable
import warnings

import abc
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .local_einops import rearrange

# increase torch dynamo compile limit
import torch._dynamo
torch._dynamo.config.cache_size_limit  = 64

from .unet_utils.layerspp import Combine, get_act
from .unet_utils.layerspp import AttnBlockpp, NIN  # only for ablation studies


class CausalStreamingModule(abc.ABC):
    def init_state(self, *args, **kwargs) -> dict:
        """
        Initialize the state of the module. This is called at the beginning of each new sequence.
        Should return a dictionary with the initial state. `forward_step` should read this state
        and potentially return a modified copy.

        Returns {} by default, assuming statelessness.
        """
        return {}

    @abc.abstractmethod
    def forward_step(self, x: torch.Tensor, *args, state: dict) -> Tuple[torch.Tensor, dict]:
        """
        Forward step for the module. This is called for each time frame in the input sequence.
        Should return the output time frame and an updated state.

        Args:
            x: Input tensor of shape (batch_size, channels, frequency, time).
            *args: Additional arguments, e.g. auxiliary tensors like for time conditioning
            state: Current state of the module

        Returns:
            Tuple of (output tensor, updated state).
        """
        pass


class StreamingIdentity(nn.Module, CausalStreamingModule):
    def forward_step(self, x: torch.Tensor, *args, state: dict) -> Tuple[torch.Tensor, dict]:
        return x, state

    def init_state(self, *args, **kwargs):
        return ()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class CausalAttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Causal along last spatial dim if specified, with optional window."""
    def __init__(self, channels, in_freqs, skip_rescale=False, init_scale=0.,
                 norm_type='subband_grouped_batchnorm', norm_kwargs=None, freq_groups=4,
                 causal_attn=False, window_size=None):
        super().__init__()
        # self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
        #                                eps=1e-6)
        self.norm_type = norm_type
        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.Norm_0 = instantiate_norm(
            norm_type,
            num_groups=max(1, min(channels // 4, 32)), num_channels=channels,
            num_freqs=in_freqs, freq_groups=freq_groups,
            **self.norm_kwargs)

        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale
        self.causal_attn = causal_attn
        self.window_size = window_size  # only applies to last dim W
        assert (self.window_size is None) or self.causal_attn

        self.is_causal = self.causal_attn and self.norm_type != 'groupnorm'

    def forward(self, x):
        _, C, Fr, T = x.shape
        h = self.Norm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        # Compute attn logits
        w = torch.einsum('bchw,bcij->bhwij', q, k) * (C ** -0.5)

        if self.causal_attn:
            # Mask along the time dimension (key columns) for each query column
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=w.dtype))
            if self.window_size is not None:
                mask = mask - torch.tril(mask, diagonal=-self.window_size)
            mask = mask.view(1, 1, T, 1, T)
            w = w.masked_fill(mask == 0, float('-inf'))

        # Softmax along last dim (key columns)
        w = torch.nn.functional.softmax(w, dim=-1)

        # Apply attention and final NIN
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)



# Full Causal U-Net-like Backbone but without any time-wise downsampling (just lowpassing/running-averaging + dilations)

from .shared import BackboneRegistry

@BackboneRegistry.register("streamunet")
class CausalNCSNpp(nn.Module, CausalStreamingModule):
    """
    Causal-adapted and optimized NCSN++ model, as described in Stream.FM https://arxiv.org/pdf/2512.19442
    No time/noise conditioning — for drift-based training only.
    """

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--nf", type=int, default=128)
        parser.add_argument("--ch_mult", type=int, nargs='+', default=[1, 2, 2, 2])
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[])
        return parser

    def __init__(self,
        nf = 128,
        ch_mult = (1, 2, 2, 2),
        num_res_blocks = 2,
        input_channels = 4,  # [xt.re, xt.im, y.re, y.im]
        output_channels = 2, # [v.re, v.im]
        nonlinearity = 'swish',
        fir_kernel = (1, 3, 3, 1),
        init_scale = 1e-5,
        input_freqs = 256,
        dropout = 0.0,
        norm_type = 'subband_grouped_batchnorm',
        norm_kwargs: Optional[dict] = None,
        down_dilation = 2,
        freq_groups: int = 4,
        no_freq_groups_below: Optional[int] = None,  # disable freq groups for n_freqs <= this value. only used if norm_type=='subband_grouped_batchnorm'

        # Ablation study stuff, unused otherwise, !!! [may cause model non-causality] !!!
        attn_resolutions = (),
        attn_bottleneck: bool = False,
        attn_bottleneck_kwargs: Optional[dict] = None,
        up_path_channel_concatenation: bool = False,
        **unused_kwargs  # absorb irrelevant JSON config keys (e.g. resblock_type, progressive, etc.)
    ):
        super().__init__()
        self.is_causal = True  # assume optimistically, then possibly set to False later

        self.act = act = get_act(nonlinearity)

        self.nf = nf = nf
        self.num_res_blocks = num_res_blocks

        # Resolution stuff
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.input_freqs = input_freqs
        self.all_resolutions = all_resolutions = [input_freqs // (2 ** i) for i in range(num_resolutions)]

        # Input projection layer
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_layer = causal_conv3x3(input_channels, nf) # Input conv to inner channels (3x3)

        # No time/noise conditioning — drift-based training only
        temb_dim = None

        # List of output channels at each U-Net level as determined by nf and the ch_mult list
        self.out_chs = out_chs = [nf * ch_mult[l] for l in range(len(ch_mult))]

        # Options for 'downsampling' (lowpass filtering) along time
        self.down_dilation = down_dilation

        # Frequency grouping for the SGBatchNorm
        self.freq_groups = freq_groups
        self.no_freq_groups_below = no_freq_groups_below

        # Options for norms
        self.norm_type = norm_type
        if norm_kwargs is None:
            norm_kwargs = {}
        self.norm_kwargs = norm_kwargs

        # Options for ablation studies
        self.up_path_channel_concatenation = up_path_channel_concatenation
        self.attn_resolutions = attn_resolutions
        self.attn_bottleneck = attn_bottleneck
        self.attn_bottleneck_kwargs = attn_bottleneck_kwargs if attn_bottleneck_kwargs is not None else {}
        self.attn_levels = [res in attn_resolutions for res in all_resolutions]
        assert(all(res in all_resolutions for res in attn_resolutions)), \
            f"attn_resolutions={attn_resolutions} contains resolutions not found in all_resolutions={all_resolutions}!"

        Combiner = functools.partial(Combine, method='sum_rescaled')
        Combiner_up = functools.partial(Combine, method='sum_rescaled' if not self.up_path_channel_concatenation else 'cat')

        if attn_bottleneck and self.attn_bottleneck_kwargs.get('causal_attn', False):
            AttnBlock_bottleneck = functools.partial(CausalAttnBlockpp, init_scale=init_scale, skip_rescale=True, **self.attn_bottleneck_kwargs)
            _use_attnblockpp = False
        else:
            AttnBlock_bottleneck = functools.partial(AttnBlockpp, init_scale=init_scale, skip_rescale=True, **self.attn_bottleneck_kwargs)
            _use_attnblockpp = True

        rnb_kwargs = dict(
            act=act, dropout=dropout, fir_kernel=fir_kernel,
            init_scale=init_scale, skip_rescale=True, temb_dim=temb_dim,
            norm_type=norm_type, norm_kwargs=norm_kwargs,
            dilation=(1, 1),
        )
        ResnetBlock_ = functools.partial(CausalResnetBlock, **rnb_kwargs)
        ResnetBlockDown_ = functools.partial(CausalResnetBlockDown, **rnb_kwargs)
        ResnetBlockUp_ = functools.partial(CausalResnetBlockUp, **rnb_kwargs)
        ResnetBlockBottleneck_ = functools.partial(CausalResnetBlockBottleneck, **rnb_kwargs)

        self.pyramid_downsample = DownOrUpSample(freq_down=True, freq_up=False, fir_kernel=fir_kernel)
        self.pyramid_upsample = DownOrUpSample(freq_down=False, freq_up=True, fir_kernel=fir_kernel)

        # Down path
        down_modules = {}
        input_pyramid_ch = input_channels
        hs_c = [nf]
        in_ch = nf
        for l in range(num_resolutions):
            out_ch = out_chs[l]
            # Residual blocks for this resolution
            for i in range(num_res_blocks):
                down_modules[f'lvl{l}_rnb{i}'] = ResnetBlock_(
                    in_ch=in_ch, out_ch=out_ch,
                    in_freqs=all_resolutions[l], freq_groups=self.get_freq_groups(all_resolutions[l]),
                )
                in_ch = out_ch
                # if attn_levels[l]:
                #     down_modules[f'lvl{l}_attn{i}'] = AttnBlock(channels=in_ch)
                hs_c.append(in_ch)

            if l != num_resolutions - 1:
                down_modules[f'lvl{l}_rnb_down'] = ResnetBlockDown_(
                    in_ch=in_ch, out_ch=in_ch, freq_down=True, freq_up=False,
                    dilation=(1, self.down_dilation),
                    in_freqs=all_resolutions[l], freq_groups=self.get_freq_groups(all_resolutions[l]),
                )
                down_modules[f'lvl{l}_combiner'] = Combiner(dim1=input_pyramid_ch, dim2=in_ch)
        self.down_modules = nn.ModuleDict(down_modules)

        # Bottleneck
        bottleneck_modules = {}
        in_ch = hs_c[-1]  # last iteration of final l and final i in downsampling path (not the l == num_resolutions-1 path!)
        bottleneck_modules['rnb1'] = ResnetBlockBottleneck_(
            in_ch=in_ch, out_ch=in_ch,
            in_freqs=all_resolutions[-1], freq_groups=self.get_freq_groups(all_resolutions[-1]),
        )
        if self.attn_bottleneck:
            if _use_attnblockpp:
                _attn_block = AttnBlock_bottleneck(channels=in_ch, **self.attn_bottleneck_kwargs)
            else:
                _attn_block = AttnBlock_bottleneck(
                    channels=in_ch, in_freqs=all_resolutions[l], freq_groups=self.get_freq_groups(all_resolutions[l]),
                    **self.attn_bottleneck_kwargs)
                self.is_causal = self.is_causal and _attn_block.is_causal
            bottleneck_modules['attn'] = _attn_block

        bottleneck_modules['rnb2'] = ResnetBlockBottleneck_(
            in_ch=in_ch, out_ch=in_ch,
            in_freqs=all_resolutions[-1], freq_groups=self.get_freq_groups(all_resolutions[-1]),
        )
        self.bottleneck_modules = nn.ModuleDict(bottleneck_modules)

        # Up path
        up_modules = {}
        for l in reversed(range(num_resolutions)):
            out_ch = out_chs[l]

            nrb = self.num_res_blocks + (1 if l == 0 else 0)
            for i in range(nrb):
                h_c = hs_c.pop()
                combiner = Combiner_up(dim1=h_c, dim2=in_ch)
                up_modules[f'lvl{l}_combiner{i}'] = combiner
                up_modules[f'lvl{l}_rnb{i}'] = ResnetBlock_(
                    in_ch=combiner.out_ch,
                    out_ch=out_ch,
                    in_freqs=all_resolutions[l], freq_groups=self.get_freq_groups(all_resolutions[l]),
                )
                in_ch = out_ch

            # Output pyramid norm-activation-convolution module
            up_modules[f'lvl{l}_pyramid_normconv'] = CausalPyramidNormConv(
                in_ch=in_ch, out_ch=output_channels,
                in_freqs=all_resolutions[l], freq_groups=self.get_freq_groups(all_resolutions[l]),
                init_scale=init_scale, act=act, norm_type=norm_type, norm_kwargs=norm_kwargs,
            )

            # Upsampling layer (only needed if not at original resolution i.e. level 0)
            if l != 0:
                up_modules[f'lvl{l}_rnb_up'] = ResnetBlockUp_(
                    in_ch=in_ch, out_ch=in_ch,
                    in_freqs=all_resolutions[l], freq_groups=self.get_freq_groups(all_resolutions[l]),
                    freq_down=False, freq_up=True,
                )
        self.up_modules = nn.ModuleDict(up_modules)

        assert not hs_c, f"{len(hs_c)}"  # we should have consumed all tracked channels

    def get_freq_groups(self, n_freqs: int) -> int:
        if self.norm_type == 'subband_grouped_batchnorm':
            if self.no_freq_groups_below is not None:
                result = 1 if n_freqs < self.no_freq_groups_below else self.freq_groups
            else:
                result = self.freq_groups
        else:
            result = 1
        return result

    def forward(self, x, y=None, t=None, use_strict_gaussian=False):
        """t is kept for interface compatibility but is not used (no time embedding)."""
        complex_wrapper = False
        if x.dtype == torch.complex64 or x.dtype == torch.complex128:
            complex_wrapper = True
            if y is not None and (y.dtype == torch.complex64 or y.dtype == torch.complex128):
                if use_strict_gaussian:
                    # Strict mode: fully decoupled x and y paths.
                    # Main trunk sees ONLY pure Gaussian noise: [x.re, x.im, x.re, x.im]
                    # input_pyramid (progressive encoder injections) carries ONLY noisy speech y: [y.re, y.im, y.re, y.im]
                    y_pyramid = torch.cat([y.real, y.imag, y.real, y.imag], dim=1)
                    x = torch.cat([x.real, x.imag, x.real, x.imag], dim=1)
                else:
                    x = torch.cat([x.real, x.imag, y.real, y.imag], dim=1)  # cat along channels
            else:
                x = torch.cat([x.real, x.imag], dim=1)  # cat along channels

        # No time embedding — drift-based model
        temb = None

        # Input layer: Conv2d: 4ch -> nf
        h0 = self.input_layer(x)
        # Tracks all activations needed for the up path. We include h0 as a special case (skip from input layer)
        hs_up = [h0]
        # Strict mode: pyramid carries pure y; normal mode: pyramid matches the [x,y] concat input
        input_pyramid = y_pyramid if use_strict_gaussian else x
        h = h0

        # Down path in U-Net
        for l in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i in range(self.num_res_blocks):
                # Residual block
                h = self.down_modules[f'lvl{l}_rnb{i}'](h, temb)
                hs_up.append(h)

            # Downsampling
            if l != self.num_resolutions - 1:
                h = self.down_modules[f'lvl{l}_rnb_down'](h, temb)
                input_pyramid = self.pyramid_downsample(input_pyramid)
                h = self.down_modules[f'lvl{l}_combiner'](input_pyramid, h)

        # Bottleneck
        h = self.bottleneck_modules['rnb1'](h, temb)  # ResNet block
        if self.attn_bottleneck:
            h = self.bottleneck_modules['attn'](h)    # Attention block. !!NON-CAUSAL!!
        h = self.bottleneck_modules['rnb2'](h, temb)  # ResNet block

        # Up path in U-Net
        pyramid = None
        for l in reversed(range(self.num_resolutions)):
            nrb = self.num_res_blocks + (1 if l == 0 else 0)
            for i in range(nrb):
                h_input = self.up_modules[f'lvl{l}_combiner{i}'](hs_up.pop(), h)
                h = self.up_modules[f'lvl{l}_rnb{i}'](h_input, temb)

            # Conv2D: 256 -> 2 (maps to output channels)
            pyramid_h = self.up_modules[f'lvl{l}_pyramid_normconv'](h)

            if l != self.num_resolutions - 1:
                # Upsample previous pyramid and sum-combine
                # (unless at lowest resolution where no previous exists)
                pyramid_up = self.pyramid_upsample(pyramid)
                pyramid = pyramid_up + pyramid_h
            else:
                pyramid = pyramid_h

            # Upsampling Layer (unless at original resolution)
            if l != 0:
                h = self.up_modules[f'lvl{l}_rnb_up'](h, temb)  # Upsampling
            else:
                # Final output from the up path is just the pyramid result
                h = pyramid

        assert not hs_up  # we should have consumed all tracked activations

        # If we received complex input, convert it back
        if complex_wrapper:
            h = rearrange(h, "b (reim c) f t -> b c f t reim", reim=2).contiguous()
            h = torch.view_as_complex(h)

        return h

    def init_state(self) -> list:
        input_freqs = self.input_freqs

        state = []
        state.append(self.input_layer.init_state(input_freqs=input_freqs))
        for l in range(self.num_resolutions):
            for i in range(self.num_res_blocks):
                state.append(self.down_modules[f'lvl{l}_rnb{i}'].init_state())
            if l != self.num_resolutions - 1:
                state.append(self.down_modules[f'lvl{l}_rnb_down'].init_state())
                state.append(self.pyramid_downsample.init_state(input_freqs=self.all_resolutions[l]))

        state.append(self.bottleneck_modules['rnb1'].init_state())
        state.append(self.bottleneck_modules['rnb2'].init_state())

        for l in reversed(range(self.num_resolutions)):
            nrb = self.num_res_blocks + (1 if l == 0 else 0)
            for i in range(nrb):
                state.append(self.up_modules[f'lvl{l}_rnb{i}'].init_state())
            state.append(self.up_modules[f'lvl{l}_pyramid_normconv'].init_state(input_freqs=self.all_resolutions[l]))
            if l != self.num_resolutions - 1:
                state.append(self.pyramid_upsample.init_state(input_freqs=self.all_resolutions[l]))
            if l != 0:
                state.append(self.up_modules[f'lvl{l}_rnb_up'].init_state())

        return state

    def prepare_state(self):
        state = self.init_state()
        return self.mark_as_graph_output(state)

    def mark_as_graph_output(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj + 0.0 # Safe no-op that ensures graph capture sees it
        elif isinstance(obj, Mapping):
            return {k: self.mark_as_graph_output(v) for k, v in obj.items()}
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            return [self.mark_as_graph_output(v) for v in obj]

    def zero_state(self, obj):
        if isinstance(obj, torch.Tensor):
            obj.fill_(0.0) # Safe no-op that ensures graph capture sees it
        elif isinstance(obj, Mapping):
            for v in obj.values():
                self.zero_state(v)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            for v in obj:
                self.zero_state(v)

    @torch.compile(fullgraph=True, options={
        'max_autotune': True, 'epilogue_fusion': True, 'shape_padding': True,
    })
    def forward_step(self, x: torch.Tensor, t=None, aux_condition=None, *, state: list) -> Tuple[torch.Tensor, List]:
        """
        Forward pass for a single frame.
        Requires a state `state` that is initialized with `init_state` and updated at each step.
        `t` is kept for interface compatibility but is not used (no time embedding).

        NOTE: This currently expects real-valued inputs, i.e. the complex values already concatenated along channels
        in the same manner as the `forward()` function automatically does when it receives complex inputs.
        """
        if not self.is_causal:
            raise NotImplementedError("forward_step cannot be implemented if the model is not causal. Check your configuration.")

        # No time embedding — drift-based model
        temb = None
        m_idx = 0

        h0, _ = self.input_layer.forward_step(x, state=state[m_idx])
        m_idx += 1
        hs_up = [h0]
        input_pyramid = x
        h = h0

        # Down path
        for l in range(self.num_resolutions):
            for k in range(self.num_res_blocks):
                h, _ = self.down_modules[f'lvl{l}_rnb{k}'].forward_step(
                    h, temb, state=state[m_idx])
                m_idx += 1

                hs_up.append(h)
            if l != self.num_resolutions - 1:
                h, _ = self.down_modules[f'lvl{l}_rnb_down'].forward_step(
                    h, temb, state=state[m_idx])
                m_idx += 1

                input_pyramid, state[m_idx] = self.pyramid_downsample.forward_step(
                    input_pyramid, state=state[m_idx])
                m_idx += 1

                h = self.down_modules[f'lvl{l}_combiner'](input_pyramid, h)

        h, _ = self.bottleneck_modules['rnb1'].forward_step(h, temb, state=state[m_idx])
        m_idx += 1
        if self.attn_bottleneck:
            raise NotImplementedError("Bottleneck attention is currently not implemented for forward_step!")
        h, _ = self.bottleneck_modules['rnb2'].forward_step(h, temb, state=state[m_idx])
        m_idx += 1

        # Up path
        pyramid = None
        for l in reversed(range(self.num_resolutions)):
            nrb = self.num_res_blocks + (1 if l == 0 else 0)
            for k in range(nrb):
                h_input = self.up_modules[f'lvl{l}_combiner{k}'](hs_up.pop(), h)
                h, _ = self.up_modules[f'lvl{l}_rnb{k}'].forward_step(
                    h_input, temb, state=state[m_idx])
                m_idx += 1

            pyramid_h, _ = self.up_modules[f'lvl{l}_pyramid_normconv'].forward_step(
                h, state=state[m_idx])
            m_idx += 1

            if l != self.num_resolutions - 1:
                pyramid_up, _ = self.pyramid_upsample.forward_step(
                    pyramid, state=state[m_idx])
                m_idx += 1

                pyramid = pyramid_up + pyramid_h
            else:
                pyramid = pyramid_h

            if l != 0:
                h, _ = self.up_modules[f'lvl{l}_rnb_up'].forward_step(
                    h, temb, state=state[m_idx])
                m_idx += 1
            else:
                h = pyramid

        return h, state


### Model compression using decoupled 2D convs (Guo et al. 2018 https://arxiv.org/pdf/1808.05517)

def compress_decoupled_(model, K: int | Callable[[str, nn.Module], int], module_filter: Callable[[str, nn.Module], bool] = lambda *_: True):
    """
    Compress the model **in-place** by approximating all 2D convs with a parallel set of depthwise convs (grouped)
    followed by a pointwise conv, using a SVD-based kernel decomposition with K singular values.
    See Guo et al. 2018 "Network Decoupling": https://arxiv.org/pdf/1808.05517

    Args:
        - K is an int, or a callable that takes (name, module) as args and returns an int K for that module.
        - module_filter is an optional callable that also takes (name, module) as args and returns True (compress) or False (skip).

    Note: 'name' is always the full dotted "absolute" path to the module, e.g. "down_modules.lvl0_rnb0.CConv_0".
    """
    if hasattr(model, '_compressed_decoupled') and model._compressed_decoupled:
        print("Model is already compressed, skipping.")
        return

    for name, module in model.named_modules():
        if isinstance(module, CausalConv2d):
            if np.prod(module.kernel_size) == 1:
                print(f"NOT compressing {name} (1x1 conv)")
                continue
            if K > np.prod(module.kernel_size) or K > module.in_channels or K > module.out_channels:
                print(f"NOT compressing {name} ({K=} too large)")
                continue
            if not module_filter(name, module):
                print(f"NOT compressing {name} (filtered by module_filter)")
                continue

            k = K(name, module) if callable(K) else K
            print(f"Compressing {name} with K={k}")
            compressed = CausalDecoupledConv2d.from_causal_conv2d(module, K=k)
            # Replace the module in the parent
            parent = model
            *path, last = name.split('.')
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, compressed)

    # we store this flag in a registered buffer so it gets saved/loaded with the model state_dict
    model.register_buffer('_compressed_decoupled', torch.tensor(True))
    return model


### Helper blocks for the DNN

class CausalResnetBlockBigGANpp(nn.Module, CausalStreamingModule):
    def __init__(
        self, in_ch, out_ch, act, in_freqs, temb_dim=None,
        # dilation for conv0 and conv1, (1,1) by default (no dilation)
        dilation=(1, 1),
        # down or upsample along fre6quency using the FIR kernel
        freq_up=False, freq_down=False,
        fir_kernel=(1, 3, 3, 1),
        # number of frequency groups for the norm layer (default 4), unused if norm_type != 'subband_grouped_batchnorm'
        freq_groups = 4,
        # optional dropout (off by default)
        dropout=0.0,
        # norm type
        norm_type='subband_grouped_batchnorm',
        norm_kwargs=None,
        # rescale the skip connection by 1/sqrt(2)
        skip_rescale=True,
        # weight initialization scale for conv1 (default 1e-3)
        init_scale=1e-3,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.act = act
        self.in_freqs = in_freqs
        self.freq_groups = freq_groups
        self.temb_dim = temb_dim
        self.dilation = dilation
        self.freq_up = freq_up
        self.freq_down = freq_down
        self.dropout = dropout
        self.norm_type = norm_type
        if norm_kwargs is None:
            norm_kwargs = {}
        self.norm_kwargs = norm_kwargs
        self.skip_rescale = skip_rescale
        self.init_scale = init_scale

        self.down_or_upsample = DownOrUpSample(freq_down=freq_down, freq_up=freq_up, fir_kernel=fir_kernel)

        # FIXME perhaps rename, it's not always a groupnorm
        self.CGroupNorm_0 = instantiate_norm(
            norm_type,
            num_groups=max(1, min(in_ch // 4, 32)), num_channels=in_ch,
            num_freqs=in_freqs, freq_groups=freq_groups,
            **norm_kwargs)
        self.CConv_0 = causal_conv3x3(in_ch, out_ch, dilation=dilation)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        group1_freq_factor = 0.5 if freq_down else (2.0 if freq_up else 1.0)
        self.CGroupNorm_1 = instantiate_norm(
            norm_type,
            num_groups=max(1, min(out_ch // 4, 32)), num_channels=out_ch,
            num_freqs=int(in_freqs * group1_freq_factor), freq_groups=freq_groups,
            **norm_kwargs)
        self.Dropout_0 = nn.Dropout(dropout)
        self.CConv_1 = causal_conv3x3(out_ch, out_ch, dilation=(1,1), init_scale=init_scale)  # no extra dilation for CConv_1

        if in_ch != out_ch or freq_up or freq_down:
            self.CConv_2 = causal_conv1x1(in_ch, out_ch)
        else:
            self.CConv_2 = StreamingIdentity()

    def init_state(self) -> dict:
        assert isinstance(self.CGroupNorm_0, SubBandGroupedBatchNorm), "only SubBandGroupedBatchNorm currently implemented for streaming"
        assert isinstance(self.CGroupNorm_1, SubBandGroupedBatchNorm), "only SubBandGroupedBatchNorm currently implemented for streaming"

        input_freqs = self.in_freqs
        freqs = int(input_freqs * (0.5 if self.freq_down else (2.0 if self.freq_up else 1.0)))
        return (
            self.down_or_upsample.init_state(input_freqs=input_freqs),  # downsample_x
            self.down_or_upsample.init_state(input_freqs=input_freqs),  # downsample_h
            self.CConv_0.init_state(input_freqs=freqs),
            self.CConv_1.init_state(input_freqs=freqs),
            self.CConv_2.init_state(input_freqs=freqs) if isinstance(self.CConv_2, CausalStreamingModule) else None,
        )

    def forward_step(self, x: torch.Tensor, temb=None, *, state: Tuple) -> Tuple[torch.Tensor, Tuple]:
        h = self.act(self.CGroupNorm_0(x))

        state_down_x, state_down_h, state_conv_0, state_conv_1, state_conv_2, state_se = state

        h, state_down_h = self.down_or_upsample.forward_step(h, state=state_down_h)
        x, state_down_x = self.down_or_upsample.forward_step(x, state=state_down_x)
        h, state_conv_0 = self.CConv_0.forward_step(h, state=state_conv_0)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.CGroupNorm_1(h))
        h = self.Dropout_0(h)
        h, state_conv_1 = self.CConv_1.forward_step(h, state=state_conv_1)

        if hasattr(self, 'CConv_2'):
            x, state_conv_2 = self.CConv_2.forward_step(x, state=state_conv_2)

        new_state = (
            state_down_x, state_down_h, state_conv_0, state_conv_1, state_conv_2, state_se
        )

        if self.skip_rescale:
            return (x + h) / np.sqrt(2.), new_state
        else:
            return x + h, new_state

    def forward(self, x, temb=None):
        h = self.act(self.CGroupNorm_0(x))

        # Perform down/upsampling as requested
        h = self.down_or_upsample(h)
        x = self.down_or_upsample(x)

        # Convolve, add time embeddings, activate, dropout, then convolve again
        h = self.CConv_0(h)
        if temb is not None:
           h = h + self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.CGroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.CConv_1(h)

        # 1x1 conv for input x if needed to skip-connect (otherwise no-op)
        x = self.CConv_2(x)

        #if self.skip_rescale:
        return (x + h) / np.sqrt(2.)


# Convenient aliases just for clearer intention and model inspection
class CausalResnetBlock(CausalResnetBlockBigGANpp):
    pass
class CausalResnetBlockDown(CausalResnetBlockBigGANpp):
    pass
class CausalResnetBlockUp(CausalResnetBlockBigGANpp):
    pass
class CausalResnetBlockBottleneck(CausalResnetBlockBigGANpp):
    pass


# Pyramid norm-activation-convolution module (for the up path), see Song et al. 2021
class CausalPyramidNormConv(nn.Module):
    """
    Pyramid norm-activation-convolution module for the pyramid 'up path' of the U-Net.
    This does not itself perform upsampling, but is used after the upsampling layer.
    """
    def __init__(
        self, in_ch: int, out_ch: int, in_freqs: int,
        init_scale: float, act: nn.Module,
        norm_type: str = 'subband_grouped_batchnorm',
        norm_kwargs: Optional[dict] = None,
        freq_groups: int = 4,
    ):
        super().__init__()

        self.norm = instantiate_norm(
            norm_type=norm_type, num_channels=in_ch, num_groups=min(in_ch // 4, 32),
            num_freqs=in_freqs, freq_groups=freq_groups,
            **(norm_kwargs if norm_kwargs is not None else {}))
        self.conv = causal_conv3x3(in_ch, out_ch, bias=True, init_scale=init_scale)
        self.act = act

    def init_state(self, input_freqs: int) -> dict:
        assert isinstance(self.norm, SubBandGroupedBatchNorm), "streaming norms not implemented yet"
        return (
            self.conv.init_state(input_freqs=input_freqs),
        )

    def forward_step(self, x: torch.Tensor, *, state: dict) -> Tuple[torch.Tensor, dict]:
        x = self.norm(x)
        x = self.act(x)
        x, conv_state = self.conv.forward_step(x, state=state[0])
        return x, (conv_state,)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


# Down/Upsampling along frequency

class DownOrUpSample(nn.Module, CausalStreamingModule):
    def __init__(self, freq_down=False, freq_up=False, fir_kernel=(1,3,3,1)):
        super().__init__()
        self.freq_down = freq_down
        self.freq_up = freq_up

        if freq_up or freq_down:
            fir_kernel = torch.tensor(fir_kernel, dtype=torch.float32)
            fir_kernel /= fir_kernel.sum()
            self.register_buffer('fir_kernel', fir_kernel)

        assert not hasattr(self, 'lowpass'), "streaming lowpass not implemented yet"

    def forward(self, x):
        # Apply frequency-wise up/downsampling as requested
        if self.freq_up:
            x = upfirdn1d_freq(x, self.fir_kernel, up=2)
        elif self.freq_down:
            x = upfirdn1d_freq(x, self.fir_kernel, down=2)
        return x

    def init_state(self, input_freqs: int) -> dict:
        return ()  # no state needed as this is point-wise in time

    def forward_step(self, x: torch.Tensor, *, state: dict) -> Tuple[torch.Tensor, dict]:
        return self.forward(x), state


def upfirdn1d_freq(x, kernel, up=1, down=1):
    # TODO could be optimized?
    assert x.ndim == 4, "Input must be 4D (batch, channel, frequency, time)"
    (B, C, Fr, T) = x.shape
    assert kernel.ndim == 1, "Kernel must be 1D"

    out = x
    if up > 1:
        out = rearrange(out, 'b c f t -> b c f 1 t')
        # intersperse 0s along frequency
        out = F.pad(out, (0, 0, 0, up - 1), mode='constant', value=0.0)
        out = rearrange(out, 'b c f pad t -> b c (f pad) t')

    # Apply 'same' padding along fre6q
    if len(kernel) > 1:
        out = rearrange(out, 'b c fpad t -> (b c t) fpad')
        kernel_n = kernel.shape[0]
        pad_left = (kernel_n - 1) // 2
        pad_right = kernel_n - 1 - pad_left
        out = F.pad(out, (pad_left, pad_right), mode='constant', value=0.0)
        out = rearrange(out, 'bct fpad -> bct 1 fpad')
        # Run the FIR kernel
        w = torch.flip(kernel, dims=(0,)).view(1, 1, kernel_n)
        out = F.conv1d(out, w)
        out = rearrange(out, '(b c t) 1 fpad -> b c fpad t', b=B, c=C, t=T)
    # Decimate along frequency
    if down > 1:
        out = out[:, :, ::down, :]

    return out


class CausalConv2d(nn.Conv2d, CausalStreamingModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int,int],
                 dilation: Tuple[int,int] = (1,1), stride: Tuple[int, int] = (1,1),
                 bias: bool = True):
        assert stride[1] == 1, "Only stride=1 in time is supported (causal conv)"

        # Only pad along time on the left for simulated causality. On frequency we do symmetric padding
        pad_time = (kernel_size[1] - 1) * dilation[1]
        pad_freq = (kernel_size[0] - 1) // 2 * dilation[0]
        time_padding = (pad_time, 0, 0, 0)

        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, bias=bias,
            padding=(pad_freq, 0),  # only pad freq in the conv itself, time padding is handled in forward/forward_step
        )

        self.dilation = dilation
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_padding = time_padding
        self.pad_freq = pad_freq

        d_t, k_t = self.dilation[1], self.kernel_size[1]
        self.Tbuf = 1 + d_t * (k_t - 1)

    def forward(self, x):
        # We pad along time manually here since this is only sensible or necessary in non-streaming mode
        # Also we have to separately pad along time and frequency since we want to use different modes :)
        x = F.pad(x, self.time_padding)
        x = super().forward(x)
        return x

    def init_state(self, input_freqs: int) -> Tuple:
        device, dtype = self.weight.device, self.weight.dtype

        # we use d*(k-1) past frames and an extra entry for the newest input frame
        Tbuf = self.Tbuf
        xbuf_shape = (self.in_channels, input_freqs, Tbuf)
        return (
            torch.zeros(xbuf_shape, dtype=dtype, device=device),
        )

    def forward_step(self, x: torch.Tensor, *, state: Tuple) -> Tuple[torch.Tensor, Tuple]:
        B, C, Fr, T = x.shape
        xbuf, = state

        # shift buffer to the left by one frame. potential for further optimization here?
        xbuf[..., :-1] = xbuf[..., 1:].clone()
        xbuf[..., :, -1] = x[0, :, :, 0].clone()

        # Run the conv, produces a single output frame
        xbuf_in = xbuf.view(1, C, Fr, -1)
        h = super().forward(xbuf_in)
        if self.depthwise_separable:
            h = self.pointwise_conv(h)
        return h, (xbuf,)


class CausalDecoupledConv2d(nn.Module, CausalStreamingModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 K: int,
                 dilation = (1,1),
                 stride = (1,1),
                 padding = (0,0),
                 time_padding = (0,0,0,0),
                 bias: bool = True):
        super().__init__()
        assert stride[1] == 1, "Only stride=1 in time is supported (causal conv)"
        assert K <= np.prod(kernel_size), "K must be smaller than <= total kernel size (product of all kernel dims)"

        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels*K,
            kernel_size=kernel_size,
            groups=in_channels,  # each input channel independent
            bias=False,  # bias is absorbed in pointwise conv
            padding=padding,
            stride=stride,
            dilation=dilation
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels*K,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )

        d_t, k_t = dilation[1], kernel_size[1]
        self.Tbuf = 1 + d_t * (k_t - 1)
        self.time_padding = time_padding

    @classmethod
    def from_causal_conv2d(cls, conv: CausalConv2d, K: int):
        """
        Layer compressoin method described in https://arxiv.org/pdf/1808.05517, slightly adapted for CausalConv2d.
        Performs an SVD-based grouped approximation of the original CausalConv2d weights.

        Uses K <= k_h*k_w singular values per input channel.
        (side note: in general it is just required that K <= min(C_out, k_h*k_w), but we assume C_out >> k_h*k_w here)
        """
        W = conv.weight.data.clone()  # (C_out, C_in, kH, kW)
        C_out, C_in, kH, kW = W.shape
        assert K <= kH * kW, f"K must be <= kH*kW ({kH*kW})"

        depthwise_kernels = []
        pointwise_kernels = []
        for i in range(C_in):
            # see https://arxiv.org/pdf/1808.05517, page 5
            Wi = W[:, i, :, :].reshape(C_out, kH*kW)
            U, S_vals, Vh = torch.li6nalg.svd(Wi, full_matrices=False)
            # for downstream training stability we spread sqrt(S) over both U and V, instead of S on only one of them
            U_r = U[:, :K] @ torch.diag(S_vals[:K].sqrt())
            V_r = torch.diag(S_vals[:K].sqrt()) @ Vh[:K, :]
            depthwise_kernels.append(V_r.unsqueeze(0))
            pointwise_kernels.append(U_r)
        # assemble and reshape depthwise kernels
        depthwise_weights = torch.cat(depthwise_kernels, dim=0)
        depthwise_weight = depthwise_weights.reshape(C_in*K, 1, kH, kW)
        # assemble and reshape pointwise kernels
        pointwise_weight = torch.cat(pointwise_kernels, dim=1).unsqueeze(-1).unsqueeze(-1)

        use_bias = conv.bias is not None
        instance = cls(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=conv.kernel_size,
            K=K,
            dilation=conv.dilation,
            stride=conv.stride,
            padding=conv.padding,
            time_padding=conv.time_padding,
            bias=use_bias
        )
        # Overwrite uninitialized weights with calculated decomposed depthwise & pointwise weights
        instance.depthwise_conv.weight.data = depthwise_weight
        instance.pointwise_conv.weight.data = pointwise_weight
        if use_bias:
            instance.pointwise_conv.bias.data = conv.bias.data.clone()
        return instance

    def forward(self, x):
        # We pad along time manually here since this is only sensible or necessary in non-streaming mode
        # Also we have to separately pad along time and frequency since we want to use different modes :)
        x = F.pad(x, self.time_padding)
        x = self.pointwise_conv(self.depthwise_conv(x))
        return x

    def init_state(self, input_freqs: int) -> Tuple:
        W = self.depthwise_conv.weight
        device, dtype = W.device, W.dtype

        # we use d*(k-1) past frames and an extra entry for the newest input frame
        Tbuf = self.Tbuf
        xbuf_shape = (self.depthwise_conv.in_channels, input_freqs, Tbuf)
        return (
            torch.zeros(xbuf_shape, dtype=dtype, device=device),
        )

    def forward_step(self, x: torch.Tensor, *, state: Tuple) -> Tuple[torch.Tensor, Tuple]:
        B, C, Fr, T = x.shape
        xbuf, = state

        # shift buffer left
        xbuf[..., :-1] = xbuf[..., 1:].clone()
        xbuf[..., :, -1] = x[0, :, :, 0].clone()

        # Run the conv, produces a single output frame
        xbuf_in = xbuf.view(1, C, Fr, -1)
        h = self.pointwise_conv(self.depthwise_conv(xbuf_in))
        return h, (xbuf,)


### Norms

def instantiate_norm(norm_type, num_channels, num_groups, num_freqs, freq_groups, **kwargs):
    if norm_type == 'cumulative':
        return CausalCumulativeGroupNorm(num_groups, num_channels, **kwargs)
    elif norm_type == 'subband_grouped_batchnorm':
        return SubBandGroupedBatchNorm(num_channels, num_freqs=num_freqs,
                                       channel_groups=num_groups, freq_groups=freq_groups, **kwargs)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, **{'eps': 1e-6, **kwargs})
    elif norm_type == 'none':
        return StreamingIdentity()
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


class CausalCumulativeGroupNorm(torch.nn.Module):
    """
    Cumulative Group Normalization that is causal along time.

    Note that this is an IIR filter but not a time-invariant one:
        due to the 1/(n+1) in the mean where n is the current time index
        the filter response changes as n increases!
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b (g cg) f t -> b g cg f t", g=self.num_groups)
        cum_sum = x.mean(dim=(-3,-2)).cumsum(dim=-1)
        cum_sum_sq = x.square().mean(dim=(-3,-2)).cumsum(dim=-1)
        cum_counts = torch.arange(1, x.size(-1)+1).type_as(x).expand_as(cum_sum)
        cum_mean = cum_sum/cum_counts
        cum_var = cum_sum_sq/cum_counts - cum_mean.square()
        cum_mean = rearrange(cum_mean, "b g t -> b g 1 1 t")
        cum_var = rearrange(cum_var, "b g t -> b g 1 1 t")
        x_n = (x - cum_mean) / (cum_var + self.eps).sqrt()
        x_n = rearrange(x_n, "b g cg f t -> b (g cg) f t", g=self.num_groups)
        if self.affine:
            return x_n * self.weight + self.bias
        return x_n


def next_smallest_divisor(N, D0):
    """
    Finds the next smallest divisor of N that is less than D0.
    Helper function for frequency-splitting at low frequency resolution (large UNet depth).

    Inefficient but should work, only called with small N and D0 usually, and only once during model init
    """
    for d in range(D0 - 1, 0, -1):
        if N % d == 0:
            return d
    return 1


class SubBandGroupedBatchNorm(nn.Module):
    """
    Sub-band grouped batch normalization for 2D time-frequency data (grouped along channels and frequencies).

    This is like a BatchNorm in that statistics are only collected during training and not based on each instance,
    and like a like a GroupNorm in that it is grouped along channels. Additionally, we group along frequency
    into sub-bands, see "SubSpectral Normalization for Neural Audio Data Processing" (https://arxiv.org/abs/2103.13620).

    The BatchNorm-like statistics aggregation during training is helpful for streaming inference
    since we are not dependent on the statistics of an instance, and don't need to do
    running statistics based on time which may be suboptiomal for audio data (e.g. blowup of silent regions)
    """
    def __init__(self, num_channels, num_freqs, channel_groups=1, freq_groups=1, eps=1e-5, affine=True, momentum=0.1):
        super().__init__()
        assert num_channels % channel_groups == 0, f"{num_channels=} is not divisible by {channel_groups=}"

        if num_freqs % freq_groups != 0:
            freq_groups, orig_freq_groups = next_smallest_divisor(num_freqs, freq_groups), freq_groups
            warnings.warn(f"num_freqs {num_freqs} is not divisible by freq_groups {orig_freq_groups}, "
                          f"using next smallest divisor {freq_groups} instead")

        self.C = num_channels
        self.F = num_freqs
        self.cg = channel_groups
        self.fg = freq_groups
        self.c = self.C // self.cg  # channels per group
        self.f = self.F // self.fg
        self.eps = eps
        self.affine = affine
        self.momentum = momentum

        if affine:
            # One weight and bias per channel and frequency group (sub-band)
            # This is a very specific choice, not sure about it yet but seems right intuitively
            self.weight = nn.Parameter(torch.ones(self.C, self.fg))
            self.bias = nn.Parameter(torch.zeros(self.C, self.fg))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # statistics are aggregated within each channel-frequency group, similar to a GroupNorm
        self.register_buffer('running_mean', torch.zeros(self.cg, self.fg))
        self.register_buffer('running_var', torch.ones(self.cg, self.fg))

    def forward(self, x):
        B, C, Fr, T = x.shape
        x = x.view(B, self.cg, self.c, self.fg, self.f, T)

        if self.training:
            # fused var+mean over (B, c, f, T)
            var, mean = torch.var_mean(x, dim=(0, 2, 4, 5), unbiased=False, keepdim=False)
            # update running stats
            with torch.no_grad():
                self.running_mean[:] = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
                self.running_var[:] = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        # reshape mean/var to broadcast shape
        mean = mean.view(1, self.cg, 1, self.fg, 1, 1)
        var = var.view(1, self.cg, 1, self.fg, 1, 1)
        inv_std = (var + self.eps).rsqrt()
        x = (x - mean) * inv_std

        if self.affine:
            # weight/bias are (C, fg), so reshape to (1, cg, c, fg, 1, 1)
            weight = self.weight.view(self.cg, self.c, self.fg).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            bias = self.bias.view(self.cg, self.c, self.fg).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            x = x * weight + bias

        return x.view(B, C, Fr, T)


### Network weight initialization, taken from NCSN++ code

def variance_scaling(scale, mode, distribution,
                                         in_axis=1, out_axis=0,
                                         dtype=torch.float32,
                                         device='cpu'):
    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


### Causal Components for 2D (causal along time, not necessarily along frequency)

def causal_conv3x3(in_channels, out_channels, init_scale=1., **kwargs):
    """3x3 causal convolution with DDPM initialization"""
    conv = CausalConv2d(in_channels, out_channels, kernel_size=(3,3), **kwargs)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    if hasattr(conv, 'bias') and conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def causal_conv1x1(in_channels, out_channels, init_scale=1., **kwargs):
    """1x1 causal convolution with DDPM initialization"""
    # Bit silly to have this, 1x1 is always causal. But the intention may be a bit clearer
    conv = CausalConv2d(in_channels, out_channels, kernel_size=(1,1), **kwargs)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    if hasattr(conv, 'bias') and conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


if __name__ == "__main__":
    from thop import profile

    # =========================================================================
    # Causality Test
    # =========================================================================
    def test_causality(F: int = 32, T: int = 16, n_trials: int = 5):
        print(f"\n--- Causality test  [CAUSAL]  F={F}, T={T}, {n_trials} random trials ---")

        torch.manual_seed(42)
        model = CausalNCSNpp(
            nf=16, ch_mult=(1, 1), num_res_blocks=1,
            input_channels=4, output_channels=2, input_freqs=F,
            norm_type="subband_grouped_batchnorm",
            no_freq_groups_below=16,
        ).eval()

        all_pass = True
        for trial in range(n_trials):
            seed = trial * 1337
            torch.manual_seed(seed)

            # Fresh random inputs and a random split point (2 .. T-2)
            x = torch.randn(1, 1, F, T, dtype=torch.cfloat)
            y = torch.randn(1, 1, F, T, dtype=torch.cfloat)
            split = torch.randint(2, T - 1, (1,)).item()

            with torch.no_grad():
                out_full = model(x, y)

                x_masked = x.clone()
                y_masked = y.clone()
                x_masked[:, :, :, split:] = 0.0
                y_masked[:, :, :, split:] = 0.0
                out_masked = model(x_masked, y_masked)

            max_diff = (out_full[:, :, :, :split] - out_masked[:, :, :, :split]).abs().max().item()
            ok = max_diff == 0.0
            all_pass = all_pass and ok

            status = "PASS ✓" if ok else "FAIL ✗"
            print(f"  trial {trial+1}/{n_trials}  seed={seed}  split={split:2d}  "
                  f"max_diff={max_diff:.3e}  [{status}]")

        verdict = "ALL PASS ✓" if all_pass else "SOME FAIL ✗"
        print(f"  → {verdict}")
        return all_pass

    causal_ok = test_causality()

    print("\n=== Summary ===")
    print(f"  model is causal:     {'YES ✓' if causal_ok else 'NO  ✗'}")

    # =========================================================================
    # Complexity Count
    # =========================================================================
    def count_macs_custom(model, inputs):
        macs = 0
        hooks = []
        
        def conv2d_hook(module, input, output):
            nonlocal macs
            if not hasattr(module, 'weight'): return
            out_shape = output.shape
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            groups = module.groups
            m = (out_shape[2] * out_shape[3] * (in_channels // groups) * out_channels * kernel_size[0] * kernel_size[1])
            macs += m

        def linear_hook(module, input, output):
            nonlocal macs
            if not hasattr(module, 'weight'): return
            out_features = module.out_features
            in_features = module.in_features
            macs += input[0].shape[0] * in_features * out_features

        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(conv2d_hook))
            elif isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(linear_hook))
                
        with torch.no_grad():
            model(*inputs)
            
        for h in hooks:
            h.remove()
            
        return macs

    configs = [
        {"name": "CH64",  "nf": 64,  "ch_mult": (1, 2, 2, 2), "num_res_blocks": 1, "input_channels": 4, "no_freq_groups_below": 16},
        {"name": "CH1282", "nf": 128, "ch_mult": (1, 2, 2, 2), "num_res_blocks": 2, "input_channels": 4, "no_freq_groups_below": 16},
        {"name": "CH1281", "nf": 128, "ch_mult": (1, 2, 2, 2), "num_res_blocks": 1, "input_channels": 4, "no_freq_groups_below": 16},
    ]

    for cfg in configs:
        model = CausalNCSNpp(
            nf=cfg["nf"],
            ch_mult=cfg["ch_mult"],
            num_res_blocks=cfg["num_res_blocks"],
            input_channels=cfg["input_channels"],
            output_channels=2,
            input_freqs=256,
            norm_type="subband_grouped_batchnorm",
            no_freq_groups_below=cfg["no_freq_groups_below"],
        ).eval()
        
        print("\n" + "-" * 50)
        print(f">>> Model Complexity Profile: CausalNCSN++ ({cfg['name']}) <<<")
        
        # Calculate Parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_m = total_params / 1e6
        trainable_params_m = trainable_params / 1e6

        # Calculate MACs (1s audio at 16kHz, hop=256 -> F=256, T=64)
        num_input_complex = cfg["input_channels"] // 2
        x_in = torch.randn(1, num_input_complex, 256, 64, dtype=torch.cfloat)
        
        # Real tensor for custom hook MACs (x_real has 4 channels)
        x_real = torch.randn(1, cfg["input_channels"], 256, 64, dtype=torch.float32)

        try:
            # Profile with thop using complex inputs
            macs_thop, params_thop = profile(model, inputs=(x_in,), verbose=False)
            mac_string_thop = f"{macs_thop / 1e9:.2f} G MACs"
        except Exception as e:
            mac_string_thop = f"N/A (Error: {e})"
            params_thop = 0

        # Profile with custom hook
        try:
            macs_custom = count_macs_custom(model, (x_real,))
            mac_string_custom = f"{macs_custom / 1e9:.2f} G MACs"
        except Exception as e:
            mac_string_custom = f"N/A (Error: {e})"

        print(f"Total Parameters (PyTorch): {total_params_m:.2f} M")
        print(f"Total Parameters (THOP):    {params_thop / 1e6:.2f} M")
        print(f"Total MACs (THOP)   [64 frames]: {mac_string_thop}")
        print(f"Total MACs (Custom) [64 frames]: {mac_string_custom}")
        print("-" * 50)
    print("")
