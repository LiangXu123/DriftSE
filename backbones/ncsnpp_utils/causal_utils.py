# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

# from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import torch.nn.functional as F
import functools
import torch
import numpy as np
from .op import upfirdn2d, upfirdn2d_freq
from . import up_or_down_sampling
from . import layerspp

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

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

class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)            
        
        self.kernel_size = kernel_size
        self.stride = stride
        # For input shape [batch, channels, freq, time]:
        # - kernel_size[0] operates on frequency dimension
        # - kernel_size[1] operates on time dimension
        # - We want causal padding in time (last dimension)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, dilation=dilation,
            padding=(kernel_size[0]//2, kernel_size[1]-1)  # (freq_padding, time_padding)
        )
    
    def forward(self, x):
        # x.shape: [batch, channels, freq, time]
        out = self.conv(x)
        # Truncate the time dimension to maintain causality
        if self.kernel_size[1] >1:
            truncate_amount = (self.kernel_size[1] - 1) // self.stride[1]
            if truncate_amount > 0:
                out = out[:, :, :, :-truncate_amount]
        # Add this after truncation operations
        # print(f"out.requires_grad = {out.requires_grad}")
        # print(f"x.requires_grad = {x.requires_grad}")
        # assert out.requires_grad == x.requires_grad, "Gradient requirement changed!"
        return out
    
        # out.requires_grad = True          
        # x.requires_grad = False
        # AssertionError: Gradient requirement changed!

class CausalGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.channels_per_group = num_channels // num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        # x: [B, C, F, T]
        B, C, F, T = x.shape
        
        # Reshape to group format: [B, G, Cg, F, T]
        # print(f"x.shape = {x.shape}, num_groups = {self.num_groups}, channels_per_group = {self.channels_per_group}")

        x_grouped = x.view(B, self.num_groups, self.channels_per_group, F, T)
        
        # For each group, we want to compute statistics over the spatial dimensions (Cg, F)
        # Flatten spatial dimensions: [B, G, Cg*F, T]
        x_flat = x_grouped.view(B, self.num_groups, -1, T)
        
        # Compute cumulative sums over time for each spatial element
        cumsum = x_flat.cumsum(dim=-1)  # [B, G, Cg*F, T]
        cumsum2 = (x_flat ** 2).cumsum(dim=-1)  # [B, G, Cg*F, T]
        
        # At each time step t, we have seen (t+1) time steps and (Cg*F) spatial elements
        # So total count at time t is (t+1) * (Cg*F)
        spatial_size = self.channels_per_group * F
        time_counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
        counts = time_counts.view(1, 1, 1, T) * spatial_size
        
        # Group-wise cumulative statistics (average over spatial dimensions)
        group_cumsum = cumsum.sum(dim=2, keepdim=True)  # [B, G, 1, T]
        group_cumsum2 = cumsum2.sum(dim=2, keepdim=True)  # [B, G, 1, T]
        
        # Compute group means and variances
        group_mean = group_cumsum / counts  # [B, G, 1, T]
        group_var = group_cumsum2 / counts - group_mean ** 2
        
        # Broadcast group statistics back to spatial dimensions
        group_mean = group_mean.expand(-1, -1, spatial_size, -1)  # [B, G, Cg*F, T]
        group_var = group_var.expand(-1, -1, spatial_size, -1)    # [B, G, Cg*F, T]
        
        # Normalize
        x_norm = (x_flat - group_mean) / torch.sqrt(group_var + self.eps)
        
        # Reshape back to original format
        x_norm = x_norm.view(B, self.num_groups, self.channels_per_group, F, T)
        x_norm = x_norm.view(B, C, F, T)
        
        # Apply affine transformation
        weight = self.weight.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)
        
        return x_norm * weight + bias

class CumulativeGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.channels_per_group = num_channels // num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: [B, C, F, T]
        B, C, F, T = x.shape

        # Reshape to grouped format: [B, G, Cg, F, T]
        x_grouped = x.view(B, self.num_groups, self.channels_per_group, F, T)

        # Flatten spatial dimensions: [B, G, Cg*F, T]
        x_flat = x_grouped.view(B, self.num_groups, -1, T)

        # Compute cumulative sums along time dimension
        cumsum = x_flat.cumsum(dim=-1)
        cumsum_sq = (x_flat ** 2).cumsum(dim=-1)

        # Calculate cumulative count
        spatial_size = self.channels_per_group * F
        counts = spatial_size * torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, T)

        # Compute cumulative mean and variance
        mean = cumsum.sum(dim=2, keepdim=True) / counts
        var = cumsum_sq.sum(dim=2, keepdim=True) / counts - mean ** 2

        # Normalize
        x_norm = (x_flat - mean) / torch.sqrt(var + self.eps)

        # Reshape to original dimensions
        x_norm = x_norm.view(B, self.num_groups, self.channels_per_group, F, T)
        x_norm = x_norm.view(B, C, F, T)

        # Apply affine transformation
        weight = self.weight.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)

        return x_norm * weight + bias


class CausalDownsample(nn.Module):
    def __init__(self, channels, factor=2, downsample_freq=True, downsample_time=False):
        super().__init__()
        self.factor = factor
        self.downsample_freq = downsample_freq
        self.downsample_time = downsample_time
        
        # Determine kernel size and stride based on what we want to downsample
        if downsample_freq and downsample_time:
            kernel_size = (factor, factor)
            stride = (factor, factor)
        elif downsample_freq:
            kernel_size = (factor, 1)
            stride = (factor, 1)
        elif downsample_time:
            kernel_size = (1, factor)
            stride = (1, factor)
        else:
            raise ValueError("Must downsample at least one dimension")
        
        self.downsample = CausalConv2d(
            channels, channels, 
            kernel_size=kernel_size, 
            stride=stride
        )
    
    def forward(self, x):
        return self.downsample(x)

class CausalUpsample(nn.Module):
    def __init__(self, channels, factor=2, upsample_freq=True, upsample_time=False):
        super().__init__()
        self.factor = factor
        self.upsample_freq = upsample_freq
        self.upsample_time = upsample_time
        
        # Determine kernel size and stride based on what we want to upsample
        if upsample_freq and upsample_time:
            kernel_size = (factor, factor)
            stride = (factor, factor)
            padding = (factor-1, factor-1)
        elif upsample_freq:
            kernel_size = (factor, 1)
            stride = (factor, 1)
            padding = (factor-1, 0)
        elif upsample_time:
            kernel_size = (1, factor)
            stride = (1, factor) 
            padding = (0, factor-1)  # Causal padding for time dimension
        else:
            raise ValueError("Must upsample at least one dimension")
        
        self.upsample = nn.ConvTranspose2d(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    
    def forward(self, x):
        # x.shape: [batch, channels, freq, time]
        B, C, F, T = x.shape
        out = self.upsample(x)
        
        # Handle causal truncation for time dimension upsampling
        if self.upsample_time:
            expected_time = T * self.factor
            if out.size(3) > expected_time:
                # Truncate excess time steps to maintain causality
                out = out[:, :, :, :expected_time]
        
        return out
    
def causal_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    """3x3 causal convolution with DDPM-style initialization."""
    conv = CausalConv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation)
    conv.conv.weight.data = default_init(init_scale)(conv.conv.weight.shape)
    if bias:
        nn.init.zeros_(conv.conv.bias)
    return conv

def causal_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
    """1x1 causal convolution with DDPM-style initialization."""
    conv = CausalConv2d(in_planes, out_planes, kernel_size=1, stride=stride)
    conv.conv.weight.data = default_init(init_scale)(conv.conv.weight.shape)
    if bias:
        nn.init.zeros_(conv.conv.bias)
    return conv

def _setup_kernel(k):
  k = np.asarray(k, dtype=np.float32)
  if k.ndim == 1:
    k = np.outer(k, k)
  k /= np.sum(k)
  assert k.ndim == 2
  assert k.shape[0] == k.shape[1]
  return k



def causal_conv1x3(in_planes, out_planes, bias=True, dilation=1, init_scale=1.):
    """
    Causal 1x3 convolution:
    - 1 in freq, 3 in time
    - stride 2 in time
    - causal padding in time (left only)
    """
    conv = CausalConv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=(1, 3),
        stride=(1, 2),
        dilation=dilation
    )

    # DDPM-style init
    conv.conv.weight.data = default_init(init_scale)(conv.conv.weight.shape)
    if bias:
        nn.init.zeros_(conv.conv.bias)

    return conv


class CausalConvTranspose2d(nn.Module):
    """
    Causal 1x3 transposed convolution for upsampling the time axis by 2x.
    - 1 in freq, 3 in time
    - stride 2 in time
    - causal output: output at time t depends only on input up to t
    """
    def __init__(self, in_channels, out_channels, bias=True, dilation=1, init_scale=1.):
        super().__init__()
        self.kernel_size = (1, 3)
        self.stride = (1, 2)
        self.dilation = dilation

        # Padding is set to 0; output_padding ensures correct length
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),            # No padding, we'll trim after
            output_padding=(0, 1),     # Ensures doubling of the time axis
            dilation=self.dilation,
            bias=bias
        )
        # DDPM-style init
        self.conv_transpose.weight.data = default_init(init_scale)(self.conv_transpose.weight.shape)
        if bias:
            nn.init.zeros_(self.conv_transpose.bias)

    def forward(self, x):
        # x: [batch, channels, freq, time]
        out = self.conv_transpose(x)
        # To ensure causality, trim the rightmost (future) time positions that depend on future input
        trim = self.kernel_size[1] - 1
        if trim > 0:
            out = out[..., :-trim]
        return out

def transpose_causal_conv1x3(in_planes, out_planes, bias=True, dilation=1, init_scale=1.):
    """
    Returns a causal 1x3 transposed convolution that upsamples time by 2x.
    """
    return CausalConvTranspose2d(
        in_channels=in_planes,
        out_channels=out_planes,
        bias=bias,
        dilation=dilation,
        init_scale=init_scale
    )


def causal_upsample_2d(x, k=None, factor=2, gain=1.0, up_causal_conv=None):
    """
    Causally upsample 2D input [B, C, Freq, Time]:
    - FIR + upsample in freq axis (dim=2)
    - causal conv + upsample in time axis (dim=3)

    Args:
        x:     Tensor of shape [B, C, Freq, Time]
        k:     1D FIR kernel (list or tensor) for freq axis
        factor: Upsampling factor (default: 2)
        gain:  Scaling factor for signal magnitude (default: 1.0)
        up_causal_conv: Causal Conv2d module for time upsampling (must upsample T)

    Returns:
        Tensor of shape [B, C, Freq * factor, Time * factor]
    """
    assert x.ndim == 4  # [B, C, F, T]
    B, C, F, T = x.shape

    if k is None:
        k = [1] * factor

    # Normalize and scale kernel
    k = torch.tensor(k, dtype=x.dtype, device=x.device)
    k = k / k.sum() * (gain * factor)
    k = k.view(-1, 1)  # [K, 1] → vertical kernel for freq

    # Compute padding for freq axis
    p = k.shape[0] - factor
    pad0 = (p + 1) // 2 + factor - 1
    pad1 = p // 2

    # --- Freq upsample ---
    freq_upsampled = upfirdn2d_freq(x, kernel=k, up=factor, pad=(pad0, pad1))
    # print(f"freq_upsampled.shape = {freq_upsampled.shape}")

    # --- Time upsample ---
    assert up_causal_conv is not None, "You must provide a causal conv for time upsampling"
    out = up_causal_conv(freq_upsampled)
    # print(f"upsampled out.shape = {out.shape}")

    return out


def causal_downsample_2d(x, k=None, factor=2, gain=1.0, down_causal_conv=None):
    """
    Downsample only frequency axis (dim=2) of x ∈ [B, C, F, T],
    keeping time (dim=3) unchanged. Applies FIR filtering + stride.

    Args:
        x:     Tensor of shape [B, C, Freq, Time]
        k:     1D kernel (list or tensor), default is average filter
        factor: Downsampling factor for freq
        gain:  Scaling factor for signal magnitude

    Returns:
        Tensor of shape [B, C, Freq//factor, Time]
    """
    assert x.ndim == 4
    B, C, F, T = x.shape

    if k is None:
        k = [1] * factor

    k = torch.tensor(k, dtype=x.dtype, device=x.device)
    k = k / k.sum() * gain
    k = k.view(-1, 1)  # shape [K, 1] — vertical kernel (freq only)

    # Compute padding: causal-like (bias toward past)
    p = k.shape[0] - factor
    pad0 = (p + 1) // 2  # pad on top
    pad1 = p // 2        # pad on bottom

    # Call adapted upfirdn2d_freq (which only applies to freq axis)
    freq_down_sampled = upfirdn2d_freq(x, kernel=k, down=factor, pad=(pad0, pad1))
    # next we will use causal conv2d to downsample the time axis
    # print(f"freq_down_sampled.shape = {freq_down_sampled.shape}")
    out = down_causal_conv(freq_down_sampled)
    # print(f"out.shape = {out.shape}")
    return out


class CausalResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0., use_causal_conv=True):
    super().__init__()
    self.up_causal_conv = None
    self.down_causal_conv = None      
    out_ch = out_ch if out_ch else in_ch
    self.up = up
    self.down = down
    if use_causal_conv:
        # norm_func = CausalGroupNorm
        norm_func = CumulativeGroupNorm
        conv3x3 = causal_conv3x3
        conv1x1 = causal_conv1x1
    else:
        norm_func = nn.GroupNorm
        conv3x3 = layerspp.conv3x3
        conv1x1 = layerspp.conv1x1    

    self.use_causal_updownsample = use_causal_conv and (self.down or self.up)
    if self.use_causal_updownsample:
        if self.down:
            self.down_causal_conv = causal_conv1x3(out_ch, out_ch)           
        if self.up:
            self.up_causal_conv = transpose_causal_conv1x3(out_ch, out_ch)
      
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = norm_func(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch
    self.GroupNorm_0 = norm_func(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.middle_feature = None

  def forward(self, x, temb=None, return_middle_feature = False):
    h = self.act(self.GroupNorm_0(x))
    if self.use_causal_updownsample:
      if self.up:
            h = causal_upsample_2d(h, self.fir_kernel, factor=2, up_causal_conv = self.up_causal_conv)
            x = causal_upsample_2d(x, self.fir_kernel, factor=2, up_causal_conv = self.up_causal_conv)
            # print(f"upsampled out.shape = {x.shape}")
      elif self.down:
            # print(f"before x.shape = {x.shape}, h.shape = {h.shape}")
            h = causal_downsample_2d(h, self.fir_kernel, factor=2, down_causal_conv = self.down_causal_conv)
            x = causal_downsample_2d(x, self.fir_kernel, factor=2, down_causal_conv = self.down_causal_conv)
            # print(f"downsample x.shape = {x.shape}")
    else:
      if self.up:
        # print(f"before x.shape = {x.shape}, h.shape = {h.shape}")
        if self.fir:
          h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
          x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
        else:
          h = up_or_down_sampling.naive_upsample_2d(h, factor=2) # repeat on axis by 2
          x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        # print(f"after x.shape = {x.shape}, h.shape = {h.shape}")
      elif self.down:
        # print(f"before x.shape = {x.shape}, h.shape = {h.shape}")
        if self.fir:
          h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
          x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
        else:
          h = up_or_down_sampling.naive_downsample_2d(h, factor=2) # mean on axis by 2
          x = up_or_down_sampling.naive_downsample_2d(x, factor=2)       
# downsample x.shape = torch.Size([1, 128, 128, 128])                                                                   
# downsample x.shape = torch.Size([1, 128, 64, 64])
# downsample x.shape = torch.Size([1, 256, 32, 32])                                                                                                                                                                                           
# downsample x.shape = torch.Size([1, 256, 16, 16])
# downsample x.shape = torch.Size([1, 256, 8, 8])                                                                                                                                                                                             
# downsample x.shape = torch.Size([1, 256, 4, 4])                                                                       
# upsampled out.shape = torch.Size([1, 256, 8, 8])                                                                      
# upsampled out.shape = torch.Size([1, 256, 16, 16])
# upsampled out.shape = torch.Size([1, 256, 32, 32])                                                                    
# upsampled out.shape = torch.Size([1, 256, 64, 64])                                                                    
# upsampled out.shape = torch.Size([1, 256, 128, 128])                                                                  
# upsampled out.shape = torch.Size([1, 128, 256, 256])  
    # print(f"x.requires_grad = {x.requires_grad}")
    # print(f"h.requires_grad = {h.requires_grad}")
    assert h.requires_grad == x.requires_grad, "Gradient requirement changed!"
    
    h = self.Conv_0(h)
    # this is for debug only
    if return_middle_feature:
        self.middle_feature = h.detach().clone()        
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
      # print(f"temb.requires_grad = {temb.requires_grad}")
      # temb.requires_grad = True

    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)
    # print(f"x2.requires_grad = {x.requires_grad}")
    # print(f"h2.requires_grad = {h.requires_grad}")
    assert h.requires_grad == x.requires_grad, "Gradient requirement changed!"
    if not self.skip_rescale:
      if return_middle_feature:
         return x + h, self.middle_feature
      return x + h
    else:
      if return_middle_feature:
         return (x + h) / np.sqrt(2.), self.middle_feature
      return (x + h) / np.sqrt(2.)
    

