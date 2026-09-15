"""Layers used for up-sampling or down-sampling images.

Many functions are ported from https://github.com/NVlabs/stylegan2.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(
        input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


def _setup_kernel(k):
  k = np.asarray(k, dtype=np.float32)
  if k.ndim == 1:
    k = np.outer(k, k)
  k /= np.sum(k)
  assert k.ndim == 2
  assert k.shape[0] == k.shape[1]
  return k


def _shape(x, dim):
  return x.shape[dim]


def upsample_2d(x, k=None, factor=2, gain=1):
  r"""Upsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and upsamples each image with the given filter. The filter is normalized so
    that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the upsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          nearest-neighbor upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]`
  """
  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * (gain * (factor ** 2))
  p = k.shape[0] - factor
  return upfirdn2d(x, torch.tensor(k, device=x.device),
                   up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
  r"""Downsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and downsamples each image with the given filter. The filter is normalized
    so that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the downsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]`
  """

  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * gain
  p = k.shape[0] - factor
  return upfirdn2d(x, torch.tensor(k, device=x.device),
                   down=factor, pad=((p + 1) // 2, p // 2))
