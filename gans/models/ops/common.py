import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class Pad(nn.Module):
    def __init__(self, padding, ring=False, mode="replicate"):
        super().__init__()
        self.padding = _quadruple(padding)
        self.horizontal = "circular" if ring else mode
        self.vertical = mode

    def forward(self, h):
        left, right, top, bottom = self.padding
        h = F.pad(h, (left, right, 0, 0), mode=self.horizontal)
        h = F.pad(h, (0, 0, top, bottom), mode=self.vertical)
        return h

    def extra_repr(self):
        return f"padding={self.padding}, horizontal={self.horizontal}, vertical={self.vertical}"


def filter2d(x, kernel, gain=1):
    assert kernel.ndim == 1
    kernel = kernel / kernel.sum()
    kernel = kernel * (gain ** (kernel.ndim / 2))
    fh = fw = len(kernel)
    pw0 = fw // 2
    pw1 = (fw - 1) // 2
    ph0 = fh // 2
    ph1 = (fh - 1) // 2
    x = F.pad(x, (pw0, pw1, 0, 0), mode="circular")
    x = F.pad(x, (0, 0, ph0, ph1), mode="replicate")
    B, C, H, W = x.shape
    kernel = kernel[None, None].repeat(C, 1, 1).to(dtype=x.dtype)
    x = F.conv2d(x, kernel[..., None, :], groups=C)
    x = F.conv2d(x, kernel[..., :, None], groups=C)
    return x


class Resample(nn.Module):
    def __init__(
        self,
        up=1,
        down=1,
        window=[1, 3, 3, 1],  # bilinear
        ring=True,
        normalize=True,
        direction="hw",
    ):
        super().__init__()
        self.up = np.asarray(_pair(up))
        self.down = np.asarray(_pair(down))
        self.window = window
        self.n_taps = len(window)
        self.ring = ring
        self.pad_mode_w = "circular" if ring else "replicate"
        self.pad_mode_h = "replicate"
        self.normalize = normalize
        self.direction = direction
        assert self.direction in ("h", "w", "hw")

        # setup sizes
        if "h" in self.direction:
            self.k_h = self.n_taps
            self.up_h = self.up[0]
            self.down_h = self.down[0]
        else:
            self.k_h = self.up_h = self.down_h = 1

        if "w" in self.direction:
            self.k_w = self.n_taps
            self.up_w = self.up[1]
            self.down_w = self.down[1]
        else:
            self.k_w = self.up_w = self.down_w = 1

        # setup filter
        kernel = torch.tensor(self.window, dtype=torch.float32)
        if self.normalize:
            kernel /= kernel.sum()
        kernel *= (self.up_h * self.up_w) ** (kernel.ndim / 2)
        self.register_buffer("kernel", kernel)

        # setup padding
        if self.up[0] > 1:
            self.ph0 = (self.k_h - self.up_h + 1) // 2 + self.up_h - 1
            self.ph1 = (self.k_h - self.up_h) // 2
        elif self.down[0] >= 1:
            self.ph0 = (self.k_h - self.down_h + 1) // 2
            self.ph1 = (self.k_h - self.down_h) // 2
        if self.up[1] > 1:
            self.pw0 = (self.k_w - self.up_w + 1) // 2 + self.up_w - 1
            self.pw1 = (self.k_w - self.up_w) // 2
        elif self.down[1] >= 1:
            self.pw0 = (self.k_w - self.down_w + 1) // 2
            self.pw1 = (self.k_w - self.down_w) // 2

        self.margin = max(self.ph0, self.ph1, self.pw0, self.pw1)

    def forward(self, h):
        # margin
        h = F.pad(h, (self.margin, self.margin, 0, 0), mode=self.pad_mode_w)
        h = F.pad(h, (0, 0, self.margin, self.margin), mode=self.pad_mode_h)
        # up by zero-insertion
        B, C, H, W = h.shape
        h = h.view(B, C, H, 1, W, 1)
        h = F.pad(h, [0, self.up_w - 1, 0, 0, 0, self.up_h - 1])
        h = h.view(B, C, H * self.up_h, W * self.up_w)
        # crop
        h = h[
            ...,
            self.margin * self.up_h
            - self.ph0 : (H - self.margin) * self.up_h
            + self.ph1,
            self.margin * self.up_w
            - self.pw0 : (W - self.margin) * self.up_w
            + self.pw1,
        ]
        # fir
        kernel = self.kernel[None, None].repeat(C, 1, 1).to(dtype=h.dtype)
        if self.direction == "hw":
            h = F.conv2d(h, kernel[..., None, :], groups=C)
            h = F.conv2d(h, kernel[..., :, None], groups=C)
        elif self.direction == "h":
            h = F.conv2d(h, kernel[..., :, None], groups=C)
        elif self.direction == "w":
            h = F.conv2d(h, kernel[..., None, :], groups=C)
        # down
        h = h[:, :, :: self.down_h, :: self.down_w]
        return h

    def extra_repr(self):
        return f'filter_type={self.window}, up={self.up}, down={self.down}, direction="{self.direction}"'


class BlurVH(nn.Module):
    """
    vertical/horizontal antialiasing from NR-GAN:
    https://arxiv.org/abs/1911.11776
    """

    def __init__(self, window=[1, 2, 1], ring=True):
        super().__init__()
        self.blur_v = Resample(window=window, ring=ring, direction="h")
        self.blur_h = Resample(window=window, ring=ring, direction="w")

    def forward(self, x):
        h_v = self.blur_v(x)
        h_h = self.blur_h(x)
        return torch.cat([h_v, h_h], dim=1)


class EqualLR(nn.Module):
    """
    A wrapper for runtime weight scaling (equalized learning rate).
    https://arxiv.org/abs/1710.10196
    """

    def __init__(self, module, gain: float = 1.0, lr_mul=1.0):
        super().__init__()
        self.module = module
        self.gain = gain
        self.lr_mul = lr_mul

        # Runtime scale factor
        self.gain_ = gain * lr_mul
        fan_in = self.module.weight[0].numel()
        self.scale = 1.0 / math.sqrt(fan_in)

        # Weights are initialized with N(0, 1)
        nn.init.normal_(self.module.weight, 0.0, 1.0 / lr_mul)
        if hasattr(self.module, "bias") and self.module.bias is not None:
            nn.init.constant_(self.module.bias, 0.0)

    def forward(self, x):
        return self.module(x * self.scale) * self.gain_

    def extra_repr(self):
        return f"gain={self.gain}, lr_mul={self.lr_mul}"


class Conv2d(nn.Sequential):
    """
    Custom padding + Conv2d + Equal LR
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
        bias=True,
        ring=False,
        equal_lr=False,
        gain=1.0,
        lr_mul=1.0,
    ):
        layers = []
        if padding != 0:
            layers += [Pad(padding=padding, ring=ring)]
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, 0, bias=bias)
        layers += [EqualLR(conv, gain, lr_mul) if equal_lr else conv]
        super().__init__(*layers)


class PixelNorm(nn.Module):
    """
    Pixel-wise normalization.
    https://arxiv.org/abs/1710.10196
    Hypersphare projection in 1D case.
    https://arxiv.org/abs/1912.04958
    """

    def forward(self, x, alpha: float = 1e-8):
        y = x.pow(2.0).mean(dim=1, keepdim=True).add(alpha).sqrt()
        return x / y


class MinibatchStdDev(nn.Module):
    """
    Minibatch discrimination.
    https://arxiv.org/abs/1710.10196
    """

    def __init__(self, group=4, features=1):
        super().__init__()
        self.group = group
        self.features = features

    def forward(self, x, alpha: float = 1e-8):
        B, C, H, W = x.shape
        group = min(B, self.group)
        # Split minibatch into groups
        y = x.view(group, -1, self.features, C // self.features, H, W)
        # Calculate stddev over group
        y = torch.sqrt(y.var(0, unbiased=False) + alpha)
        # Take average over fmaps and pixels
        y = y.mean([2, 3, 4], keepdim=True).squeeze(2)
        # [B // group, features, 1, 1]
        y = y.repeat(group, 1, H, W)
        # [B, features, H, W]
        y = torch.cat([x, y], dim=1)
        return y

    def extra_repr(self):
        return f"group={self.group}, features={self.features}"


class Dilation(nn.Module):
    def __init__(self, dilation=1, value=0):
        super().__init__()
        self.dilation = dilation
        self.value = value
        self.stride = self.dilation + 1
        kernel = F.pad(torch.ones(1, 1, 1, 1), (self.dilation,) * 4, value=self.value)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        _, C, _, _ = x.shape
        kernel = self.kernel.repeat(C, 1, 1, 1)
        return F.conv_transpose2d(x, kernel, stride=self.stride, padding=1, groups=C)

    def extra_repr(self):
        return f"dilation={self.dilation}, value={self.value}"


def init_weights(layer, mode, gain=1.0):
    for name, module in layer.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if mode == "ortho":
                nn.init.orthogonal_(module.weight, gain)
            elif mode == "N02":
                nn.init.normal_(module.weight, 0, 0.02)
            elif mode in ["glorot", "xavier"]:
                nn.init.xavier_uniform_(module.weight, gain)
            else:
                NotImplementedError
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif "BatchNorm" in name:
            if mode == "N02":
                torch.nn.init.normal_(module.weight, 1.0, 0.02)
                torch.nn.init.zeros_(module.bias)
