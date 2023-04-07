import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

from .common import EqualLR


class ModConv2d(nn.Module):
    """
    Modulated convolution layer.
    https://arxiv.org/abs/1912.04958
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mod_ch: int,
        ksize: int = 3,
        stride: int = 1,
        padding: int = 1,
        demod: bool = True,
        bias: bool = True,
        gain: float = 1.0,
        transposed: bool = False,
        factorization_rank=None,
        ema=False,
        ema_decay=0.9989,  # 0.5**(32/20_000)
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mod_ch = mod_ch
        self.ksize = _pair(ksize)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        # convolution parameters
        self.weight = nn.Parameter(torch.randn((1, out_ch, in_ch, *self.ksize)))
        self.transposed = transposed
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, out_ch, 1, 1)))
        else:
            self.bias = None

        # runtime scaling
        self.gain = gain
        self.scale = 1.0 / np.sqrt(in_ch * np.prod(self.ksize))  # kaiming normal

        # modulation
        self.factorization_rank = factorization_rank
        if factorization_rank is None:
            self.mod = EqualLR(nn.Linear(mod_ch, in_ch), gain=1.0)
        else:
            mod_out_ch = int((in_ch + out_ch) * self.factorization_rank)
            self.mod = EqualLR(nn.Linear(mod_ch, mod_out_ch), gain=1.0)
        self.demod = demod

        # ema
        self.ema = ema
        self.ema_decay = ema_decay
        self.register_buffer("ema_var", torch.tensor(1.0))

    def forward(self, x, style):
        B, _, H, W = x.shape
        dtype = x.dtype
        # modulation
        style = self.mod(style)
        weight = self.scale * self.weight.to(dtype=dtype)

        # pre-normalize
        if self.demod:
            # 'self.scale' is redundant
            weight = weight / weight.norm(float("inf"), dim=[1, 2, 3], keepdim=True)
            style = style / style.norm(float("inf"), dim=1, keepdim=True)

        if self.factorization_rank is None:
            # StyleGAN2
            style = style.view(B, 1, self.in_ch, 1, 1) + 1.0
        else:
            # INR-GAN
            left = style[:, : self.out_ch * self.factorization_rank]
            right = style[:, self.out_ch * self.factorization_rank :]
            left = left.view(B, self.out_ch, self.factorization_rank)
            right = right.view(B, self.factorization_rank, self.in_ch)
            style = torch.sigmoid(left @ right)[..., None, None]
        weight = weight * style  # [B,O,I,K,K]

        # demodulation
        if self.demod:
            r_norm = torch.rsqrt(weight.pow(2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * r_norm

        # ema
        if self.ema:
            if self.training:
                var = x.pow(2).mean((0, 1, 2, 3))
                self.ema_var = self.ema_var.lerp(var.detach(), 1 - self.ema_decay)
            weight = weight / (torch.sqrt(self.ema_var) + 1e-8)

        # reshape minibatch to conv groups
        x = x.view(1, B * self.in_ch, H, W)

        if self.transposed:
            weight = weight.transpose(1, 2)
            weight = weight.reshape(B * self.in_ch, self.out_ch, *self.ksize)
            h = F.conv_transpose2d(x, weight, None, self.stride, self.padding, groups=B)
        else:
            weight = weight.view(B * self.out_ch, self.in_ch, *self.ksize)
            h = F.conv2d(x, weight, None, self.stride, self.padding, groups=B)

        # reshape conv groups back to minibatch
        _, _, H, W = h.shape
        h = h.view(B, self.out_ch, H, W)

        if self.bias is not None:
            h = h + self.bias.to(dtype=dtype)

        if self.gain != 1.0:
            h = h * self.gain

        return h

    def extra_repr(self):
        return (
            f"in_ch={self.in_ch}, out_ch={self.out_ch}, mod_ch={self.mod_ch}, "
            + f"ksize={self.ksize}, stride={self.stride}, padding={self.padding}, "
            + f"demod={self.demod}, gain={self.gain}, transposed={self.transposed}"
        )


class NoiseInjection(nn.Module):
    """
    Spatial noise injection.
    https://arxiv.org/abs/1912.04958
    if 'ch' == incoming ch: each channel has a unique weight (StyleGAN)
    if 'ch' == 1: all channels are equally scaled (StyleGAN2)
    """

    def __init__(self, ch: int = 1):
        super().__init__()
        self.ch = ch
        self.weight = nn.Parameter(torch.zeros(1, self.ch, 1, 1))
        self.fixed_noise = None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.fixed_noise is None:
            noise = torch.randn((B, 1, H, W), device=x.device, dtype=x.dtype)
        else:
            noise = self.fixed_noise  # (1, 1, H, W)
            noise = noise.expand(B, -1, -1, -1)
        return x + self.weight * noise

    def extra_repr(self):
        return f"ch={self.ch}"
