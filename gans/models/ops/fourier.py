import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common as ops


class FourierFeature(nn.Module):
    def __init__(
        self,
        resolution,
        basis_scale="random",
        num_freqs=512,
        L_offset=(3, -1),
        mapping=False,
        mapping_ch=64,
    ):
        super().__init__()
        self.resolution = resolution

        self.L_h = int(np.ceil(np.log2(self.resolution[0]))) + L_offset[0]
        self.L_w = int(np.ceil(np.log2(self.resolution[1]))) + L_offset[1]

        band_h = 2 ** (self.L_h - 1)
        band_w = 2 ** (self.L_w - 1)
        self.max_band = (band_h**2 + band_w**2) ** 0.5

        if basis_scale in ("random",):
            freqs_h = torch.empty(num_freqs // 2, 1).uniform_(-band_h, band_h)
            freqs_w = 2 ** np.arange(self.L_w)
            freqs_w = list(-freqs_w) + [0] + list(freqs_w)
            freqs_w = np.random.choice(freqs_w, size=(num_freqs // 2, 1))
            freqs_w = torch.from_numpy(freqs_w)
            phase = torch.rand(num_freqs // 2) * 2 * np.pi
            freqs = torch.cat([freqs_h, freqs_w], dim=-1)
            self.register_buffer("freqs", freqs[..., None, None])
            self.register_buffer("phase", phase)
        # elif basis_scale in ("random_2",):
        #     freqs_h = torch.empty(num_freqs // 2, 1).uniform_(-band_h, band_h)
        #     freqs_w = np.arange(band_w)
        #     freqs_w = list(-freqs_w) + [0] + list(freqs_w)
        #     freqs_w = np.random.choice(freqs_w, size=(num_freqs // 2, 1))
        #     freqs_w = torch.from_numpy(freqs_w)
        #     phase = torch.rand(num_freqs // 2) * 2 * np.pi
        #     freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        #     self.register_buffer("freqs", freqs[..., None, None])
        #     self.register_buffer("phase", phase)
        # elif basis_scale in ("logscale",):
        #     L_min = min(self.L_h, self.L_w)
        #     freqs_h = torch.arange(self.L_h).exp2()
        #     freqs_h_diag = torch.cat([-freqs_h[:L_min], freqs_h[:L_min]])
        #     freqs_h = torch.cat([freqs_h, torch.zeros(self.L_w), freqs_h_diag])
        #     freqs_w = torch.arange(self.L_w).exp2()
        #     freqs_w_diag = torch.cat([freqs_w[:L_min], freqs_w[:L_min]])
        #     freqs_w = torch.cat([torch.zeros(self.L_h), freqs_w, freqs_w_diag])
        #     freqs = torch.stack([freqs_h, freqs_w], dim=-1)
        #     phase = torch.zeros(len(freqs_h))
        #     self.register_buffer("freqs", freqs[..., None, None])
        #     self.register_buffer("phase", phase)
        else:
            raise ValueError(basis_scale)

        self.basis_ch = int(len(freqs_h) * 2)

        if mapping:
            self.out_ch = mapping_ch
            self.mapping = ops.EqualLR(
                nn.Conv2d(self.basis_ch, self.out_ch, 1, 1, 0, bias=False)
            )
        else:
            self.out_ch = self.basis_ch
            self.mapping = None

    def forward(self, angles):
        coords = F.conv2d(angles, weight=self.freqs, bias=self.phase)
        encoded = torch.cat([coords.sin(), coords.cos()], dim=1)
        if self.mapping is not None:
            encoded = self.mapping(encoded)
        return encoded

    def extra_repr(self):
        return f"shape={self.resolution}, num_freqs={self.basis_ch}, L=({self.L_h}, {self.L_w})"
