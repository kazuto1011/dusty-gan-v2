import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair

from . import base, dusty_v1, ops


class MappingNetwork(nn.Sequential):
    def __init__(self, in_ch, out_ch, depth=2):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.depth = depth

        layers = [ops.PixelNorm()]
        ch = self.in_ch
        for _ in range(depth):
            layers.append(
                nn.Sequential(
                    ops.EqualLR(nn.Linear(ch, out_ch), gain=math.sqrt(2), lr_mul=0.01),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )
            ch = out_ch
        super().__init__(*layers)


class Head(nn.Module):
    def __init__(self, in_ch, mod_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.mod_ch = mod_ch
        self.out_ch = out_ch
        self.heads = nn.ModuleDict()
        for o in out_ch:
            if o["ch"] == 0:
                continue
            self.heads[o["name"]] = ops.ModConv2d(
                out_ch=o["ch"],
                in_ch=in_ch,
                mod_ch=mod_ch,
                ksize=1,
                stride=1,
                padding=0,
                demod=False,
                ema=True,
            )

    def forward(self, x, style):
        h = {}
        for name, head in self.heads.items():
            h[name] = head(x, style)
        return h


class SynthesisBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        mid_ch,
        out_ch,
        mod_ch,
        resolution,
        up=2,
        resample_dir="hw",
        resample_window=[1, 3, 3, 1],
        use_noise=True,
        use_pe=True,
        pe_type="random",
        pe_ch=512,
        pe_scale_offset=(3, -1),
        ring=True,
    ):
        super().__init__()
        self.use_pe = use_pe
        self.use_fp16 = False
        self.is_first = in_ch == 0
        self.num_conv = 0

        if up > 1:
            self.resample = ops.Resample(
                up=up,
                window=resample_window,
                ring=ring,
                direction=resample_dir,
            )
            self.downsample = ops.Resample(
                down=up,
                window=resample_window,
                ring=ring,
                direction=resample_dir,
            )
        else:
            self.resample = nn.Identity()
            self.downsample = None

        if self.use_pe:
            self.pe = ops.FourierFeature(
                resolution=resolution,
                basis_scale=pe_type,
                num_freqs=pe_ch,
                L_offset=pe_scale_offset,
            )
            pe_ch = self.pe.out_ch
        else:
            pe_ch = 0

        conv_kwargs = dict(
            out_ch=mid_ch,
            mod_ch=mod_ch,
            ksize=1,
            stride=1,
            padding=0,
            bias=False,
            ema=True,
        )

        self.conv1 = ops.ModConv2d(in_ch=in_ch + pe_ch, **conv_kwargs)
        self.noise1 = ops.NoiseInjection() if use_noise else None
        self.bias_act1 = ops.FusedLeakyReLU(mid_ch)
        self.num_conv += 1

        if not self.is_first:
            self.conv2 = ops.ModConv2d(in_ch=mid_ch, **conv_kwargs)
            self.noise2 = ops.NoiseInjection() if use_noise else None
            self.bias_act2 = ops.FusedLeakyReLU(mid_ch)
            self.num_conv += 1

        self.head = Head(mid_ch, mod_ch, out_ch)

    def downsample_angle(self, angle):
        _, C, _, _ = angle.shape
        periodic = torch.cat([angle.sin(), angle.cos()], dim=1)
        periodic = self.downsample(periodic)
        angle = torch.atan2(periodic[:, :C], periodic[:, C:])
        return angle

    def forward(self, h, skip, ws, angle):
        ws = iter(ws)

        dtype = (
            torch.float16
            if self.use_fp16 and angle.device.type == "cuda"
            else torch.float32
        )

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            if h is not None:
                h = self.resample(h)
            else:
                h = torch.empty(0, device=angle.device, dtype=dtype)

            if self.use_pe:
                h_pe = self.pe(angle)
                h = torch.cat([h, h_pe], dim=1)

            h = self.conv1(h, next(ws))
            if self.noise1 is not None:
                h = self.noise1(h)
            h = self.bias_act1(h)

            if not self.is_first:
                h = self.conv2(h, next(ws))
                if self.noise2 is not None:
                    h = self.noise2(h)
                h = self.bias_act2(h)

            o = self.head(h, next(ws))

        for k in o.keys():
            o[k] = o[k].to(dtype=torch.float32)
            if skip is not None:
                o[k] = o[k] + self.resample(skip[k])
            assert o[k].dtype == torch.float32

        return h, o

    def extra_repr(self):
        return f"use_fp16={self.use_fp16}"


class SynthesisNetwork(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        ch_base=64,
        ch_max=512,
        resolution=(64, 256),
        ring=True,
        layers=[2, 2, 2, 2],
        num_fp16_layers=-1,
        use_noise=True,
        pe_type="random",
        pe_scale_offset=(3, -1),
        aug_coords=True,
        aug_coords_blitting=False,
        output_scale=1 / 4.0,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.resolution_out = np.array(_pair(resolution))
        self.resolution_in = self.resolution_out // np.prod(layers)

        self.layers = nn.ModuleList()
        resolution_i = self.resolution_in
        ch = lambda i: min(ch_base << (len(layers) - i), ch_max)
        for i, scale in enumerate([1] + layers):
            resolution_i *= scale
            self.layers.append(
                SynthesisBlock(
                    in_ch=ch(i - 1) if i != 0 else 0,
                    mid_ch=ch(i),
                    out_ch=out_ch,
                    mod_ch=in_ch,
                    resolution=resolution_i,
                    up=scale,
                    resample_window=[1, 3, 3, 1],
                    use_noise=use_noise,
                    use_pe=scale > 1 or i == 0,
                    pe_type=pe_type,
                    pe_scale_offset=pe_scale_offset,
                    ring=ring,
                )
            )

        for i, m in enumerate(self.layers[::-1]):
            if i < num_fp16_layers or num_fp16_layers == -1:
                m.use_fp16 = True

        self.num_styles = len(self.layers) * 2
        self.aug_coords = aug_coords
        self.aug_coords_blitting = aug_coords_blitting
        self.output_scale = output_scale

        self.output_acts = nn.ModuleDict(
            [
                o["name"],
                nn.Identity()
                if o["act"] is None
                else (eval(o["act"])() if isinstance(o["act"], str) else o["act"]()),
            ]
            for o in out_ch
        )

    @staticmethod
    def translation_matrix(t):
        B, _ = t.shape
        t = t.div(2 * np.pi)  # [0, 2pi] -> [0, 1]
        mat = torch.eye(3, device=t.device)[None].repeat_interleave(B, dim=0)
        mat[:, 0, 2] = t[:, 1]
        mat[:, 1, 2] = t[:, 0]
        return mat

    def forward(self, ws, angle):
        B, N, _ = ws.shape
        assert N == self.num_styles, f"{self.num_styles} != {N}"
        aug_coords = self.training and self.aug_coords

        # Random shifting for subgrid consistency
        if aug_coords:
            _, W = self.resolution_out
            shifts = torch.zeros((B, 2), device=ws.device)  # [h, w]
            shifts[:, 1].uniform_(0, 1)  # horizontal only
            if self.aug_coords_blitting:  # blitting
                shifts[:, 1].mul_(W).round_().div_(W)
            shifts = shifts.mul(2 * np.pi)  # [0,1] -> [0,2pi]
            angle = angle + shifts[..., None, None]

        # Repeatedly downsample by 2
        multiscale_angle = [angle]
        for layer in self.layers[:0:-1]:
            if layer.downsample is not None:
                angle = layer.downsample_angle(angle)
            multiscale_angle = [angle] + multiscale_angle
        assert len(multiscale_angle) == len(self.layers)

        # Forward convs
        h, skip, i = None, None, 0
        for layer, angle in zip(self.layers, multiscale_angle):
            h, skip = layer(h, skip, (ws[:, i], ws[:, i + 1], ws[:, i + 2]), angle)
            i += layer.num_conv

        # Cancelling the horizontal shifts in image space
        if aug_coords:
            T = self.translation_matrix(shifts)
            for k, v in skip.items():
                v = torch.cat([v, v], dim=3)  # circular
                grid = F.affine_grid(T[:, :2], v.shape, align_corners=False)
                v = F.grid_sample(v, grid, mode="bilinear", align_corners=False)
                skip[k] = v[..., :W]

        # From stylegan3
        for k, v in skip.items():
            skip[k] = v * self.output_scale

        # Final activations
        for k, v in skip.items():
            if k in self.output_acts:
                skip[k] = self.output_acts[k](v)

        return skip


class Generator(base.Generator):
    def __init__(self, mapping_kwargs, synthesis_kwargs, measurement_kwargs):
        super().__init__(
            mapping_network=MappingNetwork(**mapping_kwargs),
            synthesis_network=SynthesisNetwork(**synthesis_kwargs),
            measurement_model=dusty_v1.RayDropModel(**measurement_kwargs),
        )

    def forward_synthesis(self, w, angle=None):
        angle = self.angle if angle is None else angle
        o = self.synthesis_network(w, angle)
        return o


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        kwargs = dict(bias=False, ring=True, equal_lr=True)
        self.conv1 = ops.Conv2d(in_ch, in_ch, 3, 1, 1, **kwargs)
        self.bias_act1 = ops.FusedLeakyReLU(in_ch)
        self.resample = ops.Resample(window=[1, 3, 3, 1], ring=True)
        self.conv2 = ops.Conv2d(in_ch, out_ch, 3, 2, 1, **kwargs)
        self.bias_act2 = ops.FusedLeakyReLU(out_ch)
        self.skip = ops.Conv2d(in_ch, out_ch, 1, 2, 0, **kwargs)

    def residual(self, x):
        h = self.conv1(x)
        h = self.bias_act1(h)
        h = self.conv2(self.resample(h))
        h = self.bias_act2(h)
        return h

    def forward(self, x):
        h = self.residual(x) + self.skip(self.resample(x))
        return h / math.sqrt(2)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_ch: int,
        ch_base: int = 32,
        ch_max: int = 512,
        mbdis_group: int = 4,
        mbdis_feat: int = 1,
        resolution=(64, 512),
        ring=True,
        num_fp16_layers=-1,
        pre_blur=True,
    ):
        super().__init__()

        # calculate an output shape and the num of layers
        resolution_in = _pair(256 if resolution is None else resolution)
        n_downsample = int(np.log2(min(resolution_in) / 4))
        resolution_out = tuple(map(lambda x: x >> n_downsample, resolution_in))
        ch = lambda i: min(ch_base << i, ch_max)
        conv_kwargs = dict(bias=False, ring=ring, equal_lr=True)
        self.num_fp16_layers = num_fp16_layers

        in_ch = in_ch * 2 if pre_blur else in_ch
        layers = [ops.BlurVH(ring=ring)] if pre_blur else []
        layers += [ops.Conv2d(in_ch, ch(0), 1, 1, 0, **conv_kwargs)]
        layers += [ops.FusedLeakyReLU(ch(0))]
        layers += [ResidualBlock(ch(i), ch(i + 1)) for i in range(n_downsample)]
        self.layers = nn.Sequential(*layers)
        self.epilogue = nn.Sequential(
            ops.MinibatchStdDev(group=mbdis_group, features=mbdis_feat),
            ops.Conv2d(ch(4) + mbdis_feat, ch(4), 3, 1, 1, **conv_kwargs),
            ops.FusedLeakyReLU(ch(4)),
            nn.Flatten(),
            ops.EqualLR(nn.Linear(ch(4) * np.prod(resolution_out), ch(4), bias=False)),
            ops.FusedLeakyReLU(ch(4)),
            ops.EqualLR(nn.Linear(ch(4), 1)),
        )

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            use_fp16 = (self.num_fp16_layers > i) or (self.num_fp16_layers == -1)
            use_fp16 = use_fp16 and (h.device.type == "cuda")
            dtype = torch.float16 if use_fp16 else torch.float32
            with torch.cuda.amp.autocast(enabled=use_fp16):
                h = layer(h.to(dtype=dtype))
        h = h.to(dtype=torch.float32)
        h = self.epilogue(h)
        return h
