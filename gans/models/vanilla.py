from einops.layers.torch import Rearrange
from torch import nn

from . import base, ops


class Projection(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__(
            Rearrange("B 1 C -> B C 1 1"),
            ops.EqualLR(nn.ConvTranspose2d(in_ch, out_ch, kernel, 1, 0, bias=False)),
            ops.FusedLeakyReLU(out_ch),
        )


class Upsample(nn.Sequential):
    def __init__(self, in_ch, out_ch, ring=True):
        super().__init__(
            ops.Pad(padding=1, ring=ring, mode="reflect"),
            ops.EqualLR(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 3, bias=False)),
            ops.FusedLeakyReLU(out_ch),
        )


class Head(nn.Module):
    def __init__(self, in_ch, out_ch, ring=True):
        super().__init__()
        self.in_ch = in_ch
        self.heads = nn.ModuleDict()
        for o in out_ch:
            if o["ch"] == 0:
                continue
            self.heads[o["name"]] = nn.Sequential(
                ops.Pad(padding=1, ring=ring, mode="reflect"),
                ops.EqualLR(nn.ConvTranspose2d(in_ch, o["ch"], 4, 2, 3, bias=True)),
                nn.Identity()
                if o["act"] is None
                else (eval(o["act"])() if isinstance(o["act"], str) else o["act"]()),
            )

    def forward(self, x):
        h = {}
        for name, head in self.heads.items():
            h[name] = head(x)
        return h


class SynthesisNetwork(nn.Sequential):
    def __init__(
        self,
        in_ch,
        out_ch,
        ch_base=64,
        ch_max=512,
        resolution=(64, 256),
        ring=True,
    ):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_styles = 1
        resolution_in = (resolution[0] >> 4, resolution[1] >> 4)
        ch = lambda i: min(ch_base << i, ch_max)
        super().__init__(
            Projection(in_ch, ch(3), resolution_in),
            Upsample(ch(3), ch(2), ring),
            Upsample(ch(2), ch(1), ring),
            Upsample(ch(1), ch(0), ring),
            Head(ch(0), out_ch, ring),
        )


class Generator(base.Generator):
    def __init__(self, synthesis_kwargs):
        super().__init__(
            mapping_network=nn.Identity(),
            synthesis_network=SynthesisNetwork(**synthesis_kwargs),
            measurement_model=nn.Identity(),
        )

    def forward_synthesis(self, w, angles=None):
        o = self.synthesis_network(w)
        return o


class Downsample(nn.Sequential):
    def __init__(self, in_ch, out_ch, ring=True):
        super().__init__(
            ops.Pad(padding=1, ring=ring, mode="reflect"),
            ops.EqualLR(nn.Conv2d(in_ch, out_ch, 4, 2, 0, bias=False)),
            ops.FusedLeakyReLU(out_ch),
        )


class Discriminator(nn.Sequential):
    def __init__(self, in_ch, ch_base=64, ch_max=512, resolution=(64, 256), ring=True):
        resolution_out = (resolution[0] >> 4, resolution[1] >> 4)
        ch = lambda i: min(ch_base << i, ch_max)
        super().__init__(
            ops.BlurVH(window=[1, 2, 1], ring=ring),
            Downsample(in_ch * 2, ch(0), ring),
            Downsample(ch(0), ch(1), ring),
            Downsample(ch(1), ch(2), ring),
            Downsample(ch(2), ch(3), ring),
            ops.EqualLR(nn.Conv2d(ch(3), 1, resolution_out, 1, 0)),
        )
