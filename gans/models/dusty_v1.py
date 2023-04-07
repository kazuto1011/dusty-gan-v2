import torch
from torch import nn

from . import base, ops, vanilla


class RayDropModel(nn.Module):
    def __init__(
        self,
        raydrop_const: float,
        gumbel_temperature: float,
    ):
        super().__init__()
        self.gumbel_sigmoid = ops.GumbelSigmoid(
            temperature=gumbel_temperature,
            straight_through=True,
        )
        self.register_buffer("raydrop_const", torch.tensor(float(raydrop_const)))

    def forward(self, h):
        assert isinstance(h, dict) and ("image" in h) and ("raydrop_logit" in h)
        h["raydrop_mask"] = self.gumbel_sigmoid(h["raydrop_logit"])
        h["image_orig"] = h["image"]
        h["image"] = h["image"].lerp(self.raydrop_const, 1 - h["raydrop_mask"])
        return h

    def extra_repr(self):
        return f"raydrop_const={self.raydrop_const}"


class Generator(base.Generator):
    def __init__(self, synthesis_kwargs, measurement_kwargs):
        super().__init__(
            mapping_network=nn.Identity(),
            synthesis_network=vanilla.SynthesisNetwork(**synthesis_kwargs),
            measurement_model=RayDropModel(**measurement_kwargs),
        )

    def forward_synthesis(self, w, angles=None):
        o = self.synthesis_network(w)
        return o
