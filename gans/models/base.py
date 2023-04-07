import random

import torch
from torch import nn


class Generator(nn.Module):
    """StyleGAN-style templte"""

    def __init__(
        self,
        mapping_network: nn.Module = nn.Identity(),
        synthesis_network: nn.Module = nn.Identity(),
        measurement_model: nn.Module = nn.Identity(),
        w_avg_decay: float = 0.995,
    ) -> None:
        super().__init__()

        self.mapping_network = mapping_network
        self.synthesis_network = synthesis_network
        self.measurement_model = measurement_model

        self.w_avg_decay = w_avg_decay
        self.register_buffer("w_avg", torch.zeros(1, self.synthesis_network.in_ch))

    def forward(
        self,
        z: torch.Tensor,
        angle: torch.Tensor | None = None,
        style_mixing: bool = False,
        truncation_psi: float = 1.0,
        input_w: bool = False,
    ) -> dict[str, torch.Tensor]:
        """_summary_

        Args:
            z (torch.Tensor): latent code
            angles (torch.Tensor, optional): anglular grid. Defaults to None.
            style_mixing (bool, optional): Defaults to False.
            truncation_psi (float, optional): Defaults to 1.0.
            input_w (bool, optional): Defaults to False.

        Returns:
            dict[str, torch.Tensor]: dictionary of `coord`, `coord_orig`, `noise_logit`, `mask`, and `styles`.
        """

        if input_w:
            w = z
        else:
            w = self.forward_mapping(z, style_mixing)
        assert w.ndim == 3  # (B,N,D)

        if self.training:
            self.moving_average_w(w)
        else:
            w = self.truncation_trick(w, truncation_psi)

        o = self.forward_synthesis(w, angle)
        o["w"] = w

        o = self.forward_measurement(o)

        return o

    def forward_mapping(
        self, z: torch.Tensor, style_mixing: bool = False
    ) -> torch.Tensor:
        """Forward a mapping network.

        Args:
            z (torch.Tensor): latent codes
            style_mixing (bool, optional): Defaults to False.

        Returns:
            torch.Tensor: style codes
        """

        if style_mixing:
            w1 = self.mapping_network(z)
            w2 = self.mapping_network(torch.randn_like(z))
            n = random.randint(1, self.synthesis_network.num_styles)
            ws = [w1] * n + [w2] * (self.synthesis_network.num_styles - n)
        else:
            w = self.mapping_network(z)
            ws = [w] * self.synthesis_network.num_styles

        return torch.stack(ws, dim=1)

    def moving_average_w(self, w: torch.Tensor) -> None:
        """Compute a moving average of style `w`.

        Args:
            w (torch.Tensor): input style codes.
        """

        batch_mean = w[:, 0].mean(dim=0, keepdim=True).data.to(self.w_avg)
        self.w_avg = torch.lerp(self.w_avg, batch_mean, 1 - self.w_avg_decay)

    def truncation_trick(self, w: torch.Tensor, psi: float = 1.0) -> torch.Tensor:
        """Truncation trick

        Args:
            w (torch.Tensor): input style codes.
            psi (float, optional): truncation rate. Defaults to 1.0.

        Returns:
            torch.Tensor: truncated style codes.
        """

        if psi != 1.0:
            w_avg = self.w_avg[None].expand_as(w)
            w = torch.lerp(w_avg, w, psi)

        return w

    def forward_synthesis(
        self, w: torch.Tensor, angle: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward a synthesis network.

        Args:
            w (torch.Tensor): style codes `(B, N, D)`.
            angles (torch.Tensor | None, optional): angular grid. Defaults to None.

        Returns:
            dict[str, torch.Tensor]: _description_
        """
        raise NotImplementedError

    def forward_measurement(
        self, x: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward a measurement model.

        Args:
            x (dict[str, torch.Tensor]): _description_

        Returns:
            dict[str, torch.Tensor]: _description_
        """
        x = self.measurement_model(x)
        return x
