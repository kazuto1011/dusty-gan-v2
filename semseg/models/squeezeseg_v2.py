from functools import partial
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.hub import download_url_to_file

from .common import (
    ConvReLUNorm,
    DeconvReLU,
    Head,
    init_weights_trunc_normal,
    init_weights_xavier,
    setup_in_ch,
)
from .crf_as_rnn import CRFRNN


class CAM(nn.Module):
    """
    Context aggregation module (CAM)
    """

    def __init__(self, ch, reduction=16):
        super().__init__()
        self.attn = nn.Sequential(
            nn.MaxPool2d(7, 1, 3),
            nn.Conv2d(ch, ch // reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn(x)


class Fire(nn.Module):
    """
    Fire module with optional deconv
    """

    def __init__(self, in_ch, s1x1, e1x1, e3x3, bn_momentum, up=False):
        super().__init__()
        self.squeeze1x1 = ConvReLUNorm(in_ch, s1x1, 1, 1, 0, bn_momentum)
        self.upsample = DeconvReLU(s1x1, s1x1, (1, 4), (1, 2), (0, 1)) if up else None
        self.expand1x1 = ConvReLUNorm(s1x1, e1x1, 1, 1, 0, bn_momentum)
        self.expand3x3 = ConvReLUNorm(s1x1, e3x3, 3, 1, 1, bn_momentum)

    def forward(self, x):
        h = self.squeeze1x1(x)
        if self.upsample is not None:
            h = self.upsample(h)
        h = self.expand1x1(h), self.expand3x3(h)
        return torch.cat(h, dim=1)


class SqueezeSegV2(nn.Module):
    def __init__(
        self,
        inputs,
        num_classes,
        bn_momentum=0.001,
        head_dropout_p=0.5,
        use_crf=False,
        crf_kernel_size=(3, 5),
        crf_init_weight_smoothness=0.02,
        crf_init_weight_appearance=0.1,
        crf_theta_gamma=0.9,
        crf_theta_alpha=0.9,
        crf_theta_beta=0.015,
        crf_num_iters=3,
        pretrained_weights=True,
    ):
        super().__init__()
        in_ch = setup_in_ch(inputs)
        self.encoder = nn.ModuleDict(
            {
                "conv_1a": nn.Sequential(
                    ConvReLUNorm(in_ch, 64, 3, (1, 2), 1, bn_momentum),
                    CAM(64),
                ),
                "conv_1b": ConvReLUNorm(in_ch, 64, 1, 1, 0, bn_momentum),
                "fire_2_3": nn.Sequential(
                    nn.MaxPool2d(3, (1, 2), 1),
                    Fire(64, 16, 64, 64, bn_momentum),
                    CAM(128),
                    Fire(128, 16, 64, 64, bn_momentum),
                    CAM(128),
                ),
                "fire_4_5": nn.Sequential(
                    nn.MaxPool2d(3, (1, 2), 1),
                    Fire(128, 32, 128, 128, bn_momentum),
                    Fire(256, 32, 128, 128, bn_momentum),
                ),
                "fire_6_9": nn.Sequential(
                    nn.MaxPool2d(3, (1, 2), 1),
                    Fire(256, 48, 192, 192, bn_momentum),
                    Fire(384, 48, 192, 192, bn_momentum),
                    Fire(384, 64, 256, 256, bn_momentum),
                    Fire(512, 64, 256, 256, bn_momentum),
                ),
            }
        )
        self.decoder = nn.ModuleDict(
            {
                "fire_10": Fire(512, 64, 128, 128, bn_momentum, up=True),
                "fire_11": Fire(256, 32, 64, 64, bn_momentum, up=True),
                "fire_12": Fire(128, 16, 32, 32, bn_momentum, up=True),
                "fire_13": Fire(64, 16, 32, 32, bn_momentum, up=True),
                "head": Head(64, num_classes, 3, head_dropout_p),
            }
        )
        self.crf = (
            CRFRNN(
                num_classes=num_classes,
                kernel_size=crf_kernel_size,
                init_weight_smoothness=crf_init_weight_smoothness,
                init_weight_appearance=crf_init_weight_appearance,
                theta_gamma=crf_theta_gamma,
                theta_alpha=crf_theta_alpha,
                theta_beta=crf_theta_beta,
                num_iters=crf_num_iters,
            )
            if use_crf
            else None
        )

        # initialization
        self.encoder.apply(partial(init_weights_trunc_normal, std=0.001))
        for name, module in self.encoder.named_modules():
            if isinstance(module, CAM):
                module.apply(init_weights_xavier)

        if pretrained_weights:
            remote_file = "https://github.com/xuanyuzhou98/SqueezeSegV2/raw/master/data/SqueezeNet/squeezenet_v1.1.pkl"
            cached_file = Path(__file__).parent / "pretrained/squeezenet_v1.1.pkl"
            if not cached_file.exists():
                download_url_to_file(remote_file, cached_file)
            pretrained_weights = joblib.load(cached_file)
            for name, module in self.encoder.named_modules():
                if isinstance(module, Fire):
                    fire = {
                        "fire_2_3.1": "fire2",
                        "fire_2_3.3": "fire3",
                        "fire_4_5.1": "fire4",
                        "fire_4_5.2": "fire5",
                        "fire_6_9.1": "fire6",
                        "fire_6_9.2": "fire7",
                        "fire_6_9.3": "fire8",
                        "fire_6_9.4": "fire9",
                    }[name]
                    for layer in ("squeeze1x1", "expand1x1", "expand3x3"):
                        weight, bias = pretrained_weights["/".join([fire, layer])]
                        weight, bias = torch.tensor(weight), torch.tensor(bias)
                        getattr(module, layer)[0].weight.data.copy_(weight)
                        getattr(module, layer)[0].bias.data.copy_(bias)
            del pretrained_weights

        self.decoder.apply(partial(init_weights_trunc_normal, std=0.1))

    def forward(self, img, xyz=None, mask=None):
        h_1b = self.encoder["conv_1b"](img)
        h_1a = self.encoder["conv_1a"](img)
        h_3 = self.encoder["fire_2_3"](h_1a)
        h_5 = self.encoder["fire_4_5"](h_3)
        h_9 = self.encoder["fire_6_9"](h_5)
        h_10 = self.decoder["fire_10"](h_9) + h_5
        h_11 = self.decoder["fire_11"](h_10) + h_3
        h_12 = self.decoder["fire_12"](h_11) + h_1a
        h_13 = self.decoder["fire_13"](h_12) + h_1b
        logit = self.decoder["head"](h_13)
        if self.crf is not None:
            assert xyz is not None
            assert mask is not None
            logit = self.crf(logit, xyz, mask)
        return logit


if __name__ == "__main__":
    model = SqueezeSegV2(
        inputs=["xyz", "depth"],
        num_classes=4,
        use_crf=True,
        pretrained_weights=True,
    )
    print(model)
    lidar = torch.randn(1, 4, 64, 2048)
    xyz = torch.randn(1, 3, 64, 2048)
    mask = torch.randn(1, 1, 64, 2048)
    logit = model(lidar, xyz, mask)
