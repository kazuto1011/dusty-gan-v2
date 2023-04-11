import torch
import torch.nn as nn

from .common import ConvReLU, DeconvReLU, Head, init_weights_trunc_normal, setup_in_ch
from .crf_as_rnn import CRFRNN


class Fire(nn.Module):
    """
    Fire module with optional deconv
    """

    def __init__(self, in_ch, s1x1, e1x1, e3x3, up=False):
        super().__init__()
        self.squeeze1x1 = ConvReLU(in_ch, s1x1, 1, 1, 0)
        self.upsample = DeconvReLU(s1x1, s1x1, (1, 4), (1, 2), (0, 1)) if up else None
        self.expand1x1 = ConvReLU(s1x1, e1x1, 1, 1, 0)
        self.expand3x3 = ConvReLU(s1x1, e3x3, 3, 1, 1)

    def forward(self, x):
        h = self.squeeze1x1(x)
        h = self.upsample(h) if self.upsample is not None else h
        h = self.expand1x1(h), self.expand3x3(h)
        return torch.cat(h, dim=1)


class SqueezeSegV1(nn.Module):
    def __init__(
        self,
        inputs,
        num_classes,
        head_dropout_p=0.5,
        use_crf=False,
        crf_kernel_size=(3, 5),
        crf_init_weight_smoothness=0.02,
        crf_init_weight_appearance=0.1,
        crf_theta_gamma=0.9,
        crf_theta_alpha=0.9,
        crf_theta_beta=0.015,
        crf_num_iters=3,
    ):
        super().__init__()
        in_ch = setup_in_ch(inputs)

        self.conv1a = ConvReLU(in_ch, 64, 3, (1, 2), 1)
        self.conv1b = ConvReLU(in_ch, 64, 1, 1, 0)
        self.fire2_3 = nn.Sequential(
            nn.MaxPool2d(3, (1, 2), 1),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
        )
        self.fire4_5 = nn.Sequential(
            nn.MaxPool2d(3, (1, 2), 1),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
        )
        self.fire6_9 = nn.Sequential(
            nn.MaxPool2d(3, (1, 2), 1),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        self.fire10 = Fire(512, 64, 128, 128, up=True)
        self.fire11 = Fire(256, 32, 64, 64, up=True)
        self.fire12 = Fire(128, 16, 32, 32, up=True)
        self.fire13 = Fire(64, 16, 32, 32, up=True)
        self.head = Head(64, num_classes, 3, head_dropout_p)
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
        self.apply(init_weights_trunc_normal)

    def forward(self, img, xyz=None, mask=None):
        # encoder
        h_1b = self.conv1b(img)
        h_1a = self.conv1a(img)
        h_3 = self.fire2_3(h_1a)
        h_5 = self.fire4_5(h_3)
        h_9 = self.fire6_9(h_5)
        # decoder
        h_10 = self.fire10(h_9) + h_5
        h_11 = self.fire11(h_10) + h_3
        h_12 = self.fire12(h_11) + h_1a
        h_13 = self.fire13(h_12) + h_1b
        logit = self.head(h_13)
        # crf
        if self.crf is not None:
            assert xyz is not None
            assert mask is not None
            logit = self.crf(logit, xyz, mask)
        return logit


if __name__ == "__main__":
    model = SqueezeSegV1(inputs=["xyz", "depth"], num_classes=4, use_crf=True)
    lidar = torch.randn(1, 4, 64, 2048)
    xyz = torch.randn(1, 3, 64, 2048)
    mask = torch.randn(1, 1, 64, 2048)
    logit = model(lidar, xyz, mask)
    print(model)
    print(logit.shape)
