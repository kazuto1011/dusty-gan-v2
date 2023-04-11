import numpy as np
import torch
from torch import nn


def init_weights_trunc_normal(module, std=0.001):
    if isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_xavier(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_bilinear(module):
    assert isinstance(module, nn.ConvTranspose2d)
    assert tuple(module.kernel_size) == (1, 4)
    assert module.in_channels == module.out_channels
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    kernel = torch.tensor([1, 3, 3, 1], dtype=torch.float32)
    kernel = kernel / kernel.sum() * 2
    for c in range(module.in_channels):
        module.weight.data[c, c] = kernel[None, :]  # (1,4)


def setup_in_ch(inputs):
    channels = {"xyz": 3, "depth": 1, "reflectance": 1, "mask": 1}
    in_ch = 0
    for modality in inputs:
        in_ch += channels[modality]
    return in_ch


class ConvNorm(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bn_momentum):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum),
        )


class ConvReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True),
            nn.ReLU(inplace=True),
        )


class ConvNormReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bn_momentum):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )


class ConvReLUNorm(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bn_momentum):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum),
        )


class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bn_momentum):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum),
            nn.LeakyReLU(0.1, inplace=True),
        )


class DeconvReLU(nn.Sequential):
    """
    Transposed conv initialized with bilinear weights -> ReLU
    """

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, init_method="bilinear"
    ):
        super().__init__(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        )
        self.init_method = init_method
        if self.init_method == "bilinear":
            init_weights_bilinear(self[0])

    def apply(self, fn):
        if self.init_method == "bilinear":
            return self  # do nothing
        else:
            return super().apply(fn)


class Head(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, dropout_p):
        super().__init__(
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(in_ch, out_ch, kernel_size, 1, kernel_size // 2),
        )
