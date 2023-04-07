# adopted from
# https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py
# modified rand_translation()

# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def rand_brightness(x, band=0.2, p=1.0):
    B, _, _, _ = x.shape
    device = x.device
    factor = torch.randn((B, 1, 1, 1), device=device) * band
    mask = torch.empty((B, 1, 1, 1), device=device).bernoulli_(p=p)
    brightness = mask * factor + (1 - mask) * 0.0
    y = x + brightness
    return y


# def rand_saturation(x, band=1.0, p=1.0):
#     B, _, _, _ = x.shape
#     device = x.device
#     factor = torch.empty((B, 1, 1, 1), device=device).uniform_(-1, 1) * band
#     mask = torch.empty((B, 1, 1, 1), device=device).bernoulli_(p=p)
#     x_mean = x.mean(dim=1, keepdim=True)
#     saturation = mask * factor + 1.0
#     y = torch.lerp(x_mean, x, saturation)
#     return y


def rand_contrast(x, band=0.5, p=1.0):
    B, _, _, _ = x.shape
    device = x.device
    factor = torch.exp2(torch.randn((B, 1, 1, 1), device=device) * band)
    mask = torch.empty((B, 1, 1, 1), device=device).bernoulli_(p=p)
    contrast = mask * factor + (1 - mask) * 1.0
    y = x * contrast
    return y


def random_flip(x, p=1.0):
    B, C, H, W = x.shape
    device = x.device
    x_flip = torch.flip(x, dims=[3])
    mask = torch.empty(B, device=device).bernoulli_(p=p * 0.5).bool()
    x_flip[~mask] = x[~mask]
    return x_flip


def rand_translation(x, ratio=(1.0 / 8.0, 1.0 / 8.0), p=1.0):
    B, C, H, W = x.shape
    device = x.device

    ratio_h, ratio_w = _pair(ratio)
    shift_h, shift_w = int(H * ratio_h / 2 + 0.5), int(W * ratio_w / 2 + 0.5)
    translation_h = torch.randint(-shift_h, shift_h + 1, size=[B, 1, 1], device=device)
    translation_w = torch.randint(-shift_w, shift_w + 1, size=[B, 1, 1], device=device)
    grid_batch, grid_h, grid_w = torch.meshgrid(
        torch.arange(B, dtype=torch.long, device=device),
        torch.arange(H, dtype=torch.long, device=device),
        torch.arange(W, dtype=torch.long, device=device),
    )
    x_pad = F.pad(x, [0, 0, 1, 1, 0, 0, 0, 0])  # pad top and left
    grid_h = torch.clamp(grid_h + translation_h + 1, min=0, max=H + 1)
    grid_w = grid_w + translation_w
    grid_w = grid_w % (W - 1)  # horizontal circulation
    y = (
        x_pad.permute(0, 2, 3, 1)
        .contiguous()[grid_batch, grid_h, grid_w]
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    mask = torch.empty(B, device=device).bernoulli_(p=p).bool()
    y[~mask] = x[~mask]
    return y


def rand_cutout(x, ratio=0.5, p=1.0):
    B, C, H, W = x.shape
    device = x.device
    cut_h, cut_w = int(H * ratio + 0.5), int(W * ratio + 0.5)
    offset_x = torch.randint(0, H + (1 - cut_h % 2), size=[B, 1, 1], device=device)
    offset_y = torch.randint(0, W + (1 - cut_w % 2), size=[B, 1, 1], device=device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(B, dtype=torch.long, device=device),
        torch.arange(cut_h, dtype=torch.long, device=device),
        torch.arange(cut_w, dtype=torch.long, device=device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cut_h // 2, min=0, max=H - 1)
    grid_y = torch.clamp(grid_y + offset_y - cut_w // 2, min=0, max=W - 1)
    mask = torch.ones(B, H, W, dtype=x.dtype, device=device)
    mask[grid_batch, grid_x, grid_y] = 0
    y = x * mask.unsqueeze(1)

    mask = torch.empty(B, device=device).bernoulli_(p=p).bool()
    y[~mask] = x[~mask]

    return y


AUGMENT_FNS = {
    "flip": random_flip,
    "brightness": rand_brightness,
    # "saturation": rand_saturation,
    "contrast": rand_contrast,
    "translation": rand_translation,
    "cutout": rand_cutout,
}


class DiffAugment(nn.Module):
    def __init__(self, policy=None, p_init=0.0, p_target=0.6, kimg=500):
        super().__init__()
        if policy is None:
            self.policy = [
                "flip",
                "brightness",
                # "saturation",
                "contrast",
                "translation",
                "cutout",
            ]
        else:
            self.policy = policy
        if p_target is None:
            p_init = 1.0
        self.register_buffer("p", torch.tensor(p_init).float())
        self.register_buffer("sign_cum", torch.zeros(1))
        self.register_buffer("n_pred_cum", torch.zeros(1))
        self.kimg = kimg * 1000
        self.p_target = p_target

    def forward(self, x):
        for policy in self.policy:
            x = AUGMENT_FNS[policy](x, p=self.p)
        return x

    def cumulate(self, y_real):
        self.sign_cum += y_real.detach().sign().sum()
        self.n_pred_cum += len(y_real)

    def update_p(self):
        # compute heuristics
        self.sign_cum = reduce_sum(self.sign_cum)
        self.n_pred_cum = reduce_sum(self.n_pred_cum)
        rt = self.sign_cum / self.n_pred_cum
        sign = torch.sign(rt - self.p_target)
        adjust = sign * self.n_pred_cum / self.kimg
        self.p = (self.p + adjust).clamp_(0, 1)
        # reset stats
        self.sign_cum *= 0
        self.n_pred_cum *= 0
        return rt


def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor
