import functools

import numpy as np
import torch
import torch.nn.functional as F

from .models import ops


class SphericalOptimizer(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.params = params

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for param in self.params:
            param.data.div_(param.pow(2).mean(dim=-1, keepdim=True).add(1e-9).sqrt())
        return loss


def masked_loss(img_ref, img_gen, mask, loss_fn=F.l1_loss, relative=True):
    loss = loss_fn(img_ref, img_gen, reduction="none")
    if relative:
        loss = (loss * mask) / img_ref.add(1e-11)
    loss = (loss * mask).sum(dim=(1, 2, 3))
    loss = loss / mask.sum(dim=(1, 2, 3)).add(1e-8)
    return loss


class MultiScaleMaskedLoss(torch.nn.Module):
    def __init__(self, loss_fn, level=None, relative=True):
        super().__init__()
        self.pad = ops.Pad(padding=1, mode="replicate", ring=True)
        # blur
        blur_kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
        blur_kernel = torch.outer(blur_kernel, blur_kernel)
        blur_kernel /= blur_kernel.sum()
        self.register_buffer("blur_kernel", blur_kernel[None, None])
        # mask
        mask_kernel = torch.ones_like(blur_kernel)
        self.register_buffer("mask_kernel", mask_kernel[None, None])
        # loss metric
        self.dissimilarity = functools.partial(
            masked_loss, loss_fn=loss_fn, relative=relative
        )
        self.level = level

    def blurpool(self, x):
        _, C, H, W = x.shape
        h = self.pad(x)
        kernel = self.blur_kernel.repeat(C, 1, 1, 1)
        h = F.conv2d(h, kernel, stride=2, padding=0, groups=C)
        return h

    def update_mask(self, mask):
        mask = self.pad(mask)
        mask = F.conv2d(mask, self.mask_kernel, bias=None, stride=2, padding=0)
        norm = 1 / mask.masked_fill(mask == 0, 1.0)
        norm *= self.mask_kernel[0].numel()
        new_mask = torch.ones_like(mask).masked_fill(mask == 0, 0.0)
        return norm, new_mask

    def forward(self, gen, ref, mask):
        _, C, H, W = gen.shape
        level = int(np.log2(H)) if self.level is None else self.level

        loss = 0
        for i in range(max(1, level)):
            loss_i = self.dissimilarity(ref, gen, mask)
            loss += loss_i
            norm, new_mask = self.update_mask(mask)
            gen = self.blurpool(gen * mask) * norm
            ref = self.blurpool(ref * mask) * norm
            mask = new_mask

        return loss


def geocross_loss(latents):
    # PULSE
    B, N, D = latents.shape
    X = latents.view(B, 1, N, D)
    Y = latents.view(B, N, 1, D)
    A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
    B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
    D = 2 * torch.atan2(A, B)
    D = (D.pow(2) * D).mean((1, 2)) / 8.0
    return D


def normalize_noise_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std)
