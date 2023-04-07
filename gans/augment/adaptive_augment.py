import math

import numpy as np
import scipy.signal
import torch
import torch.distributed as dist
from torch import autograd
from torch.cuda.amp import custom_fwd
from torch.nn import functional as F

from gans.models.ops.upfirdn2d.upfirdn2d import upfirdn2d

SYM2 = (
    -0.12940952255092145,
    0.22414386804185735,
    0.836516303737469,
    0.48296291314469025,
)

SYM6 = (
    0.015404109327027373,
    0.0034907120842174702,
    -0.11799011114819057,
    -0.048311742585633,
    0.4910559419267466,
    0.787641141030194,
    0.3379294217276218,
    -0.07263752278646252,
    -0.021060292512300564,
    0.04472490177066578,
    0.0017677118642428036,
    -0.007800708325034148,
)


def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


class GridSampleForward(autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        out = F.grid_sample(
            input, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        ctx.save_for_backward(input, grid)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = GridSampleBackward.apply(grad_output, input, grid)

        return grad_input, grad_grid


class GridSampleBackward(autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid):
        op, _ = torch._C._jit_get_operation("aten::grid_sampler_2d_backward")

        grad_input, grad_grid = op(
            grad_output,
            input,
            grid,
            0,
            0,
            False,
            (ctx.needs_input_grad[1], ctx.needs_input_grad[2]),
        )
        ctx.save_for_backward(grid)

        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad_grad_input, grad_grad_grid):
        (grid,) = ctx.saved_tensors
        grad_grad_output = None

        if ctx.needs_input_grad[0]:
            grad_grad_output = GridSampleForward.apply(grad_grad_input, grid)

        return grad_grad_output, None, None


grid_sample = GridSampleForward.apply


def scale2d_single(s_x, s_y, device="cpu"):
    return torch.tensor(
        (
            (s_x, 0, 0),
            (0, s_y, 0),
            (0, 0, 1),
        ),
        device=device,
        dtype=torch.float32,
    )


def translate2d_single(t_x, t_y, device="cpu"):
    return torch.tensor(
        (
            (1, 0, t_x),
            (0, 1, t_y),
            (0, 0, 1),
        ),
        device=device,
        dtype=torch.float32,
    )


def translate2d(t_x, t_y, device="cpu"):
    batchsize = t_x.shape[0]

    mat = torch.eye(3, device=device)[None].repeat(batchsize, 1, 1)
    translate = torch.stack((t_x, t_y), 1)
    mat[:, :2, 2] = translate

    return mat


def rotate2d(theta, device="cpu"):
    batchsize = theta.shape[0]

    mat = torch.eye(3, device=device)[None].repeat(batchsize, 1, 1)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    rot = torch.stack((cos_t, -sin_t, sin_t, cos_t), 1).view(batchsize, 2, 2)
    mat[:, :2, :2] = rot

    return mat


def scale2d(s_x, s_y, device="cpu"):
    batchsize = s_x.shape[0]

    mat = torch.eye(3, device=device)[None].repeat(batchsize, 1, 1)
    mat[:, 0, 0] = s_x
    mat[:, 1, 1] = s_y

    return mat


def translate3d(t_x, t_y, t_z, device="cpu"):
    batchsize = t_x.shape[0]

    mat = torch.eye(4, device=device)[None].repeat(batchsize, 1, 1)
    translate = torch.stack((t_x, t_y, t_z), 1)
    mat[:, :3, 3] = translate

    return mat


def rotate3d(axis, theta, device="cpu"):
    batchsize = theta.shape[0]

    u_x, u_y, u_z = axis

    eye = torch.eye(3, device=device)[None]
    cross = torch.tensor(
        [
            (0, -u_z, u_y),
            (u_z, 0, -u_x),
            (-u_y, u_x, 0),
        ],
        device=device,
    )[None]
    outer = torch.tensor(axis, device=device)
    outer = (outer.unsqueeze(1) * outer)[None]

    sin_t = torch.sin(theta).view(-1, 1, 1)
    cos_t = torch.cos(theta).view(-1, 1, 1)

    rot = cos_t * eye + sin_t * cross + (1 - cos_t) * outer

    eye_4 = torch.eye(4, device=device)[None].repeat(batchsize, 1, 1)
    eye_4[:, :3, :3] = rot

    return eye_4


def scale3d(s_x, s_y, s_z, device="cpu"):
    batchsize = s_x.shape[0]

    mat = torch.eye(4, device=device)[None].repeat(batchsize, 1, 1)
    mat[:, 0, 0] = s_x
    mat[:, 1, 1] = s_y
    mat[:, 2, 2] = s_z

    return mat


def luma_flip(axis, i, device="cpu"):
    batchsize = i.shape[0]

    eye = torch.eye(4, device=device)[None].repeat(batchsize, 1, 1)
    axis = torch.tensor(axis + (0,), device=device)
    flip = 2 * torch.ger(axis, axis) * i.view(-1, 1, 1)

    return eye - flip


def saturation(axis, i, device="cpu"):
    batchsize = i.shape[0]

    eye = torch.eye(4, device=device)[None].repeat(batchsize, 1, 1)
    axis = torch.tensor(axis + (0,), device=device)
    axis = torch.ger(axis, axis)
    saturate = axis + (eye - axis) * i.view(-1, 1, 1)

    return saturate


def lognormal_sample(size, mean=0, std=1, device="cpu"):
    return torch.empty(size, device=device).log_normal_(mean=mean, std=std)


def category_sample(size, categories, device="cpu"):
    category = torch.tensor(categories, device=device)
    sample = torch.randint(high=len(categories), size=(size,), device=device)

    return category[sample]


def uniform_sample(size, low, high, device="cpu"):
    return torch.empty(size, device=device).uniform_(low, high)


def normal_sample(size, mean=0, std=1, device="cpu"):
    return torch.empty(size, device=device).normal_(mean, std)


def bernoulli_sample(size, p, device="cpu"):
    return torch.empty(size, device=device).bernoulli_(p)


def random_mat_apply(p, transform, prev, eye, device="cpu"):
    size = transform.shape[0]
    select = bernoulli_sample(size, p, device=device).view(size, 1, 1)
    select_transform = select * transform + (1 - select) * eye

    return select_transform @ prev


def make_grid(shape, x0, x1, y0, y1, device):
    n, c, h, w = shape
    grid = torch.empty(n, h, w, 3, device=device)
    grid[:, :, :, 0] = torch.linspace(x0, x1, w, device=device)
    grid[:, :, :, 1] = torch.linspace(y0, y1, h, device=device).unsqueeze(-1)
    grid[:, :, :, 2] = 1

    return grid


def affine_grid(grid, mat):
    n, h, w, _ = grid.shape
    return (grid.view(n, h * w, 3) @ mat.transpose(1, 2)).view(n, h, w, 2)


def get_padding(G, height, width, kernel_size):
    device = G.device

    cx = (width - 1) / 2
    cy = (height - 1) / 2
    cp = torch.tensor(
        [(-cx, -cy, 1), (cx, -cy, 1), (cx, cy, 1), (-cx, cy, 1)], device=device
    )
    cp = G @ cp.T

    pad_k = kernel_size // 4

    pad = cp[:, :2, :].permute(1, 0, 2).flatten(1)
    pad = torch.cat((-pad, pad)).max(1).values
    pad = pad + torch.tensor([pad_k * 2 - cx, pad_k * 2 - cy] * 2, device=device)
    pad = pad.max(torch.tensor([0, 0] * 2, device=device))
    pad = pad.min(torch.tensor([width - 1, height - 1] * 2, device=device))

    pad_x1, pad_y1, pad_x2, pad_y2 = pad.ceil().to(torch.int32)

    return pad_x1, pad_x2, pad_y1, pad_y2


class AdaptiveAugment(torch.nn.Module):
    def __init__(
        self,
        # strength parameters
        p_init=0.0,
        p_target=0.6,
        p_max=0.9,
        kimg=500,
        # policy parameters
        lr_flip=0.0,
        ud_flip=0.0,
        int_trans=0.0,
        iso_scale=0.0,
        frac_trans=0.0,
        brightness=0.0,
        contrast=0.0,
        luma_flip=0.0,
        hue=0.0,
        saturation=0.0,
        imgfilter=0.0,
        noise=0.0,
        cutout=0.0,
        **ada_kwargs,
    ):
        super().__init__()
        # ada aparameters
        self.register_buffer("p", torch.tensor(p_init).float())
        self.register_buffer("sign_cum", torch.zeros(1))
        self.register_buffer("n_pred_cum", torch.zeros(1))
        self.kimg = kimg * 1000
        self.p_target = p_target  # p is fixed if None
        self.p_max = p_max

        # probability multipliers
        self.mul_lr_flip = float(lr_flip)
        self.mul_ud_flip = float(ud_flip)
        self.mul_int_trans = float(int_trans)
        self.mul_iso_scale = float(iso_scale)
        self.mul_frac_trans = float(frac_trans)
        self.mul_brightness = float(brightness)
        self.mul_contrast = float(contrast)
        self.mul_luma_flip = float(luma_flip)
        self.mul_hue = float(hue)
        self.mul_saturation = float(saturation)
        self.mul_imgfilter = float(imgfilter)
        self.mul_noise = float(noise)
        self.mul_cutout = float(cutout)

        if ada_kwargs.get("wonly_trans", False):
            self.h_trans_factor = 0.0
        else:
            self.h_trans_factor = 1.0

        # Construct filter bank for image-space filtering.
        self.imgfilter_bands = [1, 1, 1, 1]
        self.imgfilter_std = 1
        Hz_lo = np.asarray(SYM2)  # H(z)
        Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size))  # H(-z)
        Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2  # H(z) * H(z^-1) / 2
        Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2  # H(-z) * H(-z^-1) / 2
        Hz_fbank = np.eye(4, 1)  # Bandpass(H(z), b_i)
        for i in range(1, Hz_fbank.shape[0]):
            Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(
                Hz_fbank.shape[0], -1
            )[:, :-1]
            Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
            Hz_fbank[
                i,
                (Hz_fbank.shape[1] - Hz_hi2.size)
                // 2 : (Hz_fbank.shape[1] + Hz_hi2.size)
                // 2,
            ] += Hz_hi2
        self.register_buffer("Hz_fbank", torch.as_tensor(Hz_fbank, dtype=torch.float32))

    def cumulate(self, y_real):
        self.sign_cum += y_real.detach().sign().sum()
        self.n_pred_cum += len(y_real)

    def update_p(self):
        # compute heuristics
        self.sign_cum = reduce_sum(self.sign_cum)
        self.n_pred_cum = reduce_sum(self.n_pred_cum)
        rt = self.sign_cum / self.n_pred_cum
        if self.p_target is not None:
            sign = torch.sign(rt - self.p_target)
            adjust = sign * self.n_pred_cum / self.kimg
            self.p = (self.p + adjust).clamp_(0, self.p_max)
        # reset stats
        self.sign_cum *= 0
        self.n_pred_cum *= 0
        return rt

    def sample_affine(self, size, height, width, device="cpu"):
        I_3 = torch.eye(3, device=device)[None].repeat(size, 1, 1)
        G = I_3

        # lr flip
        if self.mul_lr_flip > 0:
            param = category_sample(size, (0, 1), device=device)
            Gc = scale2d(1 - 2.0 * param, torch.ones(size), device=device)
            G = random_mat_apply(self.p * self.mul_lr_flip, Gc, G, I_3, device=device)

        # ud flip
        if self.mul_ud_flip > 0:
            param = category_sample(size, (0, 1), device=device)
            Gc = scale2d(torch.ones(size), 1 - 2.0 * param, device=device)
            G = random_mat_apply(self.p * self.mul_ud_flip, Gc, G, I_3, device=device)

        # integer translate
        if self.mul_int_trans > 0:
            param = uniform_sample((2, size), -0.125, 0.125, device=device)
            param_height = torch.round(param[0] * height) * self.h_trans_factor
            param_width = torch.round(param[1] * width)
            Gc = translate2d(param_width, param_height, device=device)
            G = random_mat_apply(self.p * self.mul_int_trans, Gc, G, I_3, device=device)

        # isotropic scale (horizontal only)
        if self.mul_iso_scale > 0:
            param = lognormal_sample(size, std=0.2 * math.log(2), device=device)
            Gc = scale2d(torch.ones_like(param), param, device=device)
            G = random_mat_apply(self.p * self.mul_iso_scale, Gc, G, I_3, device=device)

        # fractional translate
        if self.mul_frac_trans > 0:
            param = normal_sample((2, size), std=0.125, device=device)
            param_height = param[0] * height * self.h_trans_factor
            param_width = param[1] * width
            Gc = translate2d(param_width, param_height, device=device)
            G = random_mat_apply(
                self.p * self.mul_frac_trans, Gc, G, I_3, device=device
            )

        return G

    def sample_color(self, size, device="cpu"):
        I_4 = torch.eye(4, device=device)[None].repeat(size, 1, 1)
        C = I_4

        axis_val = 1 / math.sqrt(3)
        axis = (axis_val, axis_val, axis_val)

        # brightness
        if self.mul_brightness > 0:
            param = normal_sample(size, std=0.2, device=device)
            Cc = translate3d(param, param, param, device=device)
            C = random_mat_apply(
                self.p * self.mul_brightness, Cc, C, I_4, device=device
            )

        # contrast
        if self.mul_contrast > 0:
            param = lognormal_sample(size, std=0.5 * math.log(2), device=device)
            Cc = scale3d(param, param, param, device=device)
            C = random_mat_apply(self.p * self.mul_contrast, Cc, C, I_4, device=device)

        # luma flip
        if self.mul_luma_flip > 0:
            param = category_sample(size, (0, 1), device=device)
            Cc = luma_flip(axis, param, device=device)
            C = random_mat_apply(self.p * self.mul_luma_flip, Cc, C, I_4, device=device)

        # hue rotation
        if self.mul_hue > 0:
            param = uniform_sample(size, -math.pi, math.pi, device=device)
            Cc = rotate3d(axis, param, device=device)
            C = random_mat_apply(self.p * self.mul_hue, Cc, C, I_4, device=device)

        # saturation
        if self.mul_saturation > 0:
            param = lognormal_sample(size, std=1 * math.log(2), device=device)
            Cc = saturation(axis, param, device=device)
            C = random_mat_apply(
                self.p * self.mul_saturation, Cc, C, I_4, device=device
            )

        return C

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, img):
        device = img.device
        k = SYM6
        k = torch.as_tensor(k, device=device)
        k_flip = torch.flip(k, (0,))

        # sample affine tranformation matrix
        batchsize, channels, height, width = img.shape
        G_inv = torch.inverse(
            self.sample_affine(batchsize, height, width, device=device)
        )

        # pad image and adjust origin
        pad_x1, pad_x2, pad_y1, pad_y2 = get_padding(G_inv, height, width, len(k))
        img = F.pad(img, (pad_x1, pad_x2, 0, 0), mode="circular")
        img = F.pad(img, (0, 0, pad_y1, pad_y2), mode="reflect")
        G_inv = (
            translate2d_single(
                (pad_x1 - pad_x2) / 2, (pad_y1 - pad_y2) / 2, device=device
            )
            @ G_inv
        )

        # upsample
        up_pad = (
            (len(k) + 2 - 1) // 2,
            (len(k) - 2) // 2,
            (len(k) + 2 - 1) // 2,
            (len(k) - 2) // 2,
        )
        img = upfirdn2d(img, k[None], up=(2, 1), pad=(*up_pad[:2], 0, 0))
        img = upfirdn2d(img, k[:, None], up=(1, 2), pad=(0, 0, *up_pad[2:]))
        G_inv = (
            scale2d_single(2, 2, device=device)
            @ G_inv
            @ scale2d_single(1 / 2, 1 / 2, device=device)
        )
        G_inv = (
            translate2d_single(-0.5, -0.5, device=device)
            @ G_inv
            @ translate2d_single(0.5, 0.5, device=device)
        )

        # geometric transform
        pad_k = len(k) // 4
        shape = (batchsize, channels, (height + pad_k * 2) * 2, (width + pad_k * 2) * 2)
        G_inv = (
            scale2d_single(2 / img.shape[3], 2 / img.shape[2], device=device)
            @ G_inv
            @ scale2d_single(1 / (2 / shape[3]), 1 / (2 / shape[2]), device=device)
        )
        grid = F.affine_grid(G_inv[:, :2, :], shape, align_corners=False)
        img = grid_sample(img, grid)

        # downsample
        d_p = -pad_k * 2
        down_pad = (
            d_p + (len(k) - 2 + 1) // 2,
            d_p + (len(k) - 2) // 2,
            d_p + (len(k) - 2 + 1) // 2,
            d_p + (len(k) - 2) // 2,
        )
        img = upfirdn2d(img, k_flip[None], down=(2, 1), pad=(*down_pad[:2], 0, 0))
        img = upfirdn2d(img, k_flip[:, None], down=(1, 2), pad=(0, 0, *down_pad[2:]))

        # color transformation
        C = self.sample_color(batchsize, device=device)
        img = img.reshape([batchsize, channels, height * width])
        if channels == 3:
            img = C[:, :3, :3] @ img + C[:, :3, 3:]
        elif channels == 1:
            C = C[:, :3, :].mean(dim=1, keepdims=True)
            img = img * C[:, :, :3].sum(dim=2, keepdims=True) + C[:, :, 3:]
        img = img.reshape([batchsize, channels, height, width])

        if self.mul_imgfilter > 0:
            num_bands = self.Hz_fbank.shape[0]
            assert len(self.imgfilter_bands) == num_bands
            expected_power = torch.tensor(
                np.array([10, 1, 1, 1]) / 13, device=device, dtype=torch.float32
            )
            # Expected power spectrum (1/f).

            # Apply amplification for each band with probability (mul_imgfilter * strength * band_strength).
            g = torch.ones([batchsize, num_bands], device=device)
            for i, band_strength in enumerate(self.imgfilter_bands):
                t_i = torch.exp2(
                    torch.randn([batchsize], device=device) * self.imgfilter_std
                )
                t_i = torch.where(
                    torch.rand([batchsize], device=device)
                    < self.mul_imgfilter * self.p * band_strength,
                    t_i,
                    torch.ones_like(t_i),
                )
                # Temporary gain vector.
                t = torch.ones([batchsize, num_bands], device=device)
                t[:, i] = t_i  # Replace i'th element.
                # Normalize power.
                t = t / (expected_power * t.square()).sum(dim=-1, keepdims=True).sqrt()
                g = g * t  # Accumulate into global gain.

            # Construct combined amplification filter.
            Hz_prime = g @ self.Hz_fbank  # [batch, tap]
            Hz_prime = Hz_prime.unsqueeze(1).repeat(
                [1, channels, 1]
            )  # [batch, channels, tap]
            Hz_prime = Hz_prime.reshape(
                [batchsize * channels, 1, -1]
            )  # [batch * channels, 1, tap]

            # Apply filter.
            p = self.Hz_fbank.shape[1] // 2
            img = img.reshape([1, batchsize * channels, height, width])
            # img = torch.nn.functional.pad(input=img, pad=[p, p, p, p], mode="reflect")
            img = F.pad(img, (p, p, 0, 0), mode="circular")
            img = F.pad(img, (0, 0, p, p), mode="reflect")
            img = F.conv2d(
                input=img, weight=Hz_prime.unsqueeze(2), groups=batchsize * channels
            )
            img = F.conv2d(
                input=img, weight=Hz_prime.unsqueeze(3), groups=batchsize * channels
            )
            img = img.reshape([batchsize, channels, height, width])

        if self.mul_noise > 0:
            sigma = torch.randn([batchsize, 1, 1, 1], device=device).abs() * 0.1
            sigma = torch.where(
                torch.rand([batchsize, 1, 1, 1], device=device)
                < self.mul_noise * self.p,
                sigma,
                torch.zeros_like(sigma),
            )
            img = img + torch.randn_like(img) * sigma

        if self.mul_cutout > 0:
            size = torch.full([batchsize, 2, 1, 1, 1], 0.5, device=device)
            size = torch.where(
                torch.rand([batchsize, 1, 1, 1, 1], device=device)
                < self.mul_cutout * self.p,
                size,
                torch.zeros_like(size),
            )
            center = torch.rand([batchsize, 2, 1, 1, 1], device=device)
            coord_x = torch.arange(width, device=device).reshape([1, 1, 1, -1])
            coord_y = torch.arange(height, device=device).reshape([1, 1, -1, 1])
            mask_x = ((coord_x + 0.5) / width - center[:, 0]).abs() >= size[:, 0] / 2
            mask_y = ((coord_y + 0.5) / height - center[:, 1]).abs() >= size[:, 1] / 2
            mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
            img = img * mask

        return img
