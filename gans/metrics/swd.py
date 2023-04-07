# pytorch re-implementation of
# https://github.com/tkarras/progressive_growing_of_gans
# https://github.com/koshian2/swd-pytorch

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from tqdm.auto import tqdm


def get_kernel(weight, device="cpu"):
    kernel = torch.tensor(weight, device=device).float()
    kernel = torch.outer(kernel, kernel)
    kernel /= kernel.sum()
    kernel = kernel[None, None]
    return kernel


def pyramid_down(image):
    _, C, _, _ = image.shape
    device = image.device
    gaussian = get_kernel([1, 4, 6, 4, 1], device).repeat(C, 1, 1, 1)
    padded = F.pad(image, (2, 2, 2, 2), mode="reflect")
    image = F.conv2d(padded, gaussian, stride=2, padding=0, groups=C)
    return image


def pyramid_up(image):
    _, C, _, _ = image.shape
    device = image.device
    scale = 2
    dilation = get_kernel([0, 1, 0], device).repeat(C, 1, 1, 1)
    dilated = F.conv_transpose2d(image, dilation, stride=2, padding=0, groups=C)
    padded = F.pad(dilated[..., :-1, :-1], (2, 2, 2, 2), mode="reflect")
    gaussian = get_kernel([1, 4, 6, 4, 1], device).repeat(C, 1, 1, 1) * (scale**2)
    image = F.conv2d(padded, gaussian, stride=1, padding=0, groups=C)
    return image


def laplacian_pyramid(images, num_levels):
    pyramid = [images]
    for i in range(1, num_levels):
        pyramid.append(pyramid_down(pyramid[-1]))
        pyramid[-2] -= pyramid_up(pyramid[-1])
    return pyramid


def extract_patches(minibatch, patch_size, num_patches):
    pH, pW = patch_size
    device = minibatch.device
    patches = minibatch.unfold(2, pH, 1).unfold(3, pW, 1)
    B, C, nH, nW, pH, pW = patches.shape
    N = nH * nW
    patches = patches.reshape(B, C, N, pH, pW).transpose(1, 2)
    inds = torch.randperm(N, device=device)[:num_patches]
    patches = patches.index_select(dim=1, index=inds)
    return patches


def make_descriptors(minibatch, num_levels, patch_size, num_patches):
    pyramids = laplacian_pyramid(minibatch, num_levels)
    descs = {}
    for i in range(num_levels):
        descs[i] = extract_patches(pyramids[i], patch_size, num_patches)
    return descs


def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = torch.cat(desc, dim=0)
    B, N, C, H, W = desc.shape
    C_std, C_mean = torch.std_mean(desc, dim=(0, 1, 3, 4), keepdim=True)
    desc = (desc - C_mean) / (C_std + 1e-8)
    desc = desc.reshape(-1, C * H * W)
    return desc


def sliced_wasserstein_distance(desc1, desc2, dir_repeats, dirs_per_repeat):
    D = desc1.shape[1]
    device = desc1.device
    distances = []
    for _ in tqdm(range(dir_repeats), desc="SWD: repeat", leave=False):
        dirs = torch.randn(D, dirs_per_repeat, device=device)
        dirs /= torch.std(dirs, dim=0, keepdim=True)
        proj1 = torch.matmul(desc1, dirs)
        proj2 = torch.matmul(desc2, dirs)
        proj1, _ = torch.sort(proj1, dim=0)
        proj2, _ = torch.sort(proj2, dim=0)
        d = torch.abs(proj1 - proj2)
        distances.append(torch.mean(d))
    return torch.mean(torch.stack(distances))


@torch.no_grad()
def compute_swd(
    img1,
    img2,
    num_levels=None,
    patch_size=7,
    num_patches=128,
    dir_repeats=4,
    dirs_per_repeat=128,
    batch_size=128,
):
    assert img1.ndim == img2.ndim == 4, "(B,C,H,W) shape is required"
    assert img1.shape == img2.shape
    B, C, H, W = img1.shape
    patch_size = _pair(patch_size)

    if num_levels is None:
        num_levels = int(np.log2(min(H, W) // 16) + 1)

    desc1 = defaultdict(list)
    desc2 = defaultdict(list)

    for i in tqdm(range(0, B, batch_size), desc="SWD: patch", leave=False):
        batch1 = img1[i : i + batch_size]
        batch2 = img2[i : i + batch_size]

        batch1 = make_descriptors(batch1, num_levels, patch_size, num_patches)
        batch2 = make_descriptors(batch2, num_levels, patch_size, num_patches)

        for level in batch1.keys():
            desc1[level].append(batch1[level])
            desc2[level].append(batch2[level])

    result = {}
    for level in tqdm(desc1.keys(), desc="SWD: level", leave=False):
        result["swd-" + str(16 << level)] = sliced_wasserstein_distance(
            finalize_descriptors(desc1[level]),
            finalize_descriptors(desc2[level]),
            dir_repeats,
            dirs_per_repeat,
        )

    result["swd-mean"] = sum(result.values()) / len(result)

    for key, value in result.items():
        result[key] = value.item()

    return result


if __name__ == "__main__":
    a = torch.randn(5000, 1, 64, 256).to("cuda")
    b = torch.randn(5000, 1, 64, 256).to("cuda")
    score = compute_swd(a, b)
    print(score)
