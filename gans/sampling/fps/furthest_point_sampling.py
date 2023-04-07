# adopted from https://github.com/erikwijmans/Pointnet2_PyTorch

import os.path as osp
from typing import *

import torch
from torch.utils.cpp_extension import load

module_path = osp.dirname(__file__)
fps = load(
    name="fps",
    sources=[
        osp.join(module_path, "furthest_point_sampling.cpp"),
        osp.join(module_path, "furthest_point_sampling.cu"),
    ],
    extra_cuda_cflags=["--use_fast_math"],
)


class FurthestPointSampling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = fps.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sampling = FurthestPointSampling.apply


class GatherOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor
        idx : torch.Tensor
            (B, npoint) tensor of the features to gather
        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        ctx.save_for_backward(idx, features)

        return fps.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = fps.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


def downsample_point_clouds(xyz, k):
    assert xyz.ndim == 3, "expected 3-dim, but got {}-dim tensor".format(xyz.ndim)
    assert xyz.size(2) == 3, "expected (B,N,3), but got {}".format(xyz.shape)
    assert xyz.is_cuda
    xyz = xyz.contiguous()
    source = xyz.transpose(1, 2).contiguous()  # (B,3,N)
    inds = furthest_point_sampling(xyz, k)
    xyz_sub = gather_operation(source, inds)  # (B,3,k)
    xyz_sub = xyz_sub.transpose(1, 2)  # (B,k,3)
    return xyz_sub


if __name__ == "__main__":
    device = "cuda:1"
    pts = torch.randn(5, 1000, 3).to(device)
    pts = downsample_point_clouds(pts, 10)
    print(pts.shape)
