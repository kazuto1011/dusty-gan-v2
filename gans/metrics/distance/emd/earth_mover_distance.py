import os.path as osp

import torch
from torch.utils.cpp_extension import load

module_path = osp.dirname(__file__)
emd = load(
    name="emd",
    sources=[
        osp.join(module_path, "earth_mover_distance.cpp"),
        osp.join(module_path, "earth_mover_distance.cu"),
    ],
    extra_cuda_cflags=["--use_fast_math"],
)


# Inherit from Function
class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        ctx.save_for_backward(xyz1, xyz2)
        match, temp = emd.approxmatch_forward(xyz1, xyz2)
        ctx.match = match
        cost = emd.matchcost_forward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = emd.matchcost_backward(xyz1, xyz2, ctx.match)
        grad_cost_expand = grad_cost.unsqueeze(1).unsqueeze(2)
        return grad_xyz1 * grad_cost_expand, grad_xyz2 * grad_cost_expand


earth_mover_distance = EarthMoverDistanceFunction.apply


class EarthMoverDistance(torch.nn.Module):
    def forward(self, input1, input2):
        return EarthMoverDistanceFunction.apply(input1, input2)


if __name__ == "__main__":
    device = "cuda:0"
    pts_1 = torch.randn(5, 1000, 3).to(device)
    pts_2 = torch.randn(5, 1000, 3).to(device)
    cost = earth_mover_distance(pts_1, pts_2)
    cost /= 1000
    print(cost.shape)
    print(cost)
