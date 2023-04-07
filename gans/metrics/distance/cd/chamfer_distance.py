import os.path as osp

import torch
from torch.utils.cpp_extension import load

module_path = osp.dirname(__file__)
cd = load(
    name="cd",
    sources=[
        osp.join(module_path, "chamfer_distance.cpp"),
        osp.join(module_path, "chamfer_distance.cu"),
    ],
    extra_cuda_cflags=["--use_fast_math"],
)


class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()

        dist1 = torch.zeros(batchsize, n, device=device)
        dist2 = torch.zeros(batchsize, m, device=device)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int, device=device)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int, device=device)

        if xyz1.is_cuda:
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        device = graddist1.device

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device=device)
        gradxyz2 = torch.zeros(xyz2.size(), device=device)

        if graddist1.is_cuda:
            cd.backward_cuda(
                xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
            )
        else:
            cd.backward(
                xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
            )

        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)


chamfer_distance = ChamferDistanceFunction.apply

if __name__ == "__main__":
    device = "cuda:1"
    pts = torch.randn(5, 1000, 3).to(device)
    dl, dr, _, _ = chamfer_distance(pts, pts)
    print(dl.shape, dr.shape)
