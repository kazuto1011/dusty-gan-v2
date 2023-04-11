import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gaussian_kernel(kernel_size, sigma, device="cpu"):
    H, W = nn.modules.utils._pair(kernel_size)
    assert H % 2 == 1 and W % 2 == 1, "must be odd"
    hs = torch.arange(H, device=device) - H // 2
    ws = torch.arange(W, device=device) - W // 2
    coord = torch.meshgrid(hs, ws, indexing="ij")
    pdist = torch.stack(coord, dim=-1).pow(2).sum(dim=-1)
    kernel = torch.exp(-pdist / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


class kNN2d(nn.Module):
    """
    - Simplified version of k-NN filtering introduced in RangeNet++ [Milioto et al. IROS 2019]
    - Reference: https://github.com/PRBonn/lidar-bonnetal/blob/master/train/tasks/semantic/postproc/KNN.py
    """

    def __init__(self, num_classes, k=3, kernel_size=3, sigma=1.0, cutoff=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.sigma = sigma
        self.cutoff = cutoff

        # inverse gaussian kernel
        gaussian_kernel = get_gaussian_kernel(self.kernel_size, self.sigma)
        self.register_buffer("dist_kernel", (1 - gaussian_kernel)[None, None])

    def forward(self, depth, label):
        B, C, H, W = depth.shape
        device = depth.device

        # point-wise distance
        depth_anchor = einops.rearrange(depth, "B C H W -> B C 1 (H W)")
        depth_neighbor = F.unfold(depth, self.kernel_size, padding=self.padding)
        depth_neighbor = einops.rearrange(depth_neighbor, "B (C K) HW -> B C K HW", C=C)
        depth_neighbor[depth_neighbor < 0] = float("inf")
        jump = torch.abs(depth_neighbor - depth_anchor)  # -> B C K HW

        # penalize far pixels
        jump = einops.rearrange(jump, "B C K (H W) -> B (C K) H W", H=H, W=W)
        kernel = self.dist_kernel.repeat_interleave(jump.shape[1], dim=0)
        dist = F.conv2d(jump, kernel, padding=self.padding, groups=kernel.shape[0])
        dist = einops.rearrange(dist, "B (C K) H W -> B C K (H W)", C=C)

        # find nearest points
        _, ids_topk = dist.topk(k=self.k, dim=2, largest=False, sorted=False)

        # gather labels
        label = label[:, None].float()  # add channel dim
        label_neighbor = F.unfold(label, self.kernel_size, padding=self.padding)
        label_neighbor = einops.rearrange(label_neighbor, "B (1 K) HW -> B 1 K HW")
        label_topk = label_neighbor.gather(dim=2, index=ids_topk)

        # cutoff
        if self.cutoff > 0:
            dist_topk = dist.gather(dim=2, index=ids_topk)
            label_topk[dist_topk > self.cutoff] = self.num_classes

        # majority voting
        ones = torch.ones_like(label_topk).to(depth)
        label_bins = torch.zeros(B, 1, self.num_classes + 1, H * W, device=device)
        label_bins.scatter_add_(dim=2, index=label_topk.long(), src=ones)
        refined_label = label_bins[:, :, :-1].argmax(dim=2)
        refined_label = einops.rearrange(refined_label, "B 1 (H W) -> B H W", H=H, W=W)

        return refined_label


if __name__ == "__main__":
    B, C, H, W = 16, 20, 64, 2048
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth = torch.rand(B, 1, H, W).to(device)
    label = torch.randint(0, C, (B, H, W)).long().to(device)

    knn = kNN2d(C).to(device)
    label_knn = knn(depth=depth, label=label)
    print(f"result {label_knn.shape=}")
    print((label != label_knn).sum())
