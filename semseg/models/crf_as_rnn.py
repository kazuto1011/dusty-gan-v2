import einops
import torch
import torch.nn.functional as F
from rich import print
from torch import nn


class CRFRNN(nn.Module):
    """
    - CRF-RNN [Zheng et al. ICCV'15] used in SqueezeSeg [Wu et al. ICRA'18]
    """

    def __init__(
        self,
        num_classes,
        kernel_size=(3, 5),
        init_weight_smoothness=0.02,
        init_weight_appearance=0.1,
        theta_gamma=0.9,
        theta_alpha=0.9,
        theta_beta=0.015,
        num_iters=3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_iters = num_iters
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        _fn = nn.modules.utils._ntuple(self.num_classes)
        self.register_buffer("theta_gamma", torch.tensor(_fn(theta_gamma)))
        self.register_buffer("theta_alpha", torch.tensor(_fn(theta_alpha)))
        self.register_buffer("theta_beta", torch.tensor(_fn(theta_beta)))

        # fixed smoothness kernel
        kernel_gamma = self.get_smoothness_kernel(self.kernel_size, self.theta_gamma)
        self.register_buffer("kernel_gamma", kernel_gamma)
        kernel_alpha = self.get_smoothness_kernel(self.kernel_size, self.theta_alpha)
        self.register_buffer("kernel_alpha", kernel_alpha)

        # trainable label-wise weights for balancing kernels
        init_kernel_weights = lambda scale: torch.ones(1, num_classes, 1, 1) * scale
        weight_appearance = init_kernel_weights(init_weight_appearance)
        self.register_parameter("weight_appearance", nn.Parameter(weight_appearance))
        weight_smoothness = init_kernel_weights(init_weight_smoothness)
        self.register_parameter("weight_smoothness", nn.Parameter(weight_smoothness))

        # trainable label compatibility
        init_potts_model = 1 - torch.eye(num_classes)[..., None, None]  # [i!=j]
        self.label_compatibility = nn.Conv2d(num_classes, num_classes, 1, 1, bias=False)
        self.label_compatibility.weight.data = init_potts_model

    def get_smoothness_kernel(self, kernel_size, theta, device="cpu"):
        H, W = nn.modules.utils._pair(kernel_size)
        assert H % 2 == 1 and W % 2 == 1, "must be odd"
        hs = torch.arange(H, device=device) - H // 2
        ws = torch.arange(W, device=device) - W // 2
        coord = torch.meshgrid(hs, ws, indexing="ij")
        pdist = torch.stack(coord, dim=-1).pow(2).sum(dim=-1)
        kernel = torch.zeros(self.num_classes, self.num_classes, H, W)
        for c in range(self.num_classes):
            _kernel = torch.exp(-pdist / (2 * theta[c] ** 2))
            _kernel[H // 2, W // 2] = 0  # do not penalize the center
            kernel[c, c] = _kernel
        return kernel

    def apply(self, fn):
        return self  # do nothing

    def unfold_neighbors(self, x):
        B, C, H, W = x.shape
        # unfolding pixels within a kernel
        x = F.unfold(x, self.kernel_size, padding=self.padding)
        x = einops.rearrange(x, "B (C K) HW -> B C K HW", C=C)
        # exluding the kernel center
        kernel_numel = x.shape[2]  # == np.prod(self.kernel_size)
        kernel_index = torch.arange(kernel_numel, device=x.device)
        kernel_index = kernel_index[kernel_index != kernel_numel // 2]
        x = x.index_select(dim=2, index=kernel_index)
        return x  # -> B C K-1 (H W)

    def precompute_kernel_beta(self, xyz):
        xyz_anchor = einops.rearrange(xyz, "B C H W -> B C 1 (H W)")
        xyz_neighbors = self.unfold_neighbors(xyz)  # -> B C K-1 (H W)
        pdist = (xyz_neighbors - xyz_anchor).pow(2).sum(dim=1, keepdim=True)
        theta = self.theta_beta[None, :, None, None]
        kernel = torch.exp(-pdist / (2 * theta**2))
        return kernel

    def message_passing_smoothness(self, Q, kernel):
        # gaussian filtering by group convolution
        assert kernel.shape[0] == self.num_classes
        return F.conv2d(Q, kernel, padding=self.padding)

    def message_passing_appearance(self, Q, kernel_beta, mask):
        masked_Q = Q * mask
        exp_appearance = torch.ones_like(masked_Q).flatten(2)  # B C (H W)
        for i in range(exp_appearance.shape[0]):
            # sample-by-sample basis due to high memory requirements
            Q_neighbors_i = self.unfold_neighbors(masked_Q[[i]])  # B C K-1 (H W)
            exp_appearance[[i]] = (Q_neighbors_i * kernel_beta[[i]]).sum(dim=2)
        exp_appearance = exp_appearance.reshape_as(masked_Q) * mask  # B C H W
        exp_smoothness = self.message_passing_smoothness(Q, self.kernel_alpha)
        return exp_appearance * exp_smoothness  # bilateral kernel

    def weighting_kernels(self, k_smoothness, k_appearance):
        k_smoothness = self.weight_smoothness * k_smoothness
        k_appearance = self.weight_appearance * k_appearance
        return k_smoothness + k_appearance

    def forward(self, unary, xyz, mask):
        """
        unary: (B,N,H,W)
        xyz  : (B,3,H,W)
        mask : (B,H,W)
        """
        # initialization
        Q = unary
        kernel_beta = self.precompute_kernel_beta(xyz).detach()
        mask = mask[:, None] if mask.ndim == 3 else mask
        # mean-field approximation
        for _ in range(self.num_iters):
            # normalize
            Q = F.softmax(Q, dim=1)
            # message passing (#filters=2)
            k_smoothness = self.message_passing_smoothness(Q, self.kernel_gamma)
            k_appearance = self.message_passing_appearance(Q, kernel_beta, mask)
            weighted_k = self.weighting_kernels(k_smoothness, k_appearance)
            # compatibility transform
            pairwise = self.label_compatibility(weighted_k)
            # iterative update
            Q = unary - pairwise
        return Q


if __name__ == "__main__":
    batch_size = 8
    num_classes = 4
    h, w = 64, 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    crf = CRFRNN(num_classes=num_classes, num_iters=3).to(device)
    logit = torch.rand(batch_size, num_classes, h, w).to(device)
    xyz = torch.randn(batch_size, 3, h, w).to(device)
    mask = torch.rand(batch_size, h, w).to(device)

    out = crf(logit, xyz, mask)
    print(f"{out.shape=}")
