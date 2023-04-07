# pytorch re-implementation of
# https://github.com/optas/latent_3d_points/blob/master/src/evaluation_metrics.py

import warnings

import torch
from tqdm.auto import tqdm


def unit_cube_grid_point_cloud(resolution, clip_sphere, device):
    spacing = 1.0 / float(resolution - 1)
    steps = torch.arange(resolution, device=device)
    grids = torch.meshgrid(steps, steps, steps, indexing="ij")  # (res, res, res, 3)
    grid = torch.stack(grids, dim=-1) * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[torch.norm(grid, dim=1) <= 0.5]

    return grid, spacing


def entropy_of_occupancy_grid(
    pcs, resolution, in_sphere=False, batch_size=128, verbose=True
):
    device = pcs.device
    epsilon = 1e-3
    bound = 0.5 + epsilon

    if abs(pcs.max()) > bound or abs(pcs.min()) > bound:
        warnings.warn("Point-clouds are not in unit cube.")

    if in_sphere and torch.norm(pcs, p=2, dim=2).max() > bound:
        warnings.warn("Point-clouds are not in unit sphere.")

    grid, _ = unit_cube_grid_point_cloud(resolution, in_sphere, device)
    grid = grid.reshape(-1, 3)
    grid_counters = torch.zeros(len(grid), device=device)
    grid_bernoulli_rvars = torch.zeros(len(grid), device=device)

    grid = grid[None, :, None]  # shape: (1, N_g, 1, 3)
    pcs = pcs[:, None]  # shape: (B, 1, N_p, 3)

    _, Ng, _, _ = grid.shape
    B, _, Np, _ = pcs.shape

    inds = []
    for i in tqdm(
        range(0, B, batch_size),
        leave=False,
        desc="JSD: voting",
        disable=not verbose,
    ):
        inds_ = []
        for j in range(0, Np, batch_size):
            distance = []
            for k in range(0, Ng, batch_size):
                mini_pcs = pcs[i : i + batch_size, :, j : j + batch_size]
                mini_grid = grid[:, k : k + batch_size]
                distance.append((mini_pcs - mini_grid).pow(2).sum(dim=-1))
            inds_.append(torch.cat(distance, dim=1).argmin(dim=1))  # (B', B')
        inds.append(torch.cat(inds_, dim=1))  # (B', N_p)

    inds = torch.cat(inds, dim=0)  # (B, N_p)
    uniq_inds = torch.cat([torch.unique(idx) for idx in inds])
    inds, uniq_inds = inds.flatten(), uniq_inds.flatten()

    vals = torch.ones_like(inds).float()
    grid_counters.scatter_add_(dim=0, index=inds, src=vals)

    vals = torch.ones_like(uniq_inds).float()
    grid_bernoulli_rvars.scatter_add_(dim=0, index=uniq_inds, src=vals)

    p = grid_bernoulli_rvars[grid_bernoulli_rvars > 0] / float(len(pcs))
    acc_entropy = _entropy(torch.cat([p, 1 - p])) / len(grid_counters)

    return acc_entropy, grid_counters


def _entropy(p, base=None, dim=-1, eps=1e-8):
    p += eps
    if base is None:
        log_p = torch.log(p)
    elif base == 2:
        log_p = torch.log2(p)
    elif base == 10:
        log_p = torch.log10(p)
    else:
        raise NotImplementedError
    return (-p * log_p).sum(dim=dim)


def _jensen_shannon_divergence(P, Q):
    assert (P >= 0).all() and (Q >= 0).all(), "Negative values."
    assert len(P) == len(Q), "Non equal size."

    P_ = P / P.sum()  # Ensure probabilities.
    Q_ = Q / Q.sum()

    e1 = _entropy(P_, base=2)
    e2 = _entropy(Q_, base=2)
    e_sum = _entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    return res


@torch.no_grad()
def compute_jsd(pcs_gen, pcs_ref, resolution=28, batch_size=128, verbose=True):
    _, gen_grid_var = entropy_of_occupancy_grid(
        pcs_gen, resolution, True, batch_size, verbose
    )
    _, ref_grid_var = entropy_of_occupancy_grid(
        pcs_ref, resolution, True, batch_size, verbose
    )
    return _jensen_shannon_divergence(gen_grid_var, ref_grid_var).item()
