import torch
import torch.nn.functional as F


def euler_angles_to_rotation_matrix(theta):
    R_x = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(theta[0]), -torch.sin(theta[0])],
            [0, torch.sin(theta[0]), torch.cos(theta[0])],
        ],
        device=theta.device,
    )

    R_y = torch.tensor(
        [
            [torch.cos(theta[1]), 0, torch.sin(theta[1])],
            [0, 1, 0],
            [-torch.sin(theta[1]), 0, torch.cos(theta[1])],
        ],
        device=theta.device,
    )

    R_z = torch.tensor(
        [
            [torch.cos(theta[2]), -torch.sin(theta[2]), 0],
            [torch.sin(theta[2]), torch.cos(theta[2]), 0],
            [0, 0, 1],
        ],
        device=theta.device,
    )

    matrices = [R_x, R_y, R_z]
    R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
    return R


def estimate_surface_normal(points, d=2, mode="closest"):
    # estimate surface normal from coordinated point clouds
    # re-implemented the following codes with pytorch:
    # https://github.com/wkentaro/morefusion/blob/master/morefusion/geometry/estimate_pointcloud_normals.py
    # https://github.com/jmccormac/pySceneNetRGBD/blob/master/calculate_surface_normals.py

    assert points.dim() == 4, f"expected (B,3,H,W), but got {points.shape}"
    B, C, H, W = points.shape
    assert C == 3, f"expected C==3, but got {C}"
    device = points.device

    # points = F.pad(points, (0, 0, d, d), mode="constant", value=float("inf"))
    points = F.pad(points, (0, 0, d, d), mode="replicate")
    points = F.pad(points, (d, d, 0, 0), mode="circular")
    points = points.permute(0, 2, 3, 1)  # (B,H,W,3)

    # 8 adjacent offsets
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 |   | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    offsets = torch.tensor(
        [
            # (dh,dw)
            (-d, 0),  # 0
            (-d, d),  # 1
            (0, d),  # 2
            (d, d),  # 3
            (d, 0),  # 4
            (d, -d),  # 5
            (0, -d),  # 6
            (-d, -d),  # 7
        ],
        device=device,
    )

    # (B,H,W) indices
    b = torch.arange(B, device=device)[:, None, None]
    h = torch.arange(H, device=device)[None, :, None]
    w = torch.arange(W, device=device)[None, None, :]
    k = torch.arange(8, device=device)

    # anchor points
    b1 = b[:, None]  # (B,1,1,1)
    h1 = h[:, None] + d  # (1,1,H,1)
    w1 = w[:, None] + d  # (1,1,1,W)
    anchors = points[b1, h1, w1]  # (B,H,W,3) -> (B,1,H,W,3)

    # neighbor points
    offset = offsets[k]  # (8,2)
    b2 = b1
    h2 = h1 + offset[None, :, 0, None, None]  # (1,8,H,1)
    w2 = w1 + offset[None, :, 1, None, None]  # (1,8,1,W)
    points1 = points[b2, h2, w2]  # (B,8,H,W,3)

    # anothor neighbor points
    offset = offsets[(k + 2) % 8]
    b3 = b1
    h3 = h1 + offset[None, :, 0, None, None]
    w3 = w1 + offset[None, :, 1, None, None]
    points2 = points[b3, h3, w3]  # (B,8,H,W,3)

    if mode == "closest":
        # find the closest neighbor pair
        diff = torch.norm(points1 - anchors, dim=4)
        diff = diff + torch.norm(points2 - anchors, dim=4)
        i = torch.argmin(diff, dim=1)  # (B,H,W)
        # get normals by cross product
        anchors = anchors[b, 0, h, w]  # (B,H,W,3)
        points1 = points1[b, i, h, w]  # (B,H,W,3)
        points2 = points2[b, i, h, w]  # (B,H,W,3)
        vector1 = points1 - anchors
        vector2 = points2 - anchors
        normals = torch.cross(vector1, vector2, dim=-1)  # (B,H,W,3)
    elif mode == "mean":
        # get normals by cross product
        vector1 = points1 - anchors
        vector2 = points2 - anchors
        normals = torch.cross(vector1, vector2, dim=-1)  # (B,8,H,W,3)
        normals = normals.mean(dim=1)  # (B,H,W,3)
    else:
        raise NotImplementedError(mode)

    normals = normals / (torch.norm(normals, dim=3, keepdim=True) + 1e-8)
    normals = normals.permute(0, 3, 1, 2)  # (B,3,H,W)

    return normals
