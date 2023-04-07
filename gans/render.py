import kornia
import torch
from kornia.geometry.conversions import angle_axis_to_rotation_matrix


def make_Rt(roll=0, pitch=0, yaw=0, x=0, y=0, z=0, device="cpu"):
    # rotation of point clouds
    zero = torch.zeros(1, device=device)
    roll = torch.full_like(zero, fill_value=roll, device=device)
    pitch = torch.full_like(zero, fill_value=pitch, device=device)
    yaw = torch.full_like(zero, fill_value=yaw, device=device)

    # extrinsic parameters: yaw -> pitch order
    R = angle_axis_to_rotation_matrix(torch.stack([zero, zero, yaw], dim=-1))
    R @= angle_axis_to_rotation_matrix(torch.stack([zero, pitch, zero], dim=-1))
    R @= angle_axis_to_rotation_matrix(torch.stack([roll, zero, zero], dim=-1))
    t = torch.tensor([[x, y, z]], device=device)
    return R, t


def render_point_clouds(
    points,
    colors,
    size=512,
    R=None,
    t=None,
    focal_length=1.0,
):
    points = points.clone()
    points[..., 2] *= -1

    # extrinsic parameters
    if R is not None:
        assert R.shape[-2:] == (3, 3)
        points = points @ R
    if t is not None:
        assert t.shape[-1:] == (3,)
        points += t

    B, N, _ = points.shape
    device = points.device

    # intrinsic parameters
    K = torch.eye(3, device=device)
    K[0, 0] = focal_length  # fx
    K[1, 1] = focal_length  # fy
    K[0, 2] = 0.5  # cx, points in [-1,1]
    K[1, 2] = 0.5  # cy
    K = K[None]

    # project 3d points onto the image plane
    uv = kornia.geometry.project_points(points, K)

    uv = uv * size
    mask = (0 < uv) & (uv < size - 1)
    mask = torch.logical_and(mask[..., [0]], mask[..., [1]])

    colors = colors * mask

    # z-buffering
    uv = size - uv
    depth = torch.norm(points, p=2, dim=-1, keepdim=True)  # B,N,1
    weight = 1.0 / torch.exp(3.0 * depth)
    weight *= (depth > 1e-8).detach()
    bev = bilinear_rasterizer(uv, weight * colors, (size, size))
    bev /= bilinear_rasterizer(uv, weight, (size, size)) + 1e-8
    return bev


def bilinear_rasterizer(coords, values, out_shape):
    """
    https://github.com/VCL3D/SphericalViewSynthesis/blob/master/supervision/splatting.py
    """

    B, _, C = values.shape
    H, W = out_shape
    device = coords.device

    h = coords[..., [0]].expand(-1, -1, C)
    w = coords[..., [1]].expand(-1, -1, C)

    # Four adjacent pixels
    h_t = torch.floor(h)
    h_b = h_t + 1  # == torch.ceil(h)
    w_l = torch.floor(w)
    w_r = w_l + 1  # == torch.ceil(w)

    h_t_safe = torch.clamp(h_t, 0.0, H - 1)
    h_b_safe = torch.clamp(h_b, 0.0, H - 1)
    w_l_safe = torch.clamp(w_l, 0.0, W - 1)
    w_r_safe = torch.clamp(w_r, 0.0, W - 1)

    weight_h_t = (h_b - h) * (h_t == h_t_safe).detach().float()
    weight_h_b = (h - h_t) * (h_b == h_b_safe).detach().float()
    weight_w_l = (w_r - w) * (w_l == w_l_safe).detach().float()
    weight_w_r = (w - w_l) * (w_r == w_r_safe).detach().float()

    # Bilinear weights
    weight_tl = weight_h_t * weight_w_l
    weight_tr = weight_h_t * weight_w_r
    weight_bl = weight_h_b * weight_w_l
    weight_br = weight_h_b * weight_w_r

    # For stability
    weight_tl *= (weight_tl >= 1e-3).detach().float()
    weight_tr *= (weight_tr >= 1e-3).detach().float()
    weight_bl *= (weight_bl >= 1e-3).detach().float()
    weight_br *= (weight_br >= 1e-3).detach().float()

    values_tl = values * weight_tl  # (B,N,C)
    values_tr = values * weight_tr
    values_bl = values * weight_bl
    values_br = values * weight_br

    indices_tl = (w_l_safe + W * h_t_safe).long()
    indices_tr = (w_r_safe + W * h_t_safe).long()
    indices_bl = (w_l_safe + W * h_b_safe).long()
    indices_br = (w_r_safe + W * h_b_safe).long()

    render = torch.zeros(B, H * W, C, device=device)
    render.scatter_add_(dim=1, index=indices_tl, src=values_tl)
    render.scatter_add_(dim=1, index=indices_tr, src=values_tr)
    render.scatter_add_(dim=1, index=indices_bl, src=values_bl)
    render.scatter_add_(dim=1, index=indices_br, src=values_br)
    render = render.reshape(B, H, W, C).permute(0, 3, 1, 2)

    return render
