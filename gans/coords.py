import enum
import os

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .geometry import estimate_surface_normal
from .render import render_point_clouds
from .utils import points_to_normal_2d


class _CoordType:
    DEPTH = "depth"
    DEPTH_NORM = "depth_norm"
    INV_DEPTH = "inv_depth"
    INV_DEPTH_NORM = "inv_depth_norm"
    POINT_MAP = "point_map"
    POINT_SET = "point_set"
    NORMAL_MAP = "normal_map"

    def __init__(self):
        self.mode = (
            self.DEPTH,
            self.DEPTH_NORM,
            self.INV_DEPTH,
            self.INV_DEPTH_NORM,
            self.POINT_MAP,
            self.POINT_SET,
            self.NORMAL_MAP,
        )

    def __contains__(self, key):
        return key in self.mode


CoordType = _CoordType()


class CoordBridge(nn.Module):
    def __init__(
        self,
        num_ring,
        num_points,
        min_depth,
        max_depth,
        angle_file,
        raydrop_const=0,
    ) -> None:
        super().__init__()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        assert self.max_depth > self.min_depth
        self.H, self.W = num_ring, num_points
        self.raydrop_const = raydrop_const

        angle = np.load(angle_file)  # [H, W, XY]
        angle = torch.from_numpy(angle).permute(2, 0, 1)[None]
        periodic = torch.cat([angle.sin(), angle.cos()], dim=1)
        periodic = torch.cat([periodic, periodic, periodic], dim=3)
        periodic = F.interpolate(
            periodic,
            size=(self.H, self.W * 3),
            mode="bilinear",
            align_corners=False,
        )
        periodic = periodic[..., self.W : 2 * self.W]
        angle = torch.atan2(periodic[:, :2], periodic[:, 2:])
        self.register_buffer("angle", angle)

    def get_mask(self, x, coord):
        if coord == CoordType.DEPTH:
            mask = x >= self.min_depth
            mask &= x <= self.max_depth
            mask &= x > 0.0
        elif coord == CoordType.INV_DEPTH:
            mask = x >= (1 / self.max_depth)
            mask &= x <= (1 / self.min_depth)
            mask &= x > 0.0
        elif coord in (CoordType.DEPTH_NORM, CoordType.INV_DEPTH_NORM):
            mask = (x > 0.0) & (x <= 1.0)
        else:
            raise NotImplementedError(f"{coord}")
        return mask

    def convert(self, x, src, tgt, tol=1e-11):
        assert src in CoordType, src
        assert tgt in CoordType, tgt

        if src == tgt:
            return x
        if src == CoordType.DEPTH:
            valid = self.get_mask(x, src).float()
            if tgt in (CoordType.INV_DEPTH, CoordType.INV_DEPTH_NORM):
                inv_depth = 1 / x.add(tol) * valid
                if tgt == CoordType.INV_DEPTH_NORM:
                    return self.convert(inv_depth, CoordType.INV_DEPTH, tgt)
                return inv_depth
            elif tgt == CoordType.DEPTH_NORM:
                depth_norm = x / self.max_depth
                return depth_norm
            elif tgt in (
                CoordType.POINT_MAP,
                CoordType.POINT_SET,
                CoordType.NORMAL_MAP,
            ):
                point_map = self.depth_to_point_map(x)
                if tgt == CoordType.POINT_SET:
                    return self.convert(point_map, CoordType.POINT_MAP, tgt)
                elif tgt == CoordType.NORMAL_MAP:
                    return self.convert(point_map, CoordType.POINT_MAP, tgt)
                return point_map
        elif src == CoordType.DEPTH_NORM:
            depth = x * self.max_depth
            if tgt == CoordType.DEPTH:
                return depth
            elif tgt in (CoordType.INV_DEPTH, CoordType.INV_DEPTH_NORM):
                return self.convert(depth, CoordType.DEPTH, tgt)
            elif tgt in (CoordType.POINT_MAP, CoordType.POINT_SET):
                return self.convert(depth, CoordType.DEPTH, tgt)
        elif src == CoordType.INV_DEPTH:
            if tgt == CoordType.INV_DEPTH_NORM:
                inv_depth_norm = x * self.min_depth
                return inv_depth_norm
            if tgt in (CoordType.DEPTH, CoordType.DEPTH_NORM):
                valid = self.get_mask(x, src).float()
                depth = 1 / x.add(tol) * valid
                if tgt == CoordType.DEPTH_NORM:
                    return self.convert(depth, CoordType.DEPTH, tgt)
                return depth
        elif src == CoordType.INV_DEPTH_NORM:
            if tgt == CoordType.INV_DEPTH:
                valid = (x > tol).float()
                inv_depth = x / self.min_depth
                return inv_depth
            if tgt in (CoordType.DEPTH, CoordType.DEPTH_NORM):
                valid = (x > tol).float()
                inv_depth = x / self.min_depth
                return self.convert(inv_depth, CoordType.INV_DEPTH, tgt)
            if tgt in (CoordType.POINT_MAP, CoordType.POINT_SET, CoordType.NORMAL_MAP):
                valid = (x > tol).float()
                inv_depth = x / self.min_depth
                valid *= self.get_mask(inv_depth, CoordType.INV_DEPTH).float()
                depth = 1 / inv_depth.add_(tol) * valid
                point_map = self.convert(depth, CoordType.DEPTH, CoordType.POINT_MAP)
                if tgt == CoordType.POINT_SET:
                    return self.convert(point_map, CoordType.POINT_MAP, tgt)
                elif tgt == CoordType.NORMAL_MAP:
                    return self.convert(point_map, CoordType.POINT_MAP, tgt)
                return point_map
        elif src == CoordType.POINT_MAP:
            if tgt == CoordType.POINT_SET:
                point_set = x.flatten(2).permute(0, 2, 1).contiguous()
                return point_set
            elif tgt in (
                CoordType.DEPTH,
                CoordType.DEPTH_NORM,
                CoordType.INV_DEPTH,
                CoordType.INV_DEPTH_NORM,
            ):
                depth = torch.norm(x, p=2, dim=1, keepdim=True)
                if tgt in (
                    CoordType.DEPTH_NORM,
                    CoordType.INV_DEPTH,
                    CoordType.INV_DEPTH_NORM,
                ):
                    x = self.convert(depth, CoordType.DEPTH, tgt)
                return x
            elif tgt == CoordType.NORMAL_MAP:
                point_map = x / self.max_depth
                normals = -estimate_surface_normal(point_map, d=2)
                normals[normals != normals] = 0.0
                return normals
        raise NotImplementedError(f"{src} to {tgt}")

    def depth_to_point_map(self, depth):
        assert depth.dim() == 4
        grid_cos = torch.cos(self.angle)
        grid_sin = torch.sin(self.angle)
        grid_x = depth * grid_cos[:, [0]] * grid_cos[:, [1]]
        grid_y = depth * grid_cos[:, [0]] * grid_sin[:, [1]]
        grid_z = depth * grid_sin[:, [0]]
        return torch.cat((grid_x, grid_y, grid_z), dim=1)

    def make_birds_eye_view(self, inv_depth, Rt):
        R, t = Rt
        _, _, _, W = inv_depth.shape
        points = self.convert(inv_depth, "inv_depth_norm", "point_map")
        points = points / self.max_depth
        normal = points_to_normal_2d(points, mode="closest")
        points = einops.rearrange(points, "b c h w -> b (h w) c")
        colors = einops.rearrange(normal, "b c h w -> b (h w) c")
        bev = render_point_clouds(points, colors, size=W, R=R, t=t)
        return bev

    def extra_repr(self):
        return f'H={self.H}, W={self.W}, min_depth={self.min_depth}, max_depth="{self.max_depth}"'
