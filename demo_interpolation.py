import argparse

import cv2
import einops
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import scipy
import torch
from kornia.filters import median_blur
from torch.distributions.utils import clamp_probs

from gans.coords import CoordBridge
from gans.models.builder import build_generator
from gans.models.ops import GumbelSigmoid
from gans.pretrained import autoload_ckpt
from gans.utils import colorize, cycle, init_random_seed, tanh_to_sigmoid


def visualize_2d(G, coord, args, steps, interp_fn):
    def generate():
        imgs = G(
            z=torch.from_numpy(interp_fn(next(steps))).float().to(args.device),
            angle=coord.angle,
            truncation_psi=args.truncation_psi,
            input_w=True,
        )
        grid = [tanh_to_sigmoid(imgs["image"])]
        if "image_orig" in imgs:
            grid = [imgs["raydrop_logit"].sigmoid()] + grid
            grid = [tanh_to_sigmoid(imgs["image_orig"])] + grid
        grid = torch.cat(grid, dim=2)
        grid = colorize(grid)
        return grid[0].cpu().numpy().transpose(1, 2, 0)

    print('press "q" to quit')
    while True:
        cv2.imshow("image", generate()[..., ::-1])
        if cv2.waitKey(10) == ord("q"):
            break


def visualize_3d(G, coord, args, steps, interp_fn):
    # Polyscope setting
    ps.set_program_name("Interpolating point clouds")
    ps.set_SSAA_factor(3)
    ps.set_build_gui(False)
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_height_factor(0.1)
    ps.set_shadow_darkness(0.1)
    ps.look_at((-1, -1, 1), (0, 0, 0))
    pts_kwargs = dict(radius=0.0005, color=(0, 0, 0))
    in_updating = True
    z = None
    psi = args.truncation_psi

    def render():
        nonlocal z, psi, in_updating

        # GUIs
        psim.PushItemWidth(150)
        if in_updating:
            if psim.Button("Stop"):
                in_updating = False
        else:
            if psim.Button("Resume"):
                in_updating = True
        _, psi = psim.SliderFloat("Truncation trick", psi, v_min=-1, v_max=1)
        psim.PopItemWidth()

        # Generation
        if in_updating:
            z = torch.from_numpy(interp_fn(next(steps))).float().to(args.device)
        imgs = G(z=z, angle=coord.angle, truncation_psi=psi, input_w=True)

        # Convert depth to point cloud
        inv_depth = tanh_to_sigmoid(imgs["image"])
        points = coord.convert(inv_depth, "inv_depth_norm", "point_map")
        points = median_blur(points, (3, 3))
        normal = coord.convert(points, "point_map", "normal_map")
        normal = tanh_to_sigmoid(normal)
        points = points / coord.max_depth
        points = einops.rearrange(points, "b c h w -> b (h w) c")
        colors = einops.rearrange(normal, "b c h w -> b (h w) c")
        points = points[0].cpu().numpy()
        colors = colors[0].cpu().numpy()

        if not ps.has_point_cloud("lidar"):
            ps.register_point_cloud("lidar", points, **pts_kwargs)
        else:
            ps.get_point_cloud("lidar").update_point_positions(points)
        ps.get_point_cloud("lidar").add_color_quantity("n", colors, enabled=True)

    ps.set_user_callback(render)
    ps.show()


if __name__ == "__main__":
    # setting
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--mode", choices=["2d", "3d"], default="2d")
    parser.add_argument("--num_anchors", type=int, default=10)
    parser.add_argument("--truncation_psi", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    init_random_seed(args.seed)

    # config
    ckpt = autoload_ckpt(args.ckpt_path)
    cfg = ckpt["cfg"]

    # coord converter
    H, W = cfg.model.generator.synthesis_kwargs.resolution
    coord = CoordBridge(
        num_ring=H,
        num_points=W,
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        angle_file=f"data/coords/{cfg.dataset.name}.npy",
    )
    coord.to(args.device)

    # generator
    G = build_generator(cfg.model.generator)
    G.load_state_dict(ckpt["G_ema"])
    G.eval().to(args.device)

    # make deterministic
    uniforms = clamp_probs(torch.rand(1, H, W, device=args.device))
    noise = uniforms.log() - (-uniforms).log1p()
    for n, m in G.named_modules():
        if isinstance(m, GumbelSigmoid):
            m.register_forward_hook(lambda _m, i, _o: ((i[0] + noise) > 0.0).float())
        if hasattr(m, "use_fp16"):
            m.use_fp16 = False

    # setup latent codes
    zs = []
    z_dim = cfg.model.generator.mapping_kwargs.in_ch
    for _ in range(args.num_anchors):
        noise = torch.randn(z_dim, device=args.device)
        noise /= noise.pow(2).mean(dim=0, keepdim=True).add(1e-8).sqrt()
        zs.append(noise)
    zs = G.forward_mapping(torch.stack(zs))

    # build an interpolation path between the anchors
    num_frames = int(90 * args.num_anchors)
    interp_fn = scipy.interpolate.interp1d(
        x=np.arange(-args.num_anchors * 2, args.num_anchors * 3),
        y=np.tile(zs.cpu().numpy(), [5] + [1] * (zs.ndim - 1)),
        kind="cubic",
        axis=0,
    )
    steps = np.linspace(0, args.num_anchors, num_frames, endpoint=False)
    steps = cycle(list(steps[:, None]))

    if args.mode == "2d":
        visualize_2d(G, coord, args, steps, interp_fn)
    elif args.mode == "3d":
        visualize_3d(G, coord, args, steps, interp_fn)
    else:
        pass
