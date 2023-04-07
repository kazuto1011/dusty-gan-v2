import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from torchvision.utils import make_grid
from tqdm.auto import tqdm

import gans.models.ops as ops
from gans.coords import CoordBridge
from gans.datasets.kitti import KITTIRaw
from gans.inversion import (
    MultiScaleMaskedLoss,
    SphericalOptimizer,
    geocross_loss,
    normalize_noise_,
)
from gans.models.builder import build_generator
from gans.pretrained import autoload_ckpt
from gans.utils import (
    colorize,
    init_random_seed,
    save_video,
    set_requires_grad,
    tanh_to_sigmoid,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--sample_id", type=int, default=-1)
    parser.add_argument("--latent_type", choices=["z", "w", "w+"], default="w")
    parser.add_argument("--num_steps_1st", type=int, default=500)
    parser.add_argument("--num_steps_2nd", type=int, default=500)
    parser.add_argument("--lr_1st", type=float, default=5e-2)
    parser.add_argument("--lr_1st_rampup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_1st_rampdown_ratio", type=float, default=0.25)
    parser.add_argument("--lr_2nd", type=float, default=5e-4)
    parser.add_argument("--noise_ratio", type=float, default=0.75)
    parser.add_argument("--noise_coef", type=float, default=0.05 / 10)
    parser.add_argument("--optimize_phase", action="store_true")
    parser.add_argument("--perturb_z", action="store_true")
    parser.add_argument("--hypersphere_z", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()

    # =============================================================================
    # setup
    # =============================================================================

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

    # prepare a target LiDAR data
    dataset = KITTIRaw(
        root=cfg.dataset.root,
        split="test",
        shape=(H, W),
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
    )
    if args.sample_id == -1:
        args.sample_id = np.random.randint(len(dataset))
    print(f"sample id: {args.sample_id}")

    init_random_seed(random_seed=args.seed)

    item = dataset.__getitem__(args.sample_id)
    t_depth = item["depth"][None].to(args.device).float()
    t_mask = item["mask"][None].to(args.device).float()
    batch_size = len(t_depth)
    t_depth = coord.convert(t_depth, "depth", "depth_norm")
    t_inv_depth = coord.convert(t_depth, "depth_norm", "inv_depth_norm")
    t_inv_depth *= t_mask

    params_1st = []

    # initialize a latent code
    with torch.no_grad():
        z_dim = cfg.model.generator.mapping_kwargs.in_ch
        num_z_samples = 10_000
        torch.manual_seed(args.seed)
        z_samples = torch.randn(num_z_samples, z_dim, device=args.device)
        z_samples = G.mapping_network(z_samples)
        z_avg = z_samples.mean(dim=0, keepdim=True)
        z_std = (((z_samples - z_avg) ** 2).sum() / num_z_samples).sqrt()
        if args.hypersphere_z:
            z_avg.div_(z_avg.pow(2).mean(dim=-1, keepdim=True).add(1e-9).sqrt())

    z_avg = z_avg.repeat_interleave(batch_size, dim=0)
    if args.latent_type == "z":
        z = torch.randn(batch_size, z_dim, device=args.device)
    elif args.latent_type == "w":
        z = z_avg
    elif args.latent_type == "w+":
        z = torch.stack([z_avg] * G.synthesis_network.num_styles, dim=1)
    else:
        raise ValueError(f"{args.latent_type=}")
    z = torch.nn.Parameter(z).requires_grad_()
    params_1st.append(z)

    # initialize noise inputs
    noises = []
    for m in G.modules():
        if isinstance(m, ops.NoiseInjection):
            noise = torch.randn_like(m.fixed_noise, dtype=torch.float32)
            m.fixed_noise = noise
            if len(noises) < 9:
                noise.requires_grad = True
                noises.append(noise)
    params_1st += noises

    # initialize phase inputs
    phase = torch.zeros((batch_size, 2, 1, 1), device=args.device)
    phase = torch.nn.Parameter(phase).requires_grad_()
    if args.optimize_phase:
        params_1st += [phase]

    # build a loss function
    criterion = MultiScaleMaskedLoss(loss_fn=F.l1_loss, level=2).to(args.device)

    # stylegan2's schedule
    def lr_schedule(iteration):
        t = iteration / args.num_steps_1st
        gamma = min(1.0, (1.0 - t) / args.lr_1st_rampdown_ratio)
        gamma = 0.5 - 0.5 * np.cos(gamma * np.pi)
        gamma = gamma * min(1.0, t / args.lr_1st_rampup_ratio)
        return gamma

    # one step forward
    def forward(z, progress):
        if args.latent_type == "z":
            w = G.forward_mapping(z, None)
        elif args.latent_type == "w":
            w = torch.stack([z] * G.synthesis_network.num_styles, dim=1)
        elif args.latent_type == "w+":
            w = z
        if args.perturb_z:
            t = max(0.0, 1.0 - progress / args.noise_ratio)
            noise_strength = args.noise_coef * z_std * (t**2)
            w = w + noise_strength * torch.randn_like(w)

        imgs = G(w, angle=coord.angle + phase, input_w=True)

        g_inv_depth = tanh_to_sigmoid(imgs["image"])
        g_inv_depth_orig = tanh_to_sigmoid(imgs["image_orig"])
        g_raydrop_prob = torch.sigmoid(imgs["raydrop_logit"])
        g_depth = coord.convert(g_inv_depth_orig, "inv_depth_norm", "depth_norm")

        loss = 0
        if args.latent_type == "w+":
            loss += 5e-3 * geocross_loss(w)

        loss += criterion(g_depth, t_depth, t_mask)
        loss += criterion(g_inv_depth_orig, t_inv_depth, t_mask)

        return (
            dict(
                inv_depth=g_inv_depth,
                inv_depth_orig=g_inv_depth_orig,
                raydrop_prob=g_raydrop_prob,
            ),
            loss,
        )

    frames = []

    # =============================================================================
    # (1) gan inversion
    # =============================================================================

    torch.cuda.empty_cache()

    set_requires_grad(G, False)
    if args.hypersphere_z:
        optim_1st = SphericalOptimizer(params=params_1st, lr=args.lr_1st)
    else:
        optim_1st = torch.optim.Adam(params=params_1st, lr=args.lr_1st)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim_1st, lr_lambda=lr_schedule)

    for step in tqdm(range(args.num_steps_1st), desc="(1) gan inversion  "):
        gen_imgs, loss_1st = forward(z=z, progress=step / args.num_steps_1st)

        optim_1st.zero_grad(set_to_none=True)
        loss_1st.backward(gradient=torch.ones_like(loss_1st))
        optim_1st.step()
        scheduler.step()
        normalize_noise_(noises)

        grid = []
        grid.append(colorize(t_inv_depth))
        grid.append(colorize(gen_imgs["inv_depth_orig"]))
        grid.append(colorize(gen_imgs["raydrop_prob"]))
        grid.append(colorize(gen_imgs["inv_depth"]))
        grid = torch.cat(grid, dim=2)
        grid = make_grid(grid).permute(1, 2, 0).detach().cpu().numpy()
        frames.append(np.uint8(grid * 255))

        if not args.visualize:
            continue

        cv2.imshow("Summary", grid[..., ::-1])
        key = cv2.waitKey(10)
        if key == ord("q"):
            print("[red]cancelled![/]")
            quit()
        elif key == ord("n"):
            print("[blue]skipped![/]")
            break

    # =============================================================================
    # (2) pivotal tuning
    # =============================================================================

    torch.cuda.empty_cache()

    set_requires_grad(G, True)
    optim_2nd = torch.optim.Adam(params=G.parameters(), lr=args.lr_2nd)
    args.perturb_z = False

    for step in tqdm(range(args.num_steps_2nd), desc="(2) pivotal tuning "):
        gen_imgs, loss_2nd = forward(z=z, progress=step / args.num_steps_2nd)

        optim_2nd.zero_grad(set_to_none=True)
        loss_2nd.backward(gradient=torch.ones_like(loss_2nd))
        optim_2nd.step()
        normalize_noise_(noises)

        grid = []
        grid.append(colorize(t_inv_depth))
        grid.append(colorize(gen_imgs["inv_depth_orig"]))
        grid.append(colorize(gen_imgs["raydrop_prob"]))
        grid.append(colorize(gen_imgs["inv_depth"]))
        grid = torch.cat(grid, dim=2)
        grid = make_grid(grid).permute(1, 2, 0).detach().cpu().numpy()
        frames.append(np.uint8(grid * 255))

        if not args.visualize:
            continue

        cv2.imshow("Summary", grid[..., ::-1])
        key = cv2.waitKey(10)
        if key == ord("q"):
            print("[red]cancelled![/]")
            quit()

    save_video(frames, f"demo_inversion_{args.sample_id:010d}", fps=60)
