import argparse

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from gans.models.builder import build_generator
from gans.pretrained import PRETRAINED_CKPTS, autoload_ckpt
from gans.utils import init_random_seed

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=list(PRETRAINED_CKPTS.keys()), required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--truncation_psi", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    init_random_seed(args.seed)

    # load checkpoint
    ckpt = autoload_ckpt(args.arch)
    z_dim = ckpt["cfg"].model.generator.mapping_kwargs.in_ch

    # setup model
    G = build_generator(ckpt["cfg"].model.generator)
    G.load_state_dict(ckpt["G_ema"])
    G.eval().to(args.device)

    # generate
    z = torch.randn(args.batch_size, z_dim).to(args.device)
    angle = ckpt["angle"].repeat_interleave(args.batch_size, dim=0).to(args.device)
    imgs = G(z=z, angle=angle, truncation_psi=args.truncation_psi)

    # visualize
    grid = make_grid(imgs["image"], nrow=2, pad_value=float("nan"))[0].cpu()
    plt.imshow(grid, cmap="turbo", vmin=-1, vmax=1, interpolation="none")
    plt.axis("off")
    plt.show()
