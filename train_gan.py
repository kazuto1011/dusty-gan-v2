import argparse
import datetime
import tempfile
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from gans.render import render_point_clouds
from gans.trainer import Trainer
from gans.utils import (
    colorize,
    init_dist_process,
    points_to_normal_2d,
    power_spectrum_2d,
    tanh_to_sigmoid,
)

console = Console()


def log_images(
    writer,
    tag,
    step,
    converter=None,
    image=None,
    image_orig=None,
    image_aug=None,
    raydrop_logit=None,
    raydrop_mask=None,
):
    if image_orig is not None:
        image_orig = tanh_to_sigmoid(image_orig).clamp(0, 1)
        writer.add_images(tag + "/image/orig", colorize(image_orig), step)
    if image_aug is not None:
        image_aug = tanh_to_sigmoid(image_aug).clamp(0, 1)
        writer.add_images(tag + "/image/aug", colorize(image_aug), step)
    if raydrop_logit is not None:
        raydrop_prob = torch.sigmoid(raydrop_logit)
        writer.add_images(tag + "/raydrop_prob", colorize(raydrop_prob), step)
    if raydrop_mask is not None:
        writer.add_images(tag + "/raydrop_mask", raydrop_mask, step)
    if image is not None:
        assert converter is not None
        inv_depth = tanh_to_sigmoid(image).clamp(0, 1)
        points_map = converter.convert(inv_depth, "inv_depth_norm", "point_map")
        points_map /= converter.max_depth
        normal_map = points_to_normal_2d(points_map, mode="closest")
        points_bev = render_point_clouds(
            points=einops.rearrange(points_map, "b c h w -> b (h w) c"),
            colors=einops.rearrange(normal_map, "b c h w -> b (h w) c"),
            t=torch.tensor([0, 0, 0.7]).to(inv_depth),
        )
        specrum = power_spectrum_2d(inv_depth)
        specrum -= specrum.min()
        specrum /= specrum.max()
        writer.add_images(tag + "/image", colorize(inv_depth), step)
        writer.add_images(tag + "/image/spectrum", colorize(specrum), step)
        writer.add_images(tag + "/normal", normal_map, step)
        writer.add_images(tag + "/pointcloud", points_bev, step)


def training_loop(rank, cfg, temp_dir, log_dir):
    cfg.training.rank = rank
    gpu_info = torch.cuda.get_device_properties(rank)
    console.log(
        f"rank {rank}: {gpu_info.name} {gpu_info.total_memory / 1024**3:g} GB, "
        + f"{cfg.training.num_workers} workers"
    )

    init_dist_process(rank, temp_dir, cfg.training.num_gpus, cfg.random_seed)

    trainer = Trainer(cfg)

    total_imgs = int(cfg.training.total_kimg * 1e3)
    total_iters = int(total_imgs / (cfg.training.batch_size))

    if rank == 0:
        console.log("batch size / gpu:", cfg.training.batch_size_per_gpu)
        console.log("number of gpu:", cfg.training.num_gpus)
        console.log("batch size:", cfg.training.batch_size)
        console.log(f"total imgs: {total_imgs:,}")
        console.log(f"iteration start: {trainer.start_iteration+1:,}")
        console.log(f"iteration end: {total_iters:,}")

        # tensorboard
        writer = SummaryWriter(log_dir=log_dir / "tensorboard")

        # real images
        reals = trainer.fetch_reals(next(trainer.iter_train_loader))
        log_images(
            writer,
            tag="real",
            step=1,
            converter=trainer.coord,
            image=reals["image"],
            raydrop_mask=reals["raydrop_mask"],
        )

    # moving average meters
    moving_avg = defaultdict(partial(deque, maxlen=100))

    # training loop (iteration)
    for i in tqdm(
        range(trainer.start_iteration + 1, total_iters + 1),
        desc="training",
        dynamic_ncols=True,
        disable=not rank == 0,
    ):
        scalars = trainer.step(i)
        num_imgs = trainer.iters_to_imgs(i)

        # log images
        if rank == 0 and i % cfg.training.checkpoint.save_image == 0:
            reals_aug = trainer.A(trainer.warmup(reals["image"]))
            log_images(
                writer,
                tag="real",
                step=num_imgs,
                converter=trainer.coord,
                image_aug=reals_aug,
            )
            fakes = trainer.sample(ema=True)
            log_images(
                writer,
                tag="fake",
                step=num_imgs,
                converter=trainer.coord,
                image=fakes.get("image", None),
                image_orig=fakes.get("image_orig", None),
                raydrop_logit=fakes.get("raydrop_logit", None),
                raydrop_mask=fakes.get("raydrop_mask", None),
            )

        # validation
        if rank == 0 and i % cfg.training.checkpoint.validation == 0:
            scores = trainer.validation()
            for key, scalar in scores.items():
                writer.add_scalar("score/" + key, scalar, num_imgs)

        # save models
        if rank == 0 and i % cfg.training.checkpoint.save_model == 0:
            save_path = log_dir / f"models/checkpoint_{num_imgs:010d}.pth"
            trainer.save_checkpoint(save_path, num_imgs)

        for key, value in scalars.items():
            moving_avg[key].append(value)

        # log training stats
        if rank == 0 and i % cfg.training.checkpoint.save_stats == 0:
            for key, value in moving_avg.items():
                writer.add_scalar(key, np.mean(value), num_imgs)

    # save the final model
    if rank == 0:
        num_imgs = trainer.iters_to_imgs(total_iters)
        save_path = log_dir / f"models/checkpoint_{num_imgs:010d}.pth"
        trainer.save_checkpoint(save_path, num_imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--resume", type=str)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # machine dependent settings
    assert (
        cfg.training.batch_size % args.num_gpus == 0
    ), "batch_size must be divisible by num_gpus"
    cfg.training.num_gpus = args.num_gpus
    cfg.training.batch_size_per_gpu = cfg.training.batch_size // args.num_gpus
    cfg.training.num_workers = int(
        (torch.multiprocessing.cpu_count() + args.num_gpus - 1) / args.num_gpus
    )

    if args.dry_run:
        console.log(OmegaConf.to_container(cfg))
        quit()

    # set up logging
    cfg.training.resume = args.resume
    if args.resume is None:
        log_dir = Path("logs/gans")
        log_dir /= f"{cfg.dataset.name:s}"
        log_dir /= f"{cfg.model.generator.arch:s}+{cfg.model.discriminator.arch:s}"
        log_dir /= datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, log_dir / "training_config.yaml")
    else:
        log_dir = Path(args.resume).parents[1]

    # launch training processes
    with tempfile.TemporaryDirectory() as temp_dir:
        torch.multiprocessing.spawn(
            training_loop,
            args=(cfg, Path(temp_dir), log_dir),
            nprocs=args.num_gpus,
        )
