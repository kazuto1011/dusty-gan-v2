import argparse
from collections import defaultdict

import numpy as np
import torch
from rich.console import Console
from torch.distributions.utils import clamp_probs
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import gans.utils as utils
from gans.coords import CoordBridge
from gans.datasets.kitti import KITTIRaw
from gans.metrics.cov_mmd_1nna import compute_cov_mmd_1nna
from gans.metrics.fpd_kpd import compute_frechet_distance, compute_squared_mmd
from gans.metrics.jsd import compute_jsd
from gans.metrics.pointnet import pretrained_pointnet
from gans.metrics.swd import compute_swd
from gans.models.builder import build_generator
from gans.models.ops.gumbel import GumbelSigmoid
from gans.sampling.fps import downsample_point_clouds

console = Console()


@torch.no_grad()
def preprocess(
    rank,
    ckpt_path,
    batch_size_per_gpu,
    random_seed,
    latent_codes,
    num_cpus,
    num_gpus,
    queue,
):
    utils.init_random_seed(random_seed)

    device = torch.device(rank)
    num_workers = int((num_cpus + num_gpus - 1) / num_gpus)
    gpu_info = torch.cuda.get_device_properties(rank)
    console.log(
        f"rank {rank}: {gpu_info.name} {gpu_info.total_memory / 1024**3:g} GB, {num_workers} workers"
    )

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    angle = ckpt["angle"].to(device)

    # datasets
    H, W = cfg.model.generator.synthesis_kwargs.resolution
    dataset_kwargs = dict(
        root=cfg.dataset.root,
        shape=(H, W),
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
    )
    loader_kwargs = dict(
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    train_dataset = KITTIRaw(split="train", **dataset_kwargs)
    train_sampler = np.array_split(np.arange(len(train_dataset)), num_gpus)[rank]
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    test_dataset = KITTIRaw(split="test", **dataset_kwargs)
    test_sampler = np.array_split(np.arange(len(test_dataset)), num_gpus)[rank]
    test_loader = DataLoader(test_dataset, sampler=test_sampler, **loader_kwargs)
    latent_dataset = TensorDataset(latent_codes)
    latent_sampler = np.array_split(np.arange(len(latent_dataset)), num_gpus)[rank]
    latent_loader = DataLoader(latent_dataset, sampler=latent_sampler, **loader_kwargs)
    console.log(
        f"rank {rank}: {len(train_sampler):,}/{len(train_dataset):,} (train), "
        + f"{len(test_sampler):,}/{len(test_dataset):,} (test), "
        + f"{len(latent_sampler):,}/{len(latent_dataset):,} (generation)"
    )

    # coordinate converter (i.e. depth to point cloud)
    coord = CoordBridge(
        num_ring=H,
        num_points=W,
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        angle_file=f"data/coords/{cfg.dataset.name}.npy",
    )
    coord.to(device)

    # generator
    G = build_generator(cfg.model.generator)
    G.load_state_dict(ckpt["G_ema"])
    G.eval().to(device)

    # deterministic gumbel sampling
    uniform = clamp_probs(torch.rand(1, H, W, device=device))
    noise = uniform.log() - (-uniform).log1p()
    for m in G.modules():
        if isinstance(m, GumbelSigmoid):
            m.register_forward_hook(lambda _, i, o: ((i[0] + noise) > 0.0).float())

    # feature extractor for point clouds
    pointnet = pretrained_pointnet()
    pointnet.eval().to(device)

    def transform_reals(imgs, mask):
        imgs, mask = imgs.to(device), mask.to(device)
        imgs = coord.convert(imgs, "depth", "inv_depth_norm")
        imgs = utils.sigmoid_to_tanh(imgs)
        imgs = (
            mask * imgs
            + (1 - mask) * cfg.model.generator.measurement_kwargs.raydrop_const
        )
        imgs = utils.tanh_to_sigmoid(imgs).clamp(0, 1)
        points = coord.convert(imgs, "inv_depth_norm", "point_set")
        points /= coord.max_depth
        feats = pointnet(points.transpose(1, 2))
        points = downsample_point_clouds(points, cfg.validation.num_points)
        return imgs.cpu(), points.cpu(), feats.cpu()

    def transform_fakes(imgs):
        imgs = utils.tanh_to_sigmoid(imgs).clamp(0, 1)
        points = coord.convert(imgs, "inv_depth_norm", "point_set")
        points /= coord.max_depth
        feats = pointnet(points.transpose(1, 2))
        points = downsample_point_clouds(points, cfg.validation.num_points)
        return imgs.cpu(), points.cpu(), feats.cpu()

    summary = defaultdict(list)
    desc = lambda msg: f"rank {rank}: {msg}"
    tqdm_kwargs = dict(dynamic_ncols=True, position=rank, leave=False, unit="imgs")

    # train set
    with tqdm(total=len(train_sampler), desc=desc("train set"), **tqdm_kwargs) as pbar:
        for item in train_loader:
            imgs, points, feats = transform_reals(item["depth"], item["mask"])
            summary["train-imgs"].append(imgs)
            summary["train-points"].append(points)
            summary["train-feats"].append(feats)
            pbar.update(len(imgs))

    # test set
    with tqdm(total=len(test_sampler), desc=desc("test set"), **tqdm_kwargs) as pbar:
        for item in test_loader:
            imgs, points, feats = transform_reals(item["depth"], item["mask"])
            summary["test-imgs"].append(imgs)
            summary["test-points"].append(points)
            summary["test-feats"].append(feats)
            pbar.update(len(imgs))

    # generation
    with tqdm(total=len(latent_sampler), desc=desc("gen set"), **tqdm_kwargs) as pbar:
        for (z,) in latent_loader:
            imgs = G(z=z.to(device), angle=angle.repeat_interleave(len(z), dim=0))
            imgs, points, feats = transform_fakes(imgs["image"])
            summary["gen-imgs"].append(imgs)
            summary["gen-points"].append(points)
            summary["gen-feats"].append(feats)
            pbar.update(len(imgs))

    for set_name in summary.keys():
        summary[set_name] = torch.cat(summary[set_name], dim=0)

    queue.put((rank, summary))


def subsample(batch, n):
    if len(batch) <= n:
        return batch
    else:
        return batch[torch.linspace(0, len(batch), n + 1)[:-1].long()]


@torch.no_grad()
def evaluate(args):
    console.log(args)
    num_cpus = torch.multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    manager = torch.multiprocessing.Manager()
    queue = manager.Queue()
    pretrained_pointnet()  # dry run to download weights

    utils.init_random_seed(args.random_seed)
    latent_codes = torch.randn(args.num_samples, 512)

    # prepare train, test, and generated data
    torch.multiprocessing.spawn(
        preprocess,
        args=(
            args.ckpt_path,
            args.batch_size_per_gpu,
            args.random_seed,
            latent_codes,
            num_cpus,
            num_gpus,
            queue,
        ),
        nprocs=num_gpus,
    )

    # collect and sort queued data
    summary = defaultdict(list)
    while not queue.empty():
        rank, summary_dict = queue.get()
        for set_name, value in summary_dict.items():
            summary[set_name].append((rank, value))
    for set_name, tuple_list in summary.items():
        value_list = [value for _, value in sorted(tuple_list, key=lambda x: x[0])]
        summary[set_name] = torch.cat(value_list, dim=0)

    # evaluate
    device = torch.device("cuda")
    scores = dict()
    # as inverse depth images
    if "swd" in args.metrics:
        scores.update(
            compute_swd(
                img1=subsample(summary["gen-imgs"], 2048).to(device),
                img2=subsample(summary["test-imgs"], 2048).to(device),
            )
        )
    # as point clouds
    if "jsd" in args.metrics:
        scores["jsd"] = compute_jsd(
            pcs_gen=subsample(summary["gen-points"], 2048).to(device) / 2,
            pcs_ref=subsample(summary["test-points"], 2048).to(device) / 2,
        )
    # as point clouds (> 1h)
    if "1nna" in args.metrics:
        scores.update(
            compute_cov_mmd_1nna(
                pcs_gen=subsample(summary["gen-points"], 2048).to(device),
                pcs_ref=subsample(summary["test-points"], 2048).to(device),
                batch_size=256,
                metrics=("emd",),
            )
        )
    # as pointnet features
    if "fpd" in args.metrics:
        scores["fpd"] = compute_frechet_distance(
            feats1=summary["gen-feats"].cpu().numpy(),
            feats2=summary["train-feats"].cpu().numpy(),
        )
    # as pointnet features
    if "kpd" in args.metrics:
        scores["kpd"] = compute_squared_mmd(
            feats1=summary["gen-feats"].cpu().numpy(),
            feats2=summary["train-feats"].cpu().numpy(),
        )
    console.log(f"{scores=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size_per_gpu", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--metrics", type=str, default="swd,jsd,1nna,fpd,kpd")
    args = parser.parse_args()
    args.metrics = args.metrics.replace(" ", "").split(",")

    assert torch.cuda.is_available(), "no visible cuda devices"

    evaluate(args)
