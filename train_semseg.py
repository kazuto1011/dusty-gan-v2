import argparse
import datetime
import tempfile
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from rich import print
from rich.console import Console
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from semseg.datasets.sqsg import GTALiDAR, GTALiDAR_GAN, KITTIRawFrontal
from semseg.models.loss import FocalLoss
from semseg.models.squeezeseg_v1 import SqueezeSegV1
from semseg.models.squeezeseg_v2 import SqueezeSegV2
from semseg.utils import InfiniteSampler, colorize, init_dist_process
from test_semseg import evaluate

console = Console()


def report_scores(matrix, writer, step, tag, cls_name=None, eps=1e-12):
    if cls_name is None:
        id2str = lambda c: str(c)
    else:
        id2str = lambda c: str(cls_name[c])
    iou = matrix["tp"] / (matrix["tp"] + matrix["fn"] + matrix["fp"] + eps)
    precision = matrix["tp"] / (matrix["tp"] + matrix["fp"] + eps)
    recall = matrix["tp"] / (matrix["tp"] + matrix["fn"] + eps)
    num_classes = len(iou)
    for c in range(num_classes):
        writer.add_scalar(f"{tag}/iou/class-{id2str(c)}", iou[c], step)
        writer.add_scalar(f"{tag}/precision/class-{id2str(c)}", precision[c], step)
        writer.add_scalar(f"{tag}/recall/class-{id2str(c)}", recall[c], step)
    # ignore class 0
    writer.add_scalar(f"{tag}/iou/mean", iou[1:].mean(), step)
    writer.add_scalar(f"{tag}/precision/mean", precision[1:].mean(), step)
    writer.add_scalar(f"{tag}/recall/mean", recall[1:].mean(), step)


def make_inputs(item, modalities):
    inputs = []
    for m in modalities:
        t = item[m]
        if t.ndim == 3:
            t = t[:, None, :, :]
        inputs.append(t)
    return torch.cat(inputs, dim=1)


def resize_label(label, size):
    label = label[:, None].float()
    label = F.interpolate(label, size=size, mode="nearest-exact")
    return label[:, 0].long()


def training_loop(rank, cfg, temp_dir, log_dir):
    # random seed
    gpu_info = torch.cuda.get_device_properties(rank)
    console.log(
        f"rank {rank}: {gpu_info.name} {gpu_info.total_memory / 1024**3:g} GB, "
        + f"{cfg.training.num_workers} workers"
    )
    init_dist_process(rank, temp_dir, cfg.training.num_gpus, cfg.random_seed)

    # memory format
    device = torch.device("cuda", rank)
    to_kwargs = {"device": device, "non_blocking": True}

    # ---------------------------------------------------------------------------
    # dataset
    if cfg.dataset.name == "kitti_raw_frontal":
        train_dataset = KITTIRawFrontal(split="train", flip=cfg.dataset.random_flip)
        val_dataset = KITTIRawFrontal(split="val")
    elif cfg.dataset.name == "gta_lidar":
        dropout_map = np.load("data/avg_raydrop/kitti_raw_frontal.npy")
        train_dataset = GTALiDAR(flip=cfg.dataset.random_flip, raydrop_p=dropout_map)
        val_dataset = KITTIRawFrontal(split="val")
    elif cfg.dataset.name == "gta_lidar_w_uniform_noise":
        dropout_map = np.load("data/avg_raydrop/kitti_raw_frontal.npy")
        dropout_map.fill(dropout_map.mean())
        train_dataset = GTALiDAR(flip=cfg.dataset.random_flip, raydrop_p=dropout_map)
        val_dataset = KITTIRawFrontal(split="val")
    elif cfg.dataset.name == "gta_lidar_w_gan_noise_dustyv1":
        train_dataset = GTALiDAR_GAN(
            flip=cfg.dataset.random_flip, gan_dir="GTAV_noise_v1"
        )
        val_dataset = KITTIRawFrontal(split="val")
    elif cfg.dataset.name == "gta_lidar_w_gan_noise_dustyv2":
        train_dataset = GTALiDAR_GAN(
            flip=cfg.dataset.random_flip, gan_dir="GTAV_noise_v2"
        )
        val_dataset = KITTIRawFrontal(split="val")
    elif cfg.dataset.name == "gta_lidar_wo_noise":
        train_dataset = GTALiDAR(flip=cfg.dataset.random_flip, raydrop_p=None)
        val_dataset = KITTIRawFrontal(split="val")
    else:
        raise ValueError(cfg.dataset.name)

    if rank == 0:
        print(f"{train_dataset=}")
        print(f"{val_dataset=}")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size_per_gpu,
        sampler=InfiniteSampler(
            train_dataset,
            rank=rank,
            num_replicas=cfg.training.num_gpus,
            seed=cfg.random_seed + rank,
        ),
        num_workers=cfg.training.num_workers,
        shuffle=False,
        drop_last=True,
    )
    iter_train_loader = iter(train_loader)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.training.batch_size_per_gpu,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        drop_last=False,
    )

    # ---------------------------------------------------------------------------
    # model
    if cfg.arch.name == "squeezeseg_v1":
        model = SqueezeSegV1(
            inputs=cfg.arch.inputs,
            num_classes=cfg.dataset.num_classes,
            head_dropout_p=cfg.arch.decoder.dropout_p,
            use_crf=cfg.arch.use_crf,
            crf_kernel_size=cfg.arch.crf.kernel_size,
            crf_init_weight_smoothness=cfg.arch.crf.init_weight_smoothness,
            crf_init_weight_appearance=cfg.arch.crf.init_weight_appearance,
            crf_theta_gamma=cfg.arch.crf.theta_gamma,
            crf_theta_alpha=cfg.arch.crf.theta_alpha,
            crf_theta_beta=cfg.arch.crf.theta_beta,
            crf_num_iters=cfg.arch.crf.num_iters,
        )
    elif cfg.arch.name == "squeezeseg_v2":
        model = SqueezeSegV2(
            inputs=cfg.arch.inputs,
            num_classes=cfg.dataset.num_classes,
            bn_momentum=cfg.arch.bn_momentum,
            head_dropout_p=cfg.arch.decoder.dropout_p,
            use_crf=cfg.arch.use_crf,
            crf_kernel_size=cfg.arch.crf.kernel_size,
            crf_init_weight_smoothness=cfg.arch.crf.init_weight_smoothness,
            crf_init_weight_appearance=cfg.arch.crf.init_weight_appearance,
            crf_theta_gamma=cfg.arch.crf.theta_gamma,
            crf_theta_alpha=cfg.arch.crf.theta_alpha,
            crf_theta_beta=cfg.arch.crf.theta_beta,
            crf_num_iters=cfg.arch.crf.num_iters,
        )
        if cfg.dataset.logit_bias is not None:
            # https://github.com/xuanyuzhou98/SqueezeSegV2/blob/master/src/nn_skeleton.py#L429-L431
            logit_bias = torch.tensor(cfg.dataset.logit_bias)
            model.decoder.head[1].bias.data = -torch.log((1 - logit_bias) / logit_bias)
    else:
        raise ValueError(cfg.arch.name)

    model.to(**to_kwargs)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(module=model, device_ids=[rank])
    model.train()

    # ---------------------------------------------------------------------------
    # loss
    if cfg.loss.name == "focal_loss":
        criterion = FocalLoss(
            gamma=float(cfg.loss.focal_gamma),
            alpha=torch.tensor(cfg.loss.cls_weight).float(),
            reduction="none",
        )
    elif cfg.loss.name == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(cfg.loss.cls_weight).float(),
            reduction="none",
        )
    else:
        raise ValueError(cfg.loss.name)
    criterion.to(**to_kwargs)

    def masked_loss(logit, label, mask):
        loss = criterion(logit, label) * mask
        loss = loss.sum() / mask.sum()
        return loss

    # ---------------------------------------------------------------------------
    # optimizer
    params = list(model.parameters())
    optimizer = optim.SGD(
        params=params,
        lr=cfg.training.lr,
        momentum=cfg.training.lr_momentum,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=cfg.training.lr_decay,
    )
    grad_scaler = GradScaler(enabled=cfg.use_amp)

    # ---------------------------------------------------------------------------
    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir / "tensorboard")
        OmegaConf.save(cfg, log_dir / "training_config.yaml")
        model_dir = log_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # fixed batch for visualization
        item = next(iter(val_loader))
        inputs_vis = make_inputs(item, cfg.arch.inputs).to(**to_kwargs)
        depth_vis = item["depth"].to(**to_kwargs)
        xyz_vis = item["xyz"].to(**to_kwargs)
        label_vis = item["label"].long().to(**to_kwargs)
        mask_vis = item["mask"].float().to(**to_kwargs)

    def make_summary(depth, mask, pred, label):
        summary = [
            colorize(TF.normalize(depth, [-1], [5]), "turbo"),
            colorize(mask, "binary_r"),
            colorize(pred / (cfg.dataset.num_classes - 1), "jet") * mask[:, None],
            colorize(label / (cfg.dataset.num_classes - 1), "jet") * mask[:, None],
        ]
        return torch.cat(summary, dim=2)

    # ---------------------------------------------------------------------------
    # training loop
    confusion_matrix_train = defaultdict(int)
    moving_avg = deque(maxlen=100)

    for step in tqdm(
        range(1, cfg.training.max_steps + 1),
        desc="step",
        dynamic_ncols=True,
        disable=not rank == 0,
    ):
        item = next(iter_train_loader)
        model.train()

        # fetch data
        depth = item["depth"].to(**to_kwargs)
        xyz = item["xyz"].to(**to_kwargs)
        label = item["label"].to(**to_kwargs)
        mask = item["mask"].to(**to_kwargs)
        inputs = make_inputs(item, cfg.arch.inputs).to(**to_kwargs)

        # forward
        with autocast(enabled=cfg.use_amp):
            logit = model(inputs, xyz, mask)
            if isinstance(logit, tuple):
                loss = 0
                for logit_i in logit:
                    _, _, H, W = logit_i.shape
                    label_i = resize_label(label, (H, W))
                    mask_i = resize_label(mask, (H, W))
                    loss_i = masked_loss(logit_i, label_i, mask_i)
                    loss += loss_i
                pred = logit[-1].argmax(dim=1).detach()
            else:
                loss = masked_loss(logit, label, mask)
                pred = logit.argmax(dim=1).detach()

        loss = loss * float(cfg.loss.cls_loss_coef)

        # backward
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        moving_avg.append(loss.item())

        # logging
        if rank == 0:
            # set missing points as 'unknown'
            pred = (pred * mask).long()
            label = (label * mask).long()

            _, tps, fps, fns = evaluate(label, pred, cfg.dataset.num_classes)
            confusion_matrix_train["tp"] += tps
            confusion_matrix_train["fp"] += fps
            confusion_matrix_train["fn"] += fns

            # scores, loss, lr
            if step % cfg.training.checkpoint.stats == 0:
                report_scores(
                    confusion_matrix_train,
                    writer,
                    step,
                    "train",
                    train_dataset.class_list,
                )
                confusion_matrix_train = defaultdict(int)  # reset
                writer.add_scalar("train/loss", np.mean(moving_avg), step)
                for i, o in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"train/lr/group_{i}", o["lr"], step)

            # visual progress
            if step % cfg.training.checkpoint.image == 0:
                summary = make_summary(depth, mask, pred, label)
                writer.add_images("train/prediction", summary, step)
                model.eval()
                with torch.inference_mode():
                    logit_vis = model(inputs_vis, xyz_vis, mask_vis)
                    pred_vis = logit_vis.argmax(dim=1).detach()
                summary = make_summary(depth_vis, mask_vis, pred_vis, label_vis)
                writer.add_images("val/prediction", summary, step)

            if step % cfg.training.checkpoint.test == 0:
                # validation scores
                model.eval()
                confusion_matrix_val = defaultdict(int)
                for item in tqdm(val_loader, desc="validation", leave=False):
                    xyz = item["xyz"].to(**to_kwargs)
                    label = item["label"].to(**to_kwargs)
                    mask = item["mask"].to(**to_kwargs)
                    inputs = make_inputs(item, cfg.arch.inputs).to(**to_kwargs)

                    with autocast(enabled=cfg.use_amp):
                        with torch.inference_mode():
                            logit = model(inputs, xyz, mask)
                            pred = logit.argmax(dim=1).detach()
                            pred = (pred * mask).long()
                            label = (label * mask).long()

                    _, tps, fps, fns = evaluate(label, pred, cfg.dataset.num_classes)
                    confusion_matrix_val["tp"] += tps
                    confusion_matrix_val["fp"] += fps
                    confusion_matrix_val["fn"] += fns

                report_scores(
                    confusion_matrix_val, writer, step, "val", val_dataset.class_list
                )

                # saving weights
                torch.save(
                    {
                        "cfg": cfg,
                        "step": step,
                        "model": model.module.state_dict(),
                        "optim": optimizer.state_dict(),
                    },
                    model_dir / f"checkpoint_step-{step:010d}.pth",
                )

        # scheduled lr decay
        if step % cfg.training.lr_decay_steps == 0:
            scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
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
    log_dir = Path(f"logs/semseg/{cfg.dataset.name}/{cfg.arch.name}")
    log_dir /= datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / "training_config.yaml")

    with tempfile.TemporaryDirectory() as temp_dir:
        torch.multiprocessing.spawn(
            training_loop,
            args=(cfg, Path(temp_dir), log_dir),
            nprocs=cfg.training.num_gpus,
        )
