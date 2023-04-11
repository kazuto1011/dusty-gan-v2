import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rich import print
from rich.table import Table
from torch.cuda.amp import autocast
from torch.nn.parallel import DataParallel as DP
from tqdm import tqdm

from semseg.datasets.sqsg import KITTIRawFrontal
from semseg.models.knn import kNN2d
from semseg.models.squeezeseg_v1 import SqueezeSegV1
from semseg.models.squeezeseg_v2 import SqueezeSegV2
from semseg.pretrained import autoload_ckpt
from semseg.utils import get_device


def evaluate(label, pred, num_classes, epsilon=1e-12):
    # PyTorch version of https://github.com/xuanyuzhou98/SqueezeSegV2/blob/master/src/utils/util.py

    device = label.device
    ious = torch.zeros(num_classes, device=device)
    tps = torch.zeros(num_classes, device=device)
    fns = torch.zeros(num_classes, device=device)
    fps = torch.zeros(num_classes, device=device)

    for cls_id in range(num_classes):
        tp = (pred[label == cls_id] == cls_id).sum()
        fp = (label[pred == cls_id] != cls_id).sum()
        fn = (pred[label == cls_id] != cls_id).sum()

        ious[cls_id] = tp / (tp + fn + fp + epsilon)
        tps[cls_id] = tp
        fps[cls_id] = fp
        fns[cls_id] = fn

    return ious, tps, fps, fns


def make_inputs(item, modalities):
    inputs = []
    for m in modalities:
        t = item[m]
        if t.ndim == 3:
            t = t[:, None, :, :]
        inputs.append(t)
    return torch.cat(inputs, dim=1)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--knn_enabled", action="store_true")
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--knn_kernel_size", type=int, default=5)
    args = parser.parse_args()

    device = get_device(True)
    ckpt = autoload_ckpt(args.ckpt_path)
    cfg = ckpt["cfg"]

    # ---------------------------------------------------------------------------
    # dataset
    val_dataset = KITTIRawFrontal(split="val", omit_cyclist=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
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
        )
    elif cfg.arch.name == "squeezeseg_v2":
        model = SqueezeSegV2(
            inputs=cfg.arch.inputs,
            num_classes=cfg.dataset.num_classes,
            use_crf=cfg.arch.use_crf,
            bn_momentum=cfg.arch.bn_momentum,
            head_dropout_p=cfg.arch.decoder.dropout_p,
        )
    else:
        raise ValueError(cfg.arch.name)

    model.to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model = DP(model)

    # knn post processing
    if args.knn_enabled:
        knn = kNN2d(
            num_classes=cfg.dataset.num_classes,
            k=args.knn_k,
            kernel_size=args.knn_kernel_size,
        ).to(device)

    # ---------------------------------------------------------------------------
    # evaluation
    conf_matrix = defaultdict(int)
    for item in tqdm(val_loader, desc="validation"):
        # raw_depth = item_raw["depth"].to(device, non_blocking=True).float()
        # item = val_dataset.normalize(item_raw)
        xyz = item["xyz"].to(device, non_blocking=True).float()
        depth = item["depth"].to(device, non_blocking=True).float()
        label = item["label"].to(device, non_blocking=True).long()
        mask = item["mask"].to(device, non_blocking=True).float()
        inputs = make_inputs(item, cfg.arch.inputs).to(device)

        with torch.inference_mode():
            with autocast():
                logit = model(inputs, xyz, mask)
                preds = logit.argmax(dim=1)  # (B,H,W)

                # omit 'cyclist' class
                preds[preds == 3] = 0

                if args.knn_enabled:
                    preds = knn(depth, preds)

                preds = (preds * mask).detach()
                label = (label * mask).detach()

            _, tps, fps, fns = evaluate(label, preds, cfg.dataset.num_classes)
            conf_matrix["tp"] += tps
            conf_matrix["fp"] += fps
            conf_matrix["fn"] += fns

    eps = 1e-12
    union = conf_matrix["tp"] + conf_matrix["fn"] + conf_matrix["fp"]
    iou = conf_matrix["tp"] / (union + eps)
    precision = conf_matrix["tp"] / (conf_matrix["tp"] + conf_matrix["fp"] + eps)
    recall = conf_matrix["tp"] / (conf_matrix["tp"] + conf_matrix["fn"] + eps)

    table = Table("class", "iou", "precision", "recall")
    for i, name in enumerate(val_dataset.class_list):
        table.add_row(name, f"{iou[i]:.1%}", f"{precision[i]:.1%}", f"{recall[i]:.1%}")
    # omit 'unknown' and 'cyclist' classes
    table.add_row(
        "mean",
        f"{iou[1:3].mean():.1%}",
        f"{precision[1:3].mean():.1%}",
        f"{recall[1:3].mean():.1%}",
    )
    print(table)
