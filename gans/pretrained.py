import os

import torch
from torch.hub import load_state_dict_from_url

_TAG = "weights-wacv23"
_ROOT = f"https://github.com/kazuto1011/dusty-gan-v2/releases/download/{_TAG}/"

PRETRAINED_CKPTS = {
    "dusty_v1": _ROOT + "dustyv1_kitti_64x512_25M.pth",
    "dusty_v2": _ROOT + "dustyv2_kitti_64x512_25M.pth",
    "vanilla": _ROOT + "vanilla_kitti_64x512_25M.pth",
}


def is_available_model(name: str) -> bool:
    return name in PRETRAINED_CKPTS


def download_ckpt(ckpt_name: str):
    url = PRETRAINED_CKPTS[ckpt_name]
    ckpt = load_state_dict_from_url(url, progress=True)
    return ckpt


def autoload_ckpt(ckpt_name: str):
    if is_available_model(ckpt_name):
        ckpt = download_ckpt(ckpt_name)
    elif os.path.exists(ckpt_name):
        ckpt = torch.load(ckpt_name, map_location="cpu")
    else:
        raise ValueError(f"invalid model name: {ckpt_name}")
    return ckpt
