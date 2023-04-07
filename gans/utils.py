import gc
import os
import os.path as osp
import random

import cv2
import imageio
import matplotlib
import numpy as np
import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rich import print
from tqdm.auto import tqdm

from .geometry import estimate_surface_normal


def init_random_seed(random_seed=0, rank=0):
    seed = random_seed + rank
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def init_dist_process(rank, temp_dir, num_gpus, random_seed):
    init_random_seed(random_seed, rank)
    init_method = (temp_dir / ".torch_distributed_init").resolve()
    init_method = f"file://{init_method}"
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=num_gpus,
        rank=rank,
    )


def init_weights(cfg):
    init_type = cfg.init.type
    gain = cfg.init.gain
    nonlinearity = cfg.relu_type

    def init_func(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight, gain=gain)
            elif init_type == "kaiming":
                if nonlinearity == "relu":
                    nn.init.kaiming_normal_(m.weight, 0, "fan_in", "relu")
                elif nonlinearity == "leaky_relu":
                    nn.init.kaiming_normal_(m.weight, 0.2, "fan_in", "learky_relu")
                else:
                    raise NotImplementedError(f"Unknown nonlinearity: {nonlinearity}")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f"Unknown initialization: {init_type}")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_func


def set_requires_grad(net, requires_grad: bool = True):
    for param in net.parameters():
        param.requires_grad = requires_grad


def zero_grad(optim):
    for group in optim.param_groups:
        for p in group["params"]:
            p.grad = None


def sigmoid_to_tanh(x: torch.Tensor):
    """[0,1] -> [-1,+1]"""
    out = x * 2.0 - 1.0
    return out


def tanh_to_sigmoid(x: torch.Tensor):
    """[-1,+1] -> [0,1]"""
    out = (x + 1.0) / 2.0
    return out


def get_device(cuda: bool):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        for i in range(torch.cuda.device_count()):
            print("GPU {}: {}".format(i, torch.cuda.get_device_name(i)))
    else:
        print("CPU")
    return device


def noise(tensor: torch.Tensor, std: float = 0.1):
    noise = tensor.clone().normal_(0, std)
    return tensor + noise


def print_gc():
    # https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(type(obj), obj.size())
        except:
            pass


def cycle(iterable):
    while True:
        yield from iterable


def save_video(frames, filename, fps=30.0, save_frames=False, save_gif=False):
    assert len(frames) > 0, "no frame"
    if save_gif:
        frames = [Image.fromarray(f) for f in frames]
        frames[0].save(
            filename + ".gif",
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=1 / fps * 1e3,
            loop=0,
        )
        print("Saved:", filename + ".gif")
    else:
        if len(filename.split("/")) > 1:
            os.makedirs(osp.dirname(filename), exist_ok=True)
        writer = imageio.get_writer(f"{filename}.mp4", mode="I", fps=fps)
        for i, frame in enumerate(tqdm(frames, desc="saving video...", leave=False)):
            writer.append_data(frame)
            if save_frames:
                cv2.imwrite(filename + f"_{i:05d}.png", frame[..., [2, 1, 0]])
        writer.close()
        cv2.destroyAllWindows()
        print("Saved:", f"{filename}.mp4")


def colorize(tensor, cmap="turbo"):
    if tensor.ndim == 4:
        B, C, H, W = tensor.shape
        assert C == 1, f"expected (B,1,H,W) tensor, but got {tensor.shape}"
        tensor = tensor.squeeze(1)
    assert tensor.ndim == 3, f"got {tensor.ndim}!=3"

    device = tensor.device

    if isinstance(cmap, np.ndarray):
        colors = cmap
    elif hasattr(matplotlib.cm, cmap):
        colors = eval(f"matplotlib.cm.{cmap}")(np.linspace(0, 1, 256))[:, :3]
    elif hasattr(seaborn.cm, cmap):
        colors = eval(f"seaborn.cm.{cmap}")(np.linspace(0, 1, 256))[:, :3]
    else:
        raise ValueError(f"unknown cmap: {cmap}")
    color_map = torch.tensor(colors, device=device).float()  # (256,3)
    num_colors, _ = color_map.shape

    tensor = tensor * num_colors
    tensor = tensor.clamp(0, num_colors - 1)
    index = tensor.long()

    return F.embedding(index, color_map).permute(0, 3, 1, 2)


def flatten(tensor_BCHW):
    return tensor_BCHW.flatten(2).permute(0, 2, 1).contiguous()


def points_to_normal_2d(points_map, mode="closest", d=2):
    normals = -estimate_surface_normal(points_map, d=d, mode=mode)
    normals[normals != normals] = 0.0
    normals = tanh_to_sigmoid(normals).clamp_(0.0, 1.0)
    return normals


def power_spectrum_2d(x):
    specrum = torch.fft.fft2(x, norm="forward")
    specrum = torch.fft.fftshift(specrum, dim=[-1, -2])
    specrum = 10 * torch.log10(specrum.abs() ** 2)
    return specrum


class SphericalOptimizer(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.params = params

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for param in self.params:
            param.data.div_(param.pow(2).mean(dim=1, keepdim=True).add(1e-9).sqrt())
        return loss


def masked_loss(img_ref, img_gen, mask, distance="l1"):
    if distance == "l1":
        loss = F.l1_loss(img_ref, img_gen, reduction="none")
    elif distance == "l2":
        loss = F.mse_loss(img_ref, img_gen, reduction="none")
    else:
        raise NotImplementedError
    loss = (loss * mask).sum(dim=(1, 2, 3))
    loss = loss / mask.sum(dim=(1, 2, 3))
    return loss


# this is from https://github.com/NVlabs/stylegan3
class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(
        self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5
    ):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1
