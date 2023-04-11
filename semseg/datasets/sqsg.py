from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF


class KITTIRawFrontal(torch.utils.data.Dataset):
    def __init__(
        self,
        root="data/kitti_raw_frontal",
        split="train",
        shape=(64, 512),
        min_depth=1.45,
        max_depth=80.0,
        flip=False,
        omit_cyclist=False,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.shape = tuple(shape)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.flip = flip
        self.omit_cyclist = omit_cyclist
        self.datalist = None
        self.load_datalist()
        assert split in ("all", "train", "val")

    def load_datalist(self):
        setlist = self.root / "ImageSet" / (self.split + ".txt")
        assert setlist.exists(), setlist
        with open(setlist) as f:
            self.datalist = [p.strip() + ".npy" for p in f.readlines()]

    def __getitem__(self, index):
        points_path = self.datalist[index]
        points = np.load(self.root / "lidar_2d" / points_path)  # (64,512,6)
        points = TF.to_tensor(points)
        points = TF.resize(points, self.shape, TF.InterpolationMode.NEAREST)
        mask = (points[4] > 0).float()
        points[:-1] *= mask[None]
        points = TF.normalize(points, self.mean, self.std)
        if np.random.rand() > 0.5 and self.flip:
            points = TF.hflip(points)
            points[[1]] *= -1  # flip y
            mask = TF.hflip(mask)
        if self.omit_cyclist:
            points[5][points[5] == 3] = 0
        return {
            "xyz": points[:3].float(),
            "reflectance": points[[3]].float(),
            "depth": points[[4]].float(),
            "label": points[5].long(),
            "mask": mask,
        }

    @property
    def mean(self):
        # x, y, z, intensity, depth, label
        return torch.tensor([10.88, 0.23, -1.04, 0.21, 12.12, 0.0])

    @property
    def std(self):
        # x, y, z, intensity, depth, label
        return torch.tensor([11.47, 6.91, 0.86, 0.16, 12.32, 1.0])

    @property
    def class_list(self):
        if self.omit_cyclist:
            return [
                "unknown",
                "car",
                "pedestrian",
            ]
        else:
            return [
                "unknown",
                "car",
                "pedestrian",
                "cyclist",
            ]

    def __len__(self):
        return len(self.datalist)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += ["Root location: {}".format(self.root)]
        body += ["Split: {}".format(self.split)]
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)


class GTALiDAR(torch.utils.data.Dataset):
    def __init__(
        self,
        root="data/kitti_raw_frontal",
        split="all",
        shape=(64, 512),
        min_depth=1.45,
        max_depth=80.0,
        flip=False,
        raydrop_p=None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.shape = tuple(shape)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.datalist = None
        self.load_datalist()
        self.flip = flip
        if raydrop_p is None:
            self.dropout_map = torch.ones(shape)
        else:
            assert raydrop_p.shape == shape
            self.dropout_map = torch.from_numpy(raydrop_p)
        assert split in ("all",)

    def load_datalist(self):
        data_dir = self.root / "GTAV"
        self.datalist = list(sorted(data_dir.glob("*/*.npy")))

    def __getitem__(self, index):
        points_path = self.datalist[index]
        points = np.load(points_path)  # (64,512,6)
        points = TF.to_tensor(points)
        points = TF.resize(points, self.shape, TF.InterpolationMode.NEAREST)
        mask = (points[3] > 0).float()
        mask *= torch.bernoulli(self.dropout_map)  # bernoulli sampling
        points[:-1] *= mask[None]
        points = TF.normalize(points, self.mean, self.std)
        if np.random.rand() > 0.5 and self.flip:
            points = TF.hflip(points)
            points[[1]] *= -1  # flip y
            mask = TF.hflip(mask)
        return {
            "xyz": points[:3].float(),
            "depth": points[[3]].float(),
            "label": points[4].long(),
            "mask": mask,
        }

    @property
    def mean(self):
        # x, y, z, depth, label (dummy)
        return torch.tensor([10.88, 0.23, -1.04, 12.12, 0.0])

    @property
    def std(self):
        # x, y, z, depth, label (dummy)
        return torch.tensor([11.47, 6.91, 0.86, 12.32, 1.0])

    @property
    def class_list(self):
        return [
            "unknown",
            "car",
            "pedestrian",
        ]

    def __len__(self):
        return len(self.datalist)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += ["Root location: {}".format(self.root)]
        body += ["Split: {}".format(self.split)]
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)


class GTALiDAR_GAN(torch.utils.data.Dataset):
    def __init__(
        self,
        root="data/kitti_raw_frontal",
        split="all",
        shape=(64, 512),
        min_depth=1.45,
        max_depth=80.0,
        flip=False,
        gan_dir="GTAV_noise",
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.shape = tuple(shape)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.datalist = None
        self.load_datalist()
        self.flip = flip
        self.gan_dir = gan_dir
        assert split in ("all",)

    def load_datalist(self):
        data_dir = self.root / "GTAV"
        self.datalist = list(sorted(data_dir.glob("*/*.npy")))

    def __getitem__(self, index):
        points_path = self.datalist[index]
        points = np.load(points_path)  # (64,512,6)
        points = TF.to_tensor(points)
        points = TF.resize(points, self.shape, TF.InterpolationMode.NEAREST)
        mask = (points[3] > 0).float()
        noise_path = str(points_path).replace("GTAV", self.gan_dir)
        dropout_map = torch.from_numpy(np.load(noise_path)).float()
        mask *= torch.bernoulli(dropout_map)  # bernoulli sampling
        points[:-1] *= mask[None]
        points = TF.normalize(points, self.mean, self.std)
        if np.random.rand() > 0.5 and self.flip:
            points = TF.hflip(points)
            points[[1]] *= -1  # flip y
            mask = TF.hflip(mask)
        return {
            "xyz": points[:3].float(),
            "depth": points[[3]].float(),
            "label": points[4].long(),
            "mask": mask,
        }

    @property
    def mean(self):
        # x, y, z, depth, label (dummy)
        return torch.tensor([10.88, 0.23, -1.04, 12.12, 0.0])

    @property
    def std(self):
        # x, y, z, depth, label (dummy)
        return torch.tensor([11.47, 6.91, 0.86, 12.32, 1.0])

    @property
    def class_list(self):
        return [
            "unknown",
            "car",
            "pedestrian",
        ]

    def __len__(self):
        return len(self.datalist)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += ["Root location: {}".format(self.root)]
        body += ["Split: {}".format(self.split)]
        body += ["GAN: {}".format(self.gan_dir)]
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision

    dmin, dmax = 1.45, 80.0
    dataset = GTALiDAR(split="all", shape=(64, 512), min_depth=dmin, max_depth=dmax)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    item = next(iter(loader))
    depth = 1 / item["depth"].add(1e-11) * item["mask"]
    grid = torchvision.utils.make_grid(depth, nrow=2, pad_value=float("nan"))[0].cpu()
    fig = plt.figure(constrained_layout=True)
    plt.imshow(grid, cmap="turbo", vmin=1 / dmax, vmax=1 / dmin, interpolation="none")
    plt.axis("off")
    plt.show()
