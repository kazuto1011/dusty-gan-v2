from pathlib import Path

import numba
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

_KITTI_RAW_RECORDS = {
    "calibration": [
        "2011_09_26_drive_0119_sync",
        "2011_09_28_drive_0225_sync",
        "2011_09_29_drive_0108_sync",
        "2011_09_30_drive_0072_sync",
        "2011_10_03_drive_0058_sync",
    ],
    "campus": [
        "2011_09_28_drive_0016_sync",
        "2011_09_28_drive_0021_sync",
        "2011_09_28_drive_0034_sync",
        "2011_09_28_drive_0035_sync",
        "2011_09_28_drive_0037_sync",
        "2011_09_28_drive_0038_sync",
        "2011_09_28_drive_0039_sync",
        "2011_09_28_drive_0043_sync",
        "2011_09_28_drive_0045_sync",
        "2011_09_28_drive_0047_sync",
    ],
    "city": [
        "2011_09_26_drive_0001_sync",
        "2011_09_26_drive_0002_sync",
        "2011_09_26_drive_0005_sync",
        "2011_09_26_drive_0009_sync",
        "2011_09_26_drive_0011_sync",
        "2011_09_26_drive_0013_sync",
        "2011_09_26_drive_0014_sync",
        "2011_09_26_drive_0017_sync",
        "2011_09_26_drive_0018_sync",
        "2011_09_26_drive_0048_sync",
        "2011_09_26_drive_0051_sync",
        "2011_09_26_drive_0056_sync",
        "2011_09_26_drive_0057_sync",
        "2011_09_26_drive_0059_sync",
        "2011_09_26_drive_0060_sync",
        "2011_09_26_drive_0084_sync",
        "2011_09_26_drive_0091_sync",
        "2011_09_26_drive_0093_sync",
        "2011_09_26_drive_0095_sync",
        "2011_09_26_drive_0096_sync",
        "2011_09_26_drive_0104_sync",
        "2011_09_26_drive_0106_sync",
        "2011_09_26_drive_0113_sync",
        "2011_09_26_drive_0117_sync",
        "2011_09_28_drive_0001_sync",
        "2011_09_28_drive_0002_sync",
        "2011_09_29_drive_0026_sync",
        "2011_09_29_drive_0071_sync",
    ],
    "person": [
        "2011_09_28_drive_0053_sync",
        "2011_09_28_drive_0054_sync",
        "2011_09_28_drive_0057_sync",
        "2011_09_28_drive_0065_sync",
        "2011_09_28_drive_0066_sync",
        "2011_09_28_drive_0068_sync",
        "2011_09_28_drive_0070_sync",
        "2011_09_28_drive_0071_sync",
        "2011_09_28_drive_0075_sync",
        "2011_09_28_drive_0077_sync",
        "2011_09_28_drive_0078_sync",
        "2011_09_28_drive_0080_sync",
        "2011_09_28_drive_0082_sync",
        "2011_09_28_drive_0086_sync",
        "2011_09_28_drive_0087_sync",
        "2011_09_28_drive_0089_sync",
        "2011_09_28_drive_0090_sync",
        "2011_09_28_drive_0094_sync",
        "2011_09_28_drive_0095_sync",
        "2011_09_28_drive_0096_sync",
        "2011_09_28_drive_0098_sync",
        "2011_09_28_drive_0100_sync",
        "2011_09_28_drive_0102_sync",
        "2011_09_28_drive_0103_sync",
        "2011_09_28_drive_0104_sync",
        "2011_09_28_drive_0106_sync",
        "2011_09_28_drive_0108_sync",
        "2011_09_28_drive_0110_sync",
        "2011_09_28_drive_0113_sync",
        "2011_09_28_drive_0117_sync",
        "2011_09_28_drive_0119_sync",
        "2011_09_28_drive_0121_sync",
        "2011_09_28_drive_0122_sync",
        "2011_09_28_drive_0125_sync",
        "2011_09_28_drive_0126_sync",
        "2011_09_28_drive_0128_sync",
        "2011_09_28_drive_0132_sync",
        "2011_09_28_drive_0134_sync",
        "2011_09_28_drive_0135_sync",
        "2011_09_28_drive_0136_sync",
        "2011_09_28_drive_0138_sync",
        "2011_09_28_drive_0141_sync",
        "2011_09_28_drive_0143_sync",
        "2011_09_28_drive_0145_sync",
        "2011_09_28_drive_0146_sync",
        "2011_09_28_drive_0149_sync",
        "2011_09_28_drive_0153_sync",
        "2011_09_28_drive_0154_sync",
        "2011_09_28_drive_0155_sync",
        "2011_09_28_drive_0156_sync",
        "2011_09_28_drive_0160_sync",
        "2011_09_28_drive_0161_sync",
        "2011_09_28_drive_0162_sync",
        "2011_09_28_drive_0165_sync",
        "2011_09_28_drive_0166_sync",
        "2011_09_28_drive_0167_sync",
        "2011_09_28_drive_0168_sync",
        "2011_09_28_drive_0171_sync",
        "2011_09_28_drive_0174_sync",
        "2011_09_28_drive_0177_sync",
        "2011_09_28_drive_0179_sync",
        "2011_09_28_drive_0183_sync",
        "2011_09_28_drive_0184_sync",
        "2011_09_28_drive_0185_sync",
        "2011_09_28_drive_0186_sync",
        "2011_09_28_drive_0187_sync",
        "2011_09_28_drive_0191_sync",
        "2011_09_28_drive_0192_sync",
        "2011_09_28_drive_0195_sync",
        "2011_09_28_drive_0198_sync",
        "2011_09_28_drive_0199_sync",
        "2011_09_28_drive_0201_sync",
        "2011_09_28_drive_0204_sync",
        "2011_09_28_drive_0205_sync",
        "2011_09_28_drive_0208_sync",
        "2011_09_28_drive_0209_sync",
        "2011_09_28_drive_0214_sync",
        "2011_09_28_drive_0216_sync",
        "2011_09_28_drive_0220_sync",
        "2011_09_28_drive_0222_sync",
    ],
    "residential": [
        "2011_09_26_drive_0019_sync",
        "2011_09_26_drive_0020_sync",
        "2011_09_26_drive_0022_sync",
        "2011_09_26_drive_0023_sync",
        "2011_09_26_drive_0035_sync",
        "2011_09_26_drive_0036_sync",
        "2011_09_26_drive_0039_sync",
        "2011_09_26_drive_0046_sync",
        "2011_09_26_drive_0061_sync",
        "2011_09_26_drive_0064_sync",
        "2011_09_26_drive_0079_sync",
        "2011_09_26_drive_0086_sync",
        "2011_09_26_drive_0087_sync",
        "2011_09_30_drive_0018_sync",
        "2011_09_30_drive_0020_sync",
        "2011_09_30_drive_0027_sync",
        "2011_09_30_drive_0028_sync",
        "2011_09_30_drive_0033_sync",
        "2011_09_30_drive_0034_sync",
        "2011_10_03_drive_0027_sync",
        "2011_10_03_drive_0034_sync",
    ],
    "road": [
        "2011_09_26_drive_0015_sync",
        "2011_09_26_drive_0027_sync",
        "2011_09_26_drive_0028_sync",
        "2011_09_26_drive_0029_sync",
        "2011_09_26_drive_0032_sync",
        "2011_09_26_drive_0052_sync",
        "2011_09_26_drive_0070_sync",
        "2011_09_26_drive_0101_sync",
        "2011_09_29_drive_0004_sync",
        "2011_09_30_drive_0016_sync",
        "2011_10_03_drive_0042_sync",
        "2011_10_03_drive_0047_sync",
    ],
}

_KITTI_RAW_TRAINVAL = (
    "2011_10_03_drive_0027_sync",
    "2011_10_03_drive_0042_sync",
    "2011_10_03_drive_0034_sync",
    "2011_09_26_drive_0067_sync",
    "2011_09_30_drive_0016_sync",
    "2011_09_30_drive_0018_sync",
    "2011_09_30_drive_0020_sync",
    "2011_09_30_drive_0027_sync",
    "2011_09_30_drive_0028_sync",
    "2011_09_30_drive_0033_sync",
    "2011_09_30_drive_0034_sync",
)

_KITTI_ODOMETRY_TO_RAW = {
    # sequence number, sequence name, start, end
    "00": ("2011_10_03_drive_0027_sync", int("000000"), int("004540")),
    "01": ("2011_10_03_drive_0042_sync", int("000000"), int("001100")),
    "02": ("2011_10_03_drive_0034_sync", int("000000"), int("004660")),
    "03": ("2011_09_26_drive_0067_sync", int("000000"), int("000800")),
    "04": ("2011_09_30_drive_0016_sync", int("000000"), int("000270")),
    "05": ("2011_09_30_drive_0018_sync", int("000000"), int("002760")),
    "06": ("2011_09_30_drive_0020_sync", int("000000"), int("001100")),
    "07": ("2011_09_30_drive_0027_sync", int("000000"), int("001100")),
    "08": ("2011_09_30_drive_0028_sync", int("001100"), int("005170")),
    "09": ("2011_09_30_drive_0033_sync", int("000000"), int("001590")),
    "10": ("2011_09_30_drive_0034_sync", int("000000"), int("001200")),
}

_SEQUENCE_SPLITS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    "val": [8],
    "test": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}


@numba.jit
def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array


class KITTIRaw(torch.utils.data.Dataset):
    """
    w/ scan unfolding
    """

    def __init__(
        self,
        root="data/kitti_raw",
        split="train",
        shape=(64, 2048),
        min_depth=0.9,
        max_depth=120.0,
        flip=False,
        scan_unfolding=True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        assert split in ("train", "val", "test")
        self.shape = tuple(shape)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.flip = flip
        self.scan_unfolding = scan_unfolding
        self.datalist = []

        if self.split in ("train", "val"):
            for subset in _SEQUENCE_SPLITS[self.split]:
                subset_idx = f"{subset:02d}"
                seq_name, start_idx, end_idx = _KITTI_ODOMETRY_TO_RAW[subset_idx]
                if subset_idx == "03":
                    continue  # kitti raw does not have 03 sequence
                for point_idx in range(start_idx, end_idx + 1):
                    fpath = f"{self.root}/{seq_name[:10]}/{seq_name}/velodyne_points/data/{point_idx:010d}.bin"
                    self.datalist.append(fpath)
        elif self.split in ("test"):
            for category in ["city", "road", "residential"]:
                for seq_name in _KITTI_RAW_RECORDS[category]:
                    if seq_name not in _KITTI_RAW_TRAINVAL:
                        fpaths = f"{self.root}/{seq_name[:10]}/{seq_name}/velodyne_points/data"
                        fpaths = sorted(Path(fpaths).glob("*.bin"))
                        self.datalist += fpaths

    def __getitem__(self, index):
        point_path = self.datalist[index]
        xyzrdm = self.load_pts_as_img(point_path, self.scan_unfolding)
        xyzrdm = TF.to_tensor(xyzrdm)  # (7,H,W)
        xyzrdm = TF.resize(xyzrdm, self.shape, InterpolationMode.NEAREST)
        xyzrdm *= xyzrdm[[5]]
        if self.flip and np.random.rand() > 0.5:
            xyzrdm = TF.hflip(xyzrdm)
        return {
            "xyz": xyzrdm[:3].float(),
            "reflectance": xyzrdm[[3]].float(),
            "depth": xyzrdm[[4]].float(),
            "mask": xyzrdm[[5]].float(),
        }

    def __len__(self):
        return len(self.datalist)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += ["Root location: {}".format(self.root)]
        body += ["Split: {}".format(self.split)]
        body += ["Scan unfolding: {}".format(self.scan_unfolding)]
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)

    def normalize(self, item):
        new_item = {}
        for key, value in item.items():
            if key in ("xyz", "reflectance", "depth"):
                value = TF.normalize(value, self.mean[key], self.std[key])
            new_item[key] = value
        return new_item

    @property
    def mean(self):
        return {
            "xyz": [-0.01506443, 0.45959818, -0.89225304],
            "reflectance": 0.24130844,
            "depth": 9.689281,
        }

    @property
    def std(self):
        return {
            "xyz": [11.224804, 8.237693, 0.88183135],
            "reflectance": 0.16860831,
            "depth": 10.08752,
        }

    def load_pts_as_img(self, point_path, scan_unfolding=True, H=64, W=2048):
        # load xyz & intensity and add depth & mask
        points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
        xyz = points[:, :3]  # xyz
        x = xyz[:, [0]]
        y = xyz[:, [1]]
        z = xyz[:, [2]]
        depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
        # mask = (depth > 0).astype(np.float32)
        mask = (depth >= self.min_depth) & (depth <= self.max_depth)
        points = np.concatenate([points, depth, mask], axis=1)

        if scan_unfolding:
            # the i-th quadrant
            # suppose the points are ordered counterclockwise
            quads = np.zeros_like(x, dtype=np.int32)
            quads[(x >= 0) & (y >= 0)] = 0  # 1st
            quads[(x < 0) & (y >= 0)] = 1  # 2nd
            quads[(x < 0) & (y < 0)] = 2  # 3rd
            quads[(x >= 0) & (y < 0)] = 3  # 4th

            # split between the 3rd and 1st quadrants
            diff = np.roll(quads, shift=1, axis=0) - quads
            delim_inds, _ = np.where(diff == 3)  # number of lines
            inds = list(delim_inds) + [len(points)]  # add the last index

            # vertical grid
            grid_h = np.zeros_like(x, dtype=np.int32)
            cur_ring_idx = H - 1  # ...0
            for i in reversed(range(len(delim_inds))):
                grid_h[inds[i] : inds[i + 1]] = cur_ring_idx
                if cur_ring_idx >= 0:
                    cur_ring_idx -= 1
                else:
                    break
        else:
            fup, fdown = np.deg2rad(3), np.deg2rad(-25)
            pitch = np.arcsin(z / depth) + abs(fdown)
            grid_h = 1 - pitch / (fup - fdown)
            grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)

        # horizontal grid
        yaw = -np.arctan2(y, x)  # [-pi,pi]
        grid_w = (yaw / np.pi + 1) / 2 % 1  # [0,1]
        grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)

        grid = np.concatenate((grid_h, grid_w), axis=1)

        # projection
        order = np.argsort(-depth.squeeze(1))
        proj_points = np.zeros((H, W, 4 + 2), dtype=points.dtype)
        proj_points = scatter(proj_points, grid[order], points[order])

        return proj_points


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision

    dmin, dmax = 1.45, 80.0
    dataset = KITTIRaw(split="train", shape=(64, 512), min_depth=dmin, max_depth=dmax)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    item = next(iter(loader))
    depth = 1 / item["depth"].add(1e-11) * item["mask"]
    grid = torchvision.utils.make_grid(depth, nrow=2, pad_value=float("nan"))[0].cpu()
    fig = plt.figure(constrained_layout=True)
    plt.imshow(grid, cmap="turbo", vmin=1 / dmax, vmax=1 / dmin, interpolation="none")
    plt.axis("off")
    plt.show()
