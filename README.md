# DUSty v2: Generative Range Imaging for Learning Scene Priors of 3D LiDAR Data

![interpolation](https://user-images.githubusercontent.com/9032347/230576998-34c6de2f-76ca-4892-929a-076e74d77b08.gif)

**Generative Range Imaging for Learning Scene Priors of 3D LiDAR Data**<br>
[<u>Kazuto Nakashima</u>](https://kazuto1011.github.io/), Yumi Iwashita, Ryo Kurazume<br>
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023<br>
[project](https://kazuto1011.github.io/dusty-gan-v2) | [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Nakashima_Generative_Range_Imaging_for_Learning_Scene_Priors_of_3D_LiDAR_WACV_2023_paper.pdf) | [supplemental](https://openaccess.thecvf.com/content/WACV2023/supplemental/Nakashima_Generative_Range_Imaging_WACV_2023_supplemental.pdf) | [arxiv](http://arxiv.org/abs/2210.11750) | [slide](https://kazuto1011.github.io/docs/slides/nakashima2023generative_teaser.pdf)

</center>

We propose GAN-based LiDAR data priors for sim2real and restoration tasks, which is an extension of our previous work, [DUSty [Nakashima et al. IROS'21]](https://kazuto1011.github.io/dusty-gan).

The core idea is to represent LiDAR range images as a continuous-image generative model or 2D neural fields. This model generates a range value and the corresponding dropout probability from a laser radiation angle. The generative process is trained using a GAN framework. For more details on the architecture, please refer to our [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Nakashima_Generative_Range_Imaging_for_Learning_Scene_Priors_of_3D_LiDAR_WACV_2023_paper.pdf) and [supplementary materials](https://openaccess.thecvf.com/content/WACV2023/supplemental/Nakashima_Generative_Range_Imaging_WACV_2023_supplemental.pdf).

![arch](https://user-images.githubusercontent.com/9032347/230577070-526852c6-35c6-42fa-b1bf-5f96e0ea2897.png)

## Setup

### Python environment + CUDA

The environment can be built using Anaconda. This command installs the CUDA 11.X runtime, however, we require PyTorch JIT compilation for the `gans/` directory. Please also install the corresponding CUDA locally.

```sh
$ conda env create -f environment.yaml
$ conda activate dusty-gan-v2
```

### Quick demo

The following demo generates random range images.

```sh
$ python quick_demo.py --arch dusty_v2
```

[Pretrained weights](https://github.com/kazuto1011/dusty-gan-v2/releases/tag/weights-wacv23) are automatically downloaded.
The `--arch` option can also be set to our baselines: `vanilla` and `dusty_v1`.

![dustyv2](https://user-images.githubusercontent.com/9032347/230577081-e8f7559e-5a36-4028-9f05-a4be7ab3c8aa.png)

### Dataset

To train models by your own or run the other demos, please download the [KITTI Raw]() dataset and make a symbolic link.

```sh
$ ln -sf <path to kitti raw root> ./data/kitti_raw
$ ls ./data/kitti_raw
2011_09_26  2011_09_28  2011_09_29  2011_09_30  2011_10_03
```

To check the KITTI data loader:

```sh
$ python -m gans.datasets.kitti
```

![dataset](https://user-images.githubusercontent.com/9032347/230577079-0182f034-3f6a-4f8d-99ee-9ab3fbe356a8.png)

## Training GANs

To train GANs on KITTI:

```sh
$ python train_gan.py --config configs/gans/dusty_v2.yaml  # ours
$ python train_gan.py --config configs/gans/dusty_v1.yaml  # baseline
$ python train_gan.py --config configs/gans/vanilla.yaml   # baseline
```

To monitor losses and images:

```sh
$ tensorboard --logdir ./logs
```

## Evaluation

```sh
$ python test_gan.py --ckpt_path <path to *.pth file> --metrics swd,jsd,1nna,fpd,kpd
```

|options|modality|metrics|
|:-|:-|:-|
|`swd`|2D inverse depth maps|Sliced Wasserstein distance (SWD)|
|`jsd`|3D point clouds|Jensen–Shannon divergence (JSD)|
|`1nna`|3D point clouds|Coverage (COV), minimum matching distance (MMD), and 1-nearest neighbor accuracy (1-NNA), based on the earth mover's distance (EMD)|
|`fpd`|PointNet features|Fréchet pointcloud distance (FPD)|
|`kpd`|PointNet features|Squared maximum mean discrepancy (like KID in the image domain)|

Note: `--ckpt_path` can also be the following keywords: `dusty_v2`, `dusty_v1`, or `vanilla`. In this case, the pre-trained weights are automatically downloaded.

## Demo

### Latent interpolation

```sh
$ python demo_interpolation.py --mode 2d --ckpt_path <path to *.pth file>
```

`--mode 2d`

https://user-images.githubusercontent.com/9032347/230571979-2fe94796-2df0-4e6e-9f11-f1c1d941a85d.mp4

`--mode 3d`

https://user-images.githubusercontent.com/9032347/230582262-2d900d8e-f701-4191-a534-1439aa07e8d7.mp4

### GAN inversion

```sh
$ python demo_inversion.py --ckpt_path <path to *.pth file>
```

https://user-images.githubusercontent.com/9032347/230580669-6c650b01-0e31-4a5c-9274-ac739731b247.mp4

## Sim2Real semantic segmentation

The `semseg/` directory includes an implementation of Sim2Real semantic segmentation. The basic setup is to train the SqueezeSegV2 model [Wu et al. ICRA'19] on GTA-LiDAR (simulation) and test it on KITTI (real). To mitigate the domain gap, our paper proposed reproducing the ray-drop noises onto the simulation data using our learned GAN. For details, please refer to our [paper (Section 4.2)](https://openaccess.thecvf.com/content/WACV2023/papers/Nakashima_Generative_Range_Imaging_for_Learning_Scene_Priors_of_3D_LiDAR_WACV_2023_paper.pdf).

### Dataset

1. Please setup GTA-LiDAR (simulation) and KITTI (real) datasets provided by the [SqueezeSegV2 repository](https://github.com/xuanyuzhou98/SqueezeSegV2).

```yaml
├── GTAV  # GTA-LiDAR
│   ├──1
│   │  ├── 00000000.npy
│   │  └── ...
│   └── ...
├── ImageSet  # KITTI
│   ├── all.txt
│   ├── train.txt
│   └── val.txt
└── lidar_2d  # KITTI
    ├── 2011_09_26_0001_0000000000.npy
    └── ...
```

1. Compute the raydrop probability map (64x512 shape) for each GTA-LiDAR depth map (`*.npy`) using the GAN inversion, and save them with the same structure. *We will also release the pre-computed data.*

```yaml
data/kitti_raw_frontal
├── GTAV
│   ├──1
│   │  ├── 00000000.npy
│   │  └── ...
│   └── ...
├── GTAV_noise_v1  # computed with DUSty v1
│   ├──1
│   │  ├── 00000000.npy
│   │  └── ...
│   └── ...
├── GTAV_noise_v2  # computed with DUSty v2
│   ├──1
│   │  ├── 00000000.npy
│   │  └── ...
│   └── ...
```

3. Finally, please make a symbolic link.

```sh
$ ln -sf <a path to the root above> ./data/kitti_raw_frontal
```

### Training

Training configuration files can be found in `configs/semseg/`. We compare five approaches (config-A-E) to reproduce the raydrop noises.

```sh
$ python train_semseg.py --config <path to *.yaml file>
```

|config|training domain|raydrop probability|file|
|:-|:-|:-|:-|
| A |Simulation||`configs/semseg/sim2real_wo_noise.yaml`             |
| B |Simulation|Global frequency|`configs/semseg/sim2real_w_uniform_noise.yaml`      |
| C |Simulation|Pixel-wise frequency|`configs/semseg/sim2real_w_spatial_noise.yaml`      |
| D |Simulation|Computed w/ DUSty v1|`configs/semseg/sim2real_w_gan_noise_dustyv1.yaml`  |
| E |Simulation|Computed w/ DUSty v2|`configs/semseg/sim2real_w_gan_noise_dustyv2.yaml`  |
| F |Real|N/A|`configs/semseg/real2real.yaml`                     |

Note: `--ckpt_path` can also be the following keywords: `clean`, `uniform`, `spatial`, `dusty_v1`, `dusty_v2`, or `real`. In this case, the pre-trained weights are automatically downloaded.

### Evaluation

```sh
$ python test_semseg.py --ckpt_path <path to *.pth file>
```

## Citation

```bibtex
@InProceedings{nakashima2023wacv,
    author    = {Nakashima, Kazuto and Iwashita, Yumi and Kurazume, Ryo},
    title     = {Generative Range Imaging for Learning Scene Priors of 3{D} LiDAR Data},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year      = {2023},
    pages     = {1256-1266}
}
```