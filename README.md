# DUSty v2: Generative Range Imaging for Learning Scene Priors of 3D LiDAR Data

![interpolation](https://user-images.githubusercontent.com/9032347/230576998-34c6de2f-76ca-4892-929a-076e74d77b08.gif)

**Generative Range Imaging for Learning Scene Priors of 3D LiDAR Data**<br>
[<u>Kazuto Nakashima</u>](https://kazuto1011.github.io/), Yumi Iwashita, Ryo Kurazume<br>
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023<br>
[project](https://kazuto1011.github.io/dusty-gan-v2) | [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Nakashima_Generative_Range_Imaging_for_Learning_Scene_Priors_of_3D_LiDAR_WACV_2023_paper.pdf) | [supplemental](https://openaccess.thecvf.com/content/WACV2023/supplemental/Nakashima_Generative_Range_Imaging_WACV_2023_supplemental.pdf) | [arxiv](http://arxiv.org/abs/2210.11750) | [slide](https://kazuto1011.github.io/docs/slides/nakashima2023generative_teaser.pdf)

</center>

We propose GAN-based LiDAR data priors for sim2real and restoration tasks. Extended version of our previous work [DUSty [Nakashima et al. IROS'21]](https://kazuto1011.github.io/dusty-gan).

The core idea is to represent LiDAR range images as a continuous image generative model (or 2D neural fields), which generates a range value and the corresponding dropout probability from a laser radiation angle. The generative process is trained by the GAN framework. Please check out our [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Nakashima_Generative_Range_Imaging_for_Learning_Scene_Priors_of_3D_LiDAR_WACV_2023_paper.pdf) and [supplementary materials](https://openaccess.thecvf.com/content/WACV2023/supplemental/Nakashima_Generative_Range_Imaging_WACV_2023_supplemental.pdf) for more details on the architecture.

![arch](https://user-images.githubusercontent.com/9032347/230577070-526852c6-35c6-42fa-b1bf-5f96e0ea2897.png)

## Setup

### Python environment + CUDA

The environment can be built by Anaconda. This command installs CUDA 11.X runtime, while we require the PyTorch JIT compillation for `gans/` dirs. Please also install the matching CUDA locally.

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
$ ln -sf <a path to the kitti raw root> ./data/kitti_raw
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

For convinience, let `$CKPT_PATH` be a path to the `checkpoint_*.pth` file hereafter.
`$CKPT_PATH` can also be the pretrained model IDs: `dusty_v2`, `dusty_v1`, or `vanilla`.

```sh
$ python test_gan.py --ckpt_path $CKPT_PATH --metrics swd,jsd,1nna,fpd,kpd
```

|options|modality|metrics|
|:-|:-|:-|
|`swd`|2D inverse depth maps|Sliced Wasserstein distance (SWD)|
|`jsd`|3D point clouds|Jensen–Shannon divergence (JSD)|
|`1nna`|3D point clouds|Coverage (COV), minimum matching distance (MMD), and 1-nearest neighbor accuracy (1-NNA), based on the earth mover's distance (EMD)|
|`fpd`|PointNet features|Fréchet pointcloud distance (FPD)|
|`kpd`|PointNet features|Squared maximum mean discrepancy (like KID in the image domain)|

## Demo

### Latent interpolation

```sh
$ python demo_interpolation.py --mode 2d --ckpt_path $CKPT_PATH
```

`--mode 2d`

https://user-images.githubusercontent.com/9032347/230571979-2fe94796-2df0-4e6e-9f11-f1c1d941a85d.mp4

`--mode 3d`

https://user-images.githubusercontent.com/9032347/230582262-2d900d8e-f701-4191-a534-1439aa07e8d7.mp4

### GAN inversion

```sh
$ python demo_inversion.py --ckpt_path $CKPT_PATH
```

https://user-images.githubusercontent.com/9032347/230580669-6c650b01-0e31-4a5c-9274-ac739731b247.mp4

## Sim2Real domain adaptation

This part is still on refactoring.

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