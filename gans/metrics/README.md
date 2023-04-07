## References

Distance implementations are borrowed from other repositories. Some parts were modified to enable JIT compilation. Our paper only used the Chamfer distance for computational reason.
* Chamfer distance (CD)
  * `metrics/distance/cd`: https://github.com/chrdiller/pyTorchChamferDistance
* Earth Mover's distance (EMD, option)
  * `metrics/distance/emd`: https://github.com/daerduoCarey/PyTorchEMD

This repository includes five types of distributional similarity metrics. The contents were from existing implementations, while we re-implemented with PyTorch and simplified them.

* Jensen-Shannon divergence (JSD)
  * `metrics/jsd.py`: https://github.com/optas/latent_3d_points
* Converage (COV)
  * `metrics/point_clouds.py`: https://github.com/stevenygd/PointFlow
* Minimum matching distance (MMD)
  * `metrics/point_clouds.py`: https://github.com/stevenygd/PointFlow
* 1-nearest neighbor accuracy (1-NNA)
  * `metrics/point_clouds.py`: https://github.com/stevenygd/PointFlow
* Sliced Wasserstein distance (SWD)
  * `metrics/swd.py`: https://github.com/tkarras/progressive_growing_of_gans, https://github.com/koshian2/swd-pytorch