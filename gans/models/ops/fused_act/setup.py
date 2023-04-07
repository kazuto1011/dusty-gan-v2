import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

module_path = os.path.dirname(__file__)
setup(
    name="fused",
    ext_modules=[
        CUDAExtension(
            name="fused",
            sources=[
                os.path.join(module_path, "fused_bias_act.cpp"),
                os.path.join(module_path, "fused_bias_act_kernel.cu"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
