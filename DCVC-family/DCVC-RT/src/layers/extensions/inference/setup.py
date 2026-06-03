# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import glob
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


cxx_flags = ["-O3"]
nvcc_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization", "-arch=native"]
if sys.platform == 'win32':
    cxx_flags = ["/O2"]


setup(
    name='inference_extensions_cuda',
    ext_modules=[
        CUDAExtension(
            name='inference_extensions_cuda',
            sources=glob.glob('*.cpp') + glob.glob('*.cu'),
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
