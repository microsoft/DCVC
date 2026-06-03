# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import glob
import sys
import psutil
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Note: assume inference kernel is only built upon NVIDIA platforms.
major, minor = torch.cuda.get_device_capability(0)
sm = major * 10 + minor
if sm == 90:
    arch_str = '--generate-code=arch=compute_90a,code=[sm_90a]'
elif sm == 100:
    arch_str = '--generate-code=arch=compute_100a,code=[sm_100a]'
else:
    arch_str = '-arch=native'

cxx_flags = ['-O3', '-Wno-deprecated-declarations']
# Per-nvcc internal parallelism via --split-compile=N. The inference
# extension is CUTLASS-heavy, so individual .cu translation units are
# large and benefit from intra-nvcc parallelism on top of ninja's
# cross-file parallelism. Empirically saturates around 4; pick N based
# on cpu_count so smaller hosts still leave headroom for ninja.
cpu_count = os.cpu_count() or 4
if cpu_count < 16:
    nvcc_split_compile = 1
elif cpu_count < 32:
    nvcc_split_compile = 2
else:
    nvcc_split_compile = 4
nvcc_flags = [f'-DCURRENT_DEVICE_SM={sm}', '-O3', '--use_fast_math',
              '--extra-device-vectorization', arch_str, '-Wno-deprecated-declarations',
              f'--split-compile={nvcc_split_compile}']
if sm == 90:
    nvcc_flags.append('-DCUTLASS_ENABLE_GDC_FOR_SM90=1')
elif sm == 100:
    nvcc_flags.append('-DCUTLASS_ENABLE_GDC_FOR_SM100=1')

if sys.platform == 'win32':
    cxx_flags = ['/O2']

cutlass_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../../../../third_party/cutlass')
py_rans_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../../../cpp/py_rans')

mem = psutil.virtual_memory()
mem_gb = mem.available / 1024**3
if mem_gb < 32:
    os.environ['MAX_JOBS'] = '8'
elif mem_gb < 64:
    os.environ['MAX_JOBS'] = '16'

setup(
    name='inference_extensions_cuda',
    ext_modules=[
        CUDAExtension(
            name='inference_extensions_cuda',
            include_dirs=[
                os.path.join(cutlass_path, 'include'),
                os.path.join(cutlass_path, 'tools', 'util', 'include'),
                py_rans_path
            ],
            sources=(
                glob.glob('**/*.cpp', recursive=True) +
                glob.glob('**/*.cu', recursive=True) +
                [f'{py_rans_path}/rans.cpp', f'{py_rans_path}/py_rans.cpp',]
            ),
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': nvcc_flags,
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
