from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


cxx_flags = ["-O3"]
nvcc_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization", "-arch=native"]


setup(
    name='block_mc_cpp',
    ext_modules=[
        CUDAExtension(
            name='block_mc_cpp_cuda',
            sources=[
                'block_mc.cpp',
                'block_mc_kernel.cu',
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },)
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
    )
