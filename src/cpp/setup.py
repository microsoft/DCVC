# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import glob
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


if sys.platform == "win32":
    extra_compile_args = ['/std:c++17', '/O2', '/W4', '/WX', '/wd4100']
    extra_link_args = []
else:
    extra_compile_args = ['-std=c++17', '-O3', '-fPIC', '-Wall', '-Wextra', '-Werror']
    extra_link_args = []


setup(
    name="MLCodec_extensions_cpp",
    ext_modules=[
        Pybind11Extension(
            name='MLCodec_extensions_cpp',
            sources=glob.glob('py_rans/*.cpp'),
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.12",
)
