#!/usr/bin/env python
import io
import os
import sys
import warnings

from shutil import rmtree

import numpy as np
from setuptools import find_packages, setup, Command
from Cython.Distutils import build_ext
from distutils.extension import Extension


def locate_cuda(home):
    cuda_config = {
        'home': home,
        'nvcc': os.path.join(home, 'bin', 'nvcc'),
        'include': os.path.join(home, 'include'),
        'lib64': os.path.join(home, 'lib64')
    }
    for k, v in cuda_config.items():
        if not os.path.exists(v):
            raise EnvironmentError(
                f'The CUDA {k} path could not be located in {v}'
            )

    return cuda_config


CUDA = None
try:
    CUDA = locate_cuda("/usr/local/cuda")
except EnvironmentError as err:
    warnings.warn(str(err))


# Obtain the numpy include directory.  This logic works across numpy versions.
numpy_include = np.get_include()


def customize_compiler_for_nvcc(self):
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    _super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if CUDA is not None and os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        _super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class CustomBuildExt(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        self.cython_directives['language_level'] = 3
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "fnms.cpu_nms",
        ["fnms/cpu_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    )
]

if CUDA is not None:
    ext_modules.append(Extension(
        'fnms.gpu_nms',
        ['fnms/nms_kernel.cu', 'fnms/gpu_nms.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        # this syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc and not with gcc
        # the implementation of this trick is in customize_compiler() below
        extra_compile_args={
            'gcc': ["-Wno-unused-function"],
            'nvcc': [
                '-arch=sm_52',
                '--ptxas-options=-v',
                '-c',
                '--compiler-options',
                "'-fPIC'"
            ]
        },
        include_dirs=[numpy_include, CUDA['include']]
    ))

# Where the magic happens:
setup(
    name='fnms',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': CustomBuildExt
    },
)