#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
import warnings

from shutil import rmtree

import numpy as np
from setuptools import find_packages, setup, Command
from Cython.Distutils import build_ext
from distutils.extension import Extension



# Package meta-data.
NAME = 'fnms'
DESCRIPTION = 'Fast non-maximum suppression.'
URL = 'https://github.com/raikel/fnms'
EMAIL = 'raikelbl@gmail.com'
AUTHOR = 'Raikel Bordon'
REQUIRES_PYTHON = '>=3.4.0'
VERSION = '0.4.0'
README = 'README.rst'

# What packages are required for this module to be executed?
REQUIRED = [
    'numpy>=1.16.0',
    'Cython>=0.29'
]

# What packages are optional?
EXTRAS = {
}

# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, README), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        sys.exit()


def find_in_path(name, path):
    """Find a file in a search path"""
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    home = "/usr/local/cuda"
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


# CUDA = None

try:
    CUDA = locate_cuda()
except EnvironmentError as err:
    warnings.warn(str(err))


# Obtain the numpy include directory.  This logic works across numpy versions.
numpy_include = np.get_include()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kind of like a weird functional
    subclassing going on."""

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
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    ext_modules=ext_modules,
    cmdclass={
        'upload': UploadCommand,
        'build_ext': CustomBuildExt
    },
)