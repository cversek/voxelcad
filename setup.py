"""
Build configuration for VoxelCAD Cython extensions.

Usage:
    python setup.py build_ext --inplace

The Cython extensions are optional — VoxelCAD works without them via
NumPy evaluate_slice() fallback. Extensions provide 50-100x speedup
for geometry evaluation via fused evaluate+threshold+pack kernels.

(working with Craig Wm. Versek <cversek@gmail.com>)
"""

import os
import sys
import platform
import subprocess

from setuptools import setup, Extension


def get_openmp_flags():
    """Get platform-specific OpenMP compile/link flags."""
    system = platform.system()

    if system == 'Darwin':
        # macOS: Apple clang needs -Xpreprocessor and libomp from Homebrew
        try:
            prefix = subprocess.check_output(
                ['brew', '--prefix', 'libomp']
            ).decode().strip()
            return {
                'compile': ['-Xpreprocessor', '-fopenmp'],
                'link': ['-lomp'],
                'include': [prefix + '/include'],
                'library': [prefix + '/lib'],
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("WARNING: libomp not found. Install with: brew install libomp")
            print("         Building without OpenMP (kernels will be single-threaded)")
            return {'compile': [], 'link': [], 'include': [], 'library': []}
    else:
        # Linux (gcc)
        return {
            'compile': ['-fopenmp'],
            'link': ['-lgomp'],
            'include': [],
            'library': [],
        }


def get_extensions():
    """Build list of Cython extensions. Returns [] if deps missing."""
    try:
        import numpy as np
        from Cython.Build import cythonize
    except ImportError as e:
        print(f"WARNING: {e}. Skipping Cython extension build.")
        return []

    omp = get_openmp_flags()

    # Suppress NumPy 2.0 deprecation warnings
    numpy_api_macro = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

    extensions = [
        Extension(
            "voxelcad._kernels._fused_parallel",
            ["src/voxelcad/_kernels/_fused_parallel.pyx"],
            include_dirs=[np.get_include()] + omp['include'],
            library_dirs=omp['library'],
            extra_compile_args=['-O3'] + omp['compile'],
            extra_link_args=omp['link'],
            define_macros=numpy_api_macro,
        ),
    ]

    return cythonize(
        extensions,
        compiler_directives={'language_level': "3"},
    )


# Only build extensions when explicitly requested (build_ext),
# not during pip install metadata discovery
ext_modules = []
if 'build_ext' in sys.argv:
    ext_modules = get_extensions()

setup(ext_modules=ext_modules)
