"""
VoxelCAD fused evaluate-and-pack kernels.

Provides Cython+OpenMP accelerated geometry evaluation with direct bit-packing.
Falls back to NumPy evaluate_slice() path when Cython extensions are unavailable.

Usage:
    from voxelcad._kernels import CYTHON_AVAILABLE
    from voxelcad._kernels import evaluate_and_pack_gyroid  # None if unavailable
"""

import os

CYTHON_AVAILABLE = False

# Kernel function references (None when unavailable)
evaluate_and_pack_cube = None
evaluate_and_pack_sphere = None
evaluate_and_pack_cylinder = None
evaluate_and_pack_gyroid = None
evaluate_and_pack_wiggly_gyroid = None
evaluate_and_pack_hyperwiggly_gyroid = None
resample_and_pack = None
compute_sdf_cdt = None
convolve_sdf_spatial = None
fused_scale_convolve = None
streaming_mc_mesh = None
streaming_mc_stl = None

try:
    from ._fused_parallel import (
        evaluate_and_pack_cube,
        evaluate_and_pack_sphere,
        evaluate_and_pack_cylinder,
        evaluate_and_pack_gyroid,
        evaluate_and_pack_wiggly_gyroid,
        evaluate_and_pack_hyperwiggly_gyroid,
        resample_and_pack,
        compute_sdf_cdt,
        convolve_sdf_spatial,
        fused_scale_convolve,
        streaming_mc_mesh,
        streaming_mc_stl,
        _detect_p_cores,
        _get_optimal_threads,
    )
    CYTHON_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    'CYTHON_AVAILABLE',
    'evaluate_and_pack_cube',
    'evaluate_and_pack_sphere',
    'evaluate_and_pack_cylinder',
    'evaluate_and_pack_gyroid',
    'evaluate_and_pack_wiggly_gyroid',
    'evaluate_and_pack_hyperwiggly_gyroid',
    'resample_and_pack',
    'compute_sdf_cdt',
    'convolve_sdf_spatial',
    'fused_scale_convolve',
    'streaming_mc_mesh',
    'streaming_mc_stl',
]
