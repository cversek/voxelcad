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
sweep_mc_mesh = None
fused_stl_export = None
fused_mesh_export = None

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
        sweep_mc_mesh,
        fused_stl_export,
        fused_mesh_export,
        _detect_p_cores,
        _get_optimal_threads,
    )
    CYTHON_AVAILABLE = True
except ImportError:
    pass

def compute_cdt_field(packed, rx, ry, rz, stride=1, voxel_size=None):
    """Compute CDT distance field as float32 in real units (mm).

    Returns a float32 3D array of signed distances from the surface,
    scaled from chessboard voxel units to real-world metric units.
    Positive = inside, negative = outside.

    Args:
        packed: F-order packed binary volume (uint8).
        rx, ry, rz: Full-resolution grid dimensions.
        stride: Subsample factor (default 1).
        voxel_size: Scalar or array-like [vsx, vsy, vsz] for unit scaling.
            If None, returns distances in voxel units (no scaling).
            For isotropic grids, pass the scalar voxel_size.
            For anisotropic grids, distances are scaled by min(vsx, vsy, vsz)
            since chessboard metric uses unit cost per step.

    Returns:
        np.ndarray[float32]: Signed distance field, shape
        (ceil(rx/stride)+2, ceil(ry/stride)+2, ceil(rz/stride)+2).
    """
    import numpy as np
    if compute_sdf_cdt is None:
        raise RuntimeError("Cython extension not available for CDT computation")
    sdf_int8 = compute_sdf_cdt(packed, rx, ry, rz, stride)
    sdf_f32 = sdf_int8.astype(np.float32)
    if voxel_size is not None:
        vs = np.asarray(voxel_size, dtype=np.float64).ravel()
        if vs.size == 1:
            scale = float(vs[0]) * stride
        else:
            scale = float(np.min(vs)) * stride
        sdf_f32 *= scale
    return sdf_f32


__all__ = [
    'CYTHON_AVAILABLE',
    'compute_cdt_field',
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
    'sweep_mc_mesh',
    'fused_stl_export',
    'fused_mesh_export',
]
