""" This module defines shared defaults that can be overrriden by the user.
"""
import warnings

voxel_size = 0.01 # 10 microns

# Auto-detect Cython availability; user can override to False for testing
try:
    from voxelcad._kernels import CYTHON_AVAILABLE as _cython_compiled
except ImportError:
    _cython_compiled = False

if _cython_compiled:
    use_cython = True
else:
    use_cython = False
    warnings.warn(
        "VoxelCAD Cython kernels not compiled. "
        "Performance will be degraded. "
        "Run 'python setup.py build_ext --inplace' to compile.",
        stacklevel=1,
    )

# Auto-detect tqdm for progress bars in mesh smoothing etc.
try:
    import tqdm as _tqdm  # noqa: F401
    progress_bar = True
except ImportError:
    progress_bar = False