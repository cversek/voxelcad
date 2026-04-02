""" This module defines shared defaults that can be overrriden by the user.
"""
import os
import warnings


def _get_log_level():
    """Resolve log level: env var > pyproject.toml > default WARNING.

    Priority:
        1. VOXELCAD_LOG_LEVEL environment variable (e.g. "DEBUG", "INFO")
        2. [tool.voxelcad].log_level in pyproject.toml (for dev branches)
        3. "WARNING" (release default — silent)
    """
    env_val = os.environ.get('VOXELCAD_LOG_LEVEL')
    if env_val:
        return env_val.upper()
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return 'WARNING'
    try:
        import pathlib
        pyproject = pathlib.Path(__file__).parent.parent.parent / 'pyproject.toml'
        if pyproject.exists():
            with open(pyproject, 'rb') as f:
                cfg = tomllib.load(f)
            return cfg.get('tool', {}).get('voxelcad', {}).get(
                'log_level', 'WARNING').upper()
    except Exception:
        pass
    return 'WARNING'


log_level = _get_log_level()

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