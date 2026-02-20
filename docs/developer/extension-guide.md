# Extension Guide

How to add a new primitive to VoxelCAD.

## Overview

Every primitive (Sphere, Cube, Cylinder, etc.) is a subclass of `VoxelModel` that defines its geometry through two rendering methods: a fast Cython path and a NumPy fallback. The system dispatches automatically based on what's compiled.

## File Structure

```
src/voxelcad/
    my_shape.py                 # Your primitive class
    _kernels/
        _fused_parallel.pyx     # Optional: Cython kernel
        __init__.py             # Kernel import with graceful fallback
    __init__.py                 # Export your class here
```

## Step 1: Subclass VoxelModel

Create `src/voxelcad/torus.py`. The `__init__` must:

1. Call `super().__init__(**kwargs)`
2. Create `self.grid` with the bounding box
3. Store geometry parameters as instance attributes

```python
import numpy as np
from voxelcad.voxel_model import VoxelModel
from voxelcad.voxel_grid import VoxelGrid

class Torus(VoxelModel):
    def __init__(self, R, r, voxel_size=None, **kwargs):
        """
        Parameters:
            R: Major radius (center of tube to center of torus)
            r: Minor radius (radius of the tube)
        """
        super().__init__(**kwargs)
        self.R = R
        self.r = r
        extent = R + r
        self.grid = VoxelGrid(
            xlim=(-extent, extent),
            ylim=(-extent, extent),
            zlim=(-r, r),
            voxel_size=voxel_size,
        )
```

## Step 2: Implement _render_numpy

The NumPy path is required. It iterates Z-slices and evaluates your geometry condition:

```python
def _render_numpy(self, grid, M4inv=None):
    from voxelcad.debug import TIMING_START, TIMING_END
    TIMING_START("torus_render_numpy")

    cx, cy, cz = self.grid.compute_center_vector()
    rx, ry, rz = [int(r) for r in grid.res_vector]
    V = np.zeros((rx, ry, rz), dtype='bool')

    for X_2d, Y_2d, z_val, k in grid.iter_slices():
        if M4inv is not None:
            Z_2d = np.full_like(X_2d, z_val)
            Xp = M4inv[0,0]*X_2d + M4inv[0,1]*Y_2d + M4inv[0,2]*Z_2d + M4inv[0,3]
            Yp = M4inv[1,0]*X_2d + M4inv[1,1]*Y_2d + M4inv[1,2]*Z_2d + M4inv[1,3]
            Zp = M4inv[2,0]*X_2d + M4inv[2,1]*Y_2d + M4inv[2,2]*Z_2d + M4inv[2,3]
        else:
            Xp, Yp, Zp = X_2d, Y_2d, z_val

        # Torus equation: (sqrt(x^2 + y^2) - R)^2 + z^2 <= r^2
        dist_xy = np.sqrt((Xp - cx)**2 + (Yp - cy)**2)
        V[:, :, k] = (dist_xy - self.R)**2 + (Zp - cz)**2 <= self.r**2

    TIMING_END("torus_render_numpy")
    return np.packbits(V.ravel(order='F'), bitorder='big')
```

Key rules:

- Accept `grid` and `M4inv` parameters (for transform support)
- Apply `M4inv` when not None (inverse transform matrix)
- Pack output with `np.packbits(..., order='F', bitorder='big')`
- Wrap with `TIMING_START`/`TIMING_END` for profiling

## Step 3: Implement _render_cython (Optional)

For 10-60x speedup, add a Cython kernel in `_kernels/_fused_parallel.pyx`. See existing kernels (`evaluate_and_pack_sphere`, `evaluate_and_pack_cube`) for the pattern.

The Python-side dispatch:

```python
def _render_cython(self, grid, M4inv=None):
    from voxelcad._kernels import evaluate_and_pack_torus
    if evaluate_and_pack_torus is None:
        if ENV.use_cython:
            import warnings
            warnings.warn(
                "Torus: Cython kernel unavailable, falling back to NumPy",
                RuntimeWarning, stacklevel=3,
            )
        return self._render_numpy(grid, M4inv)

    xcc, ycc, zcc = grid.compute_cell_center_ranges()
    cx, cy, cz = self.grid.compute_center_vector()
    return evaluate_and_pack_torus(
        xcc, ycc, zcc, cx, cy, cz,
        self.R, self.r,
        M4inv=M4inv,
    )
```

## Step 4: Register and Export

Add your class to `src/voxelcad/__init__.py`:

```python
from voxelcad.torus import Torus
```

The doc image pipeline and test fixtures will discover it automatically via `inspect`.

## Step 5: Add Tests

Create `tests/test_torus.py`:

```python
import numpy as np
import pytest
from voxelcad.torus import Torus

LOW_RES_VS = 10.0 / 32

@pytest.fixture
def torus32():
    return Torus(R=4, r=1.5, voxel_size=LOW_RES_VS)

def test_torus_render(torus32):
    data = torus32.render_volume()
    assert data.dtype == np.uint8
    assert data.sum() > 0

def test_torus_transform(torus32):
    moved = torus32.translate([0, 0, 5])
    moved.render_volume()
    assert moved.voxel_data is not None

def test_torus_boolean(torus32):
    cube = Cube(size=6, voxel_size=LOW_RES_VS, center=True)
    result = torus32 & cube
    result.render_volume()
    assert result.voxel_data.sum() > 0
```

Run with `make test` or `python -m pytest tests/test_torus.py -v`.

## Checklist

- [ ] Subclass `VoxelModel`, set `self.grid` in `__init__`
- [ ] Implement `_render_numpy()` with F-order packbits
- [ ] Handle `M4inv` parameter for transform support
- [ ] Add `TIMING_START`/`TIMING_END` instrumentation
- [ ] Optional: Cython kernel with fallback guard
- [ ] Export from `__init__.py`
- [ ] Tests: render, transform, boolean ops
- [ ] Add to `docs/user/geometry-catalog.md` with renderable example
