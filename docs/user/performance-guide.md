# Performance Guide

## Resolution Selection

`voxel_size` is the single biggest performance lever. Halving it multiplies memory by 8x and render time by ~8x.

| voxel_size | Grid (r=5 sphere) | Packed memory | Cython render |
|------------|-------------------|---------------|---------------|
| 1.0 | 10^3 | <1 KB | <1 ms |
| 0.5 | 20^3 | <1 KB | <1 ms |
| 0.1 | 100^3 | 122 KB | ~5 ms |
| 0.05 | 200^3 | 977 KB | ~30 ms |
| 0.01 | 1000^3 | 119 MB | ~1 s |
| 0.005 | 2000^3 | 954 MB | ~10 s |

The visual difference between resolutions:

```python
from voxelcad import Sphere

coarse = Sphere(r=5, voxel_size=1.0)
medium = Sphere(r=5, voxel_size=0.5)
fine = Sphere(r=5, voxel_size=0.1)
```

| voxel_size=1.0 | voxel_size=0.5 | voxel_size=0.1 |
|:--------------:|:--------------:|:--------------:|
| ![Coarse](_images/performance-guide/resolution-selection_0_coarse.png) | ![Medium](_images/performance-guide/resolution-selection_0_medium.png) | ![Fine](_images/performance-guide/resolution-selection_0_fine.png) |

Start coarse (`voxel_size=0.5`) for iteration. Increase for final export.

## Cython Acceleration

VoxelCAD includes Cython kernels that provide 10-60x speedups over the NumPy fallback. Build them after installation:

```bash
python setup.py build_ext --inplace
```

Check if they're active:

```python
from voxelcad.environment import ENV
print(ENV.use_cython)  # True if compiled, False if fallback
```

If Cython isn't compiled, VoxelCAD still works - it falls back to NumPy with a warning on first use.

### Force NumPy Fallback

For debugging or comparison:

```python
from voxelcad.environment import ENV
ENV.use_cython = False
```

## Memory: Packed Storage

VoxelCAD stores voxel data as packed booleans (1 bit per voxel, 8 voxels per byte). This means:

- 1024^3 grid = 128 MB (not 1 GB as a bool array)
- Boolean operations on packed data use byte-level bitwise ops
- No unpacking needed for most operations

## Boolean Operation Speed

Performance depends on grid compatibility:

| Scenario | What happens | Relative speed |
|----------|-------------|----------------|
| Same grid (same voxel_size + bounds) | Byte-level `bitwise_or/and/xor` | ~1 ms |
| Compatible grid (same voxel_size) | Render to union grid, then bitwise | ~10-100 ms |
| Different voxel_size | Full resampling + bitwise | ~100 ms - 1 s |

Keep operands at the same `voxel_size` to hit the fast path.

## Transform Performance

Transforms are lazy - they store a matrix, not a rendered result. The cost comes at render time:

- Primitives with Cython: geometry is evaluated directly in transformed coordinates (fast)
- Data-only models: nearest-neighbor resampling via Cython `resample_and_pack` kernel

Chaining multiple transforms has no extra cost. Ten chained transforms compose into one matrix multiplication.

## Large Model Tips

**Iterate at low resolution, export at high resolution:**

```python
model = complex_csg_tree(voxel_size=0.5)  # fast preview
model.plot()

# Happy with the shape? Re-create at export resolution:
model = complex_csg_tree(voxel_size=0.05)
model.export("output.stl")
```

**Monitor memory for large grids:**

At 0.005 voxel_size with a large bounding box, a single model can consume 1+ GB. Boolean operations on two such models need memory for both inputs plus the result.

**Use same-grid operands when possible:**

If you're combining many shapes, construct them all with the same `voxel_size` and overlapping bounding boxes. The same-grid fast path avoids resampling entirely.

## OpenMP Parallelism

Cython kernels use OpenMP for parallel evaluation. Thread count defaults to the number of CPU cores. On Apple Silicon, VoxelCAD uses only performance cores (P-cores) for better throughput.

To check thread usage, look for the parallel kernel dispatch in verbose output when building:

```bash
python setup.py build_ext --inplace 2>&1 | grep -i openmp
```

If OpenMP isn't available, kernels run single-threaded - still much faster than NumPy.
