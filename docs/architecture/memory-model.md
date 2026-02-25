# Memory Model

How VoxelCAD manages memory through streaming evaluation and packed storage.

## The Problem: Vectorized Intermediates

NumPy's vectorized operations allocate full intermediate arrays. For a 1024^3 sphere evaluation:

```python
# Each line allocates a full-volume array
dx = X - cx           # 8 GB float64
dy = Y - cy           # 8 GB float64
dz = Z - cz           # 8 GB float64
dist = dx**2 + dy**2 + dz**2  # 8 GB float64
V = dist <= r**2      # 1 GB bool
packed = np.packbits(V.ravel(order='F'))  # 128 MB uint8
```

Peak memory: ~25 GB for temporaries that exist only to produce 128 MB of output. At typical container limits (~8 GB), this triggers the OOM killer.

## Solution: Fused Evaluate-and-Pack

Cython kernels evaluate geometry, threshold, and bit-pack in a single pass. The intermediate bool array never exists.

**Pattern** (`_fused_parallel.pyx`):

```cython
packed = np.zeros(total_bytes, dtype=np.uint8)  # Only allocation: 128 MB
cdef unsigned char *out = &packed_view[0]

for k in prange(rz, nogil=True, num_threads=actual_threads):
    for j in range(ry):
        for i in range(rx):
            # Evaluate geometry (few arithmetic ops)
            if geometry_test(x, y, z):
                # Pack directly into output byte
                set_bit(out, i + j * rx + k * slice_bits)
```

Peak memory: the output array only (~128 MB at 1024^3). No temporaries.

## NumPy Fallback: Slice Streaming

When Cython is unavailable, the NumPy path reduces memory by iterating Z-slices instead of allocating the full 3D volume:

```python
V = np.zeros((rx, ry, rz), dtype='bool')  # 1 GB at 1024^3
for X_2d, Y_2d, z_val, k in grid.iter_slices():
    V[:, :, k] = geometry_test(X_2d, Y_2d, z_val)
return np.packbits(V.ravel(order='F'), bitorder='big')
```

`iter_slices()` (`voxel_grid.py:98-111`) pre-allocates 2D meshgrids (`X_2d`, `Y_2d`) once and reuses them for each Z level. Memory: O(rx * ry) for the meshgrid, plus the full output bool array.

This is worse than Cython (1 GB bool still allocated) but better than full 3D vectorization (no 8 GB float64 intermediates).

## Separable Precomputation

Gyroid kernels exploit axis-separable trig to eliminate per-voxel trig calls (`_fused_parallel.pyx:386-421`):

```cython
# Before prange: O(rx + ry) trig evaluations
for i in range(rx):
    cos_x_arr[i] = cos(xcc[i] * ax + phi_x)
    sin_x_arr[i] = sin(xcc[i] * ax + phi_x)
for j in range(ry):
    cos_y_arr[j] = cos(ycc[j] * ay + phi_y)
    sin_y_arr[j] = sin(ycc[j] * ay + phi_y)

# Inside prange: 3 multiplies + 2 adds per voxel, zero trig
for k in prange(rz, ...):
    cos_z = cos(zcc[k] * az + phi_z)  # 1 trig per Z-slice
    for j in range(ry):
        for i in range(rx):
            F = cos_x[i]*sin_y[j] + cos_y[j]*sin_z + cos_z*sin_x[i]
```

At 1024^3, this replaces 6 billion trig calls with ~2048 trig calls plus 3 billion multiply-adds. Memory cost: two arrays of size `rx` and two of size `ry` (~16 KB total at 1024).

This optimization only works when the transform is identity (axis-aligned). With arbitrary M4inv, coordinates could be coupled, so the full trig evaluation runs per-voxel.

## OpenMP Parallelization

All Cython kernels parallelize over the Z-axis:

```cython
for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
```

**Thread safety**: each Z-slice writes to a disjoint byte range in the output array (because F-order makes Z-slices contiguous). No locks or atomics needed.

**Thread count**: auto-detected via `_get_optimal_threads()`. On Apple Silicon, this targets P-cores only (E-cores cause cross-cluster cache coherency overhead).

**Edge case**: when `slice_bits % 8 != 0`, slices share byte boundaries. The kernel falls back to single-threaded to avoid data races.

## Early-Exit Optimizations

Without transforms, geometry kernels skip entire slices or rows:

```cython
# Sphere: skip Z-slices outside radius
dz_sq = (zcc[k] - cz) * (zcc[k] - cz)
if dz_sq > r_sq:
    continue

# Cube: skip Y-rows outside bounds
if fabs(ycc[j] - cy) > half_sy:
    continue
```

For a sphere occupying 10% of its bounding box, this skips ~90% of iterations. With M4inv, axis-aligned pruning isn't possible — the transform couples all axes.

## Scaling Laws

| Resolution | Packed Output | Cython Peak | NumPy Peak | Vectorized Peak |
|------------|--------------|-------------|------------|-----------------|
| 64^3 | 32 KB | 32 KB | 262 KB | ~2 MB |
| 256^3 | 2 MB | 2 MB | 16 MB | ~130 MB |
| 512^3 | 16 MB | 16 MB | 128 MB | ~1 GB |
| 1024^3 | 128 MB | 128 MB | 1 GB | ~8 GB |

Cython peak equals the output size — no overhead. NumPy peak includes the bool volume. Vectorized peak includes float64 intermediates (distance arrays, etc.).

## Profiling Integration

Rendering paths are instrumented with `TIMING_START`/`TIMING_END` from `voxelcad.debug`:

```python
TIMING_START("sphere_render_cython")
result = evaluate_and_pack_sphere(...)
TIMING_END("sphere_render_cython")
```

These are lightweight wrappers that record wall-clock time when enabled. See the performance guide for usage.
