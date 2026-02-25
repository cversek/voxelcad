# GPU Backend Design

Design notes for a future GPU-accelerated rendering backend.

## Current State

All rendering runs on CPU via Cython+OpenMP or NumPy. The fused evaluate-and-pack kernels are memory-efficient (~128 MB at 1024^3) and parallel (5-8x speedup on Apple Silicon P-cores), but remain compute-bound for complex geometry (gyroid variants) at high resolution.

## Motivation

At 1024^3 with transforms, the Cython parallel path takes 80-400ms per primitive. A GPU backend could reduce this to single-digit milliseconds by exploiting massive parallelism (thousands of cores vs 12 P-cores).

## Approach 1: OpenMP Target Offloading

The most natural path: the existing Cython kernels already use OpenMP `prange` for Z-slice parallelism. OpenMP 4.0+ introduced `target` directives that can offload parallel regions to GPU.

**Current CPU code**:
```cython
for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
    z = zcc[k]
    for j in range(ry):
        for i in range(rx):
            if geometry_test(x, y, z):
                set_bit(out, lin_idx)
```

**With OpenMP target offloading** (conceptual):
```c
#pragma omp target teams distribute parallel for map(to: xcc, ycc, zcc, ...) map(from: out)
for (k = 0; k < rz; k++) {
    for (j = 0; j < ry; j++) {
        for (i = 0; i < rx; i++) {
            if (geometry_test(x, y, z))
                set_bit(out, lin_idx);
        }
    }
}
```

**Advantages**:
- Same source code for CPU and GPU — no separate kernel rewrite
- Z-slice parallelism maps naturally to GPU thread blocks (one block per Z-slice, threads within a block handle voxels in that slice)
- `map(to:)` / `map(from:)` handles host-device data transfer
- Compiler chooses optimal thread mapping

**Challenges**:
- Cython doesn't emit `#pragma omp target` — would need to either (a) write the hot loops in raw C called from Cython, or (b) use Cython's `cdef extern` to wrap a C file containing the target directives
- Compiler support varies: `clang` (Apple) has limited offloading support; `gcc` targets NVIDIA GPUs via `nvptx`; `nvc` (NVIDIA HPC SDK) has the most mature support
- `set_bit` atomicity: Z-slice parallelism keeps byte writes disjoint on CPU. On GPU, if thread mapping changes to per-voxel, atomic OR becomes necessary for shared bytes
- Apple Silicon: Metal is not an OpenMP target. MLX or direct Metal would be needed for macOS GPU

**Practical path**: Extract the inner loops from `.pyx` into `.c` files with OpenMP target pragmas. Cython wraps these via `cdef extern`. The C code compiles with or without target support — `#ifdef _OPENMP` guards keep CPU fallback clean.

## Approach 2: Library-Based GPU Backends

Replace the inner loop with a GPU array library call.

| Backend | Platform | Language | Array Library |
|---------|----------|----------|---------------|
| CuPy | NVIDIA CUDA | Python/CUDA | CuPy (NumPy-compatible) |
| MLX | Apple Silicon | Python/Metal | MLX (NumPy-like) |
| Taichi | Cross-platform | Python DSL | Taichi fields |

These would implement `_render_gpu()` as an alternative dispatch path:

```python
def render_on_grid(self, grid, M4inv=None):
    if M4inv is None and self.voxel_data is not None and self.grid.same_grid(grid):
        return self.voxel_data
    if ENV.use_gpu and self._has_gpu_kernel():
        return self._render_gpu(grid, M4inv)
    if ENV.use_cython:
        return self._render_cython(grid, M4inv)
    return self._render_numpy(grid, M4inv)
```

**Advantages**: native GPU performance, platform-optimized
**Disadvantages**: separate kernel code per backend, additional dependencies, maintenance burden

## Kernel Translation Considerations

**Parallelism granularity**: CPU parallelizes over Z-slices (~1024 units); GPU can parallelize over individual voxels (~10^9 units). The Z-slice structure still works on GPU (one thread block per slice) but underutilizes hardware. Per-voxel dispatch maximizes occupancy.

**Bit-packing atomicity**: On CPU, F-order guarantees each Z-slice writes disjoint bytes — no synchronization needed. On GPU with per-voxel parallelism, multiple threads may target the same byte. Use `atomicOr` (CUDA) or equivalent. Z-slice-level GPU parallelism avoids this but limits thread count.

**Memory transfer**: GPU results must be copied to host RAM for mesh generation and export. For CSG trees with many leaves, keeping packed arrays on GPU across operations avoids repeated transfers.

**Separable trig**: The gyroid's per-axis precomputation maps well to GPU shared memory. Pre-compute `cos_x[]`, `sin_x[]` in shared memory, then compose in registers.

## Design Constraints

1. **Packed output format must match**: GPU kernels must produce F-order packed uint8 with MSB-first bit order, identical to Cython output. All downstream consumers depend on this format.

2. **M4inv support required**: The 4x4 inverse transform matrix is a small constant buffer — straightforward on any GPU backend.

3. **Fallback chain**: GPU failure must fall through to Cython, then NumPy. No silent corruption.

4. **ENV.use_gpu flag**: User-controllable, defaults to auto-detect (GPU available → True).

## Recommended Path

Start with OpenMP target offloading for NVIDIA GPUs (via `gcc` or `nvc`). This preserves the existing code structure and adds GPU support incrementally. For Apple Silicon, evaluate MLX as a platform-specific backend if Metal performance justifies the maintenance cost.

## Open Questions

- **Crossover resolution**: Below what grid size is GPU overhead (kernel launch, memory transfer) slower than CPU? Likely around 128^3-256^3.
- **Boolean combination on GPU**: Byte-level NumPy is ~10ms at 1024^3 on CPU. Worth offloading, or only geometry evaluation?
- **Cython-to-C extraction**: How much of `_fused_parallel.pyx` can be cleanly extracted to `.c` files while keeping the Python interface?
