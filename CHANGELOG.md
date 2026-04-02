# Changelog

All notable changes to VoxelCAD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-02

First public release of VoxelCAD — a Python 3D CAD library for voxel-based
solid modeling with numpy and Cython streaming kernels.

### Added

- **Geometry primitives**: Cube, Sphere, Cylinder, GyroidCube (with lattice parameters), WigglyGyroidCube, HyperWigglyGyroidCube
- **Boolean operations**: union (`|`), intersection (`&`), difference (`-`), XOR (`^`), complement (`~`) with three-tier dispatch:
  - Tier 1: byte-level bitwise ops for same-grid operands (~0.3 ms)
  - Tier 2: render-on-grid for compatible grids
  - Tier 3: CSG (Constructive Solid Geometry) tree with query-planned execution for arbitrary combinations
- **Affine transforms**: translate, rotate (x/y/z), scale with 4x4 matrix composition and lazy evaluation
- **Packed bit storage**: 8x memory reduction via `np.packbits` with F-order layout for cache-friendly Z-slice access
- **Cython streaming kernels**: fused evaluate-and-pack with OpenMP parallelism (60x speedup over NumPy vectorized at 1024^3)
- **STL export pipeline**: SDF (Signed Distance Field) + FFT (Fast Fourier Transform) Butterworth low-pass + Lewiner MC (Marching Cubes), eliminating meshfix entirely
  - Fused streaming kernel: packed bits to binary STL in a single pass (~10 ms at 128^3)
  - Fused mesh export: packed bits to PyVista PolyData with pthread convolution/MC overlap
  - CDT (Chamfer Distance Transform) precision path for distance field visualization
  - Configurable smoothing: `lowpass_cutoff`, `lowpass_order`, `mc_stride`
  - `only_largest_component` option for clean single-body output (LCC — Largest Connected Component extraction)
- **Surface mesh extraction**: `render_surface_mesh()` with CDT and fast_smooth methods
- **CDT volume exposure**: `render_cdt_grid()` for distance field visualization and analysis
- **Environment configuration**: `voxelcad.environment` module for `voxel_size`, `use_cython`, `log_level` settings
- **Benchmark suite**: 14 benchmarks covering render, boolean ops, transforms, and STL export with super_utils BenchmarkBase integration
- **Test suite**: 100 tests covering primitives, transforms, CSG, export winding, endian regression, edge cases, and fallback paths
- **Documentation**: architecture docs, getting-started guide, geometry catalog, boolean operations guide, transforms guide, performance guide, troubleshooting, and CONTRIBUTING.md
- **Examples**: 8 runnable examples from hello_world to gyroid electrode support plugs

### Performance (Linux aarch64, Cython enabled, 128^3 grid)

| Operation | Time | Memory |
|-----------|------|--------|
| Sphere render | 23 ms | <1 MB |
| Same-grid boolean | 0.3 ms | 2 MB |
| CSG 4-primitive tree | 89 ms | 11 MB |
| Sphere STL export | 12 ms | <1 MB |
| Gyroid+Sphere STL export | 18 ms | <1 MB |
