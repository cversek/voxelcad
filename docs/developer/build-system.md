# Build System

## Overview

VoxelCAD uses a split build configuration:

- `pyproject.toml` — package metadata, dependencies, version
- `setup.py` — Cython extension compilation only
- `Makefile` — developer workflow targets

## pyproject.toml

Standard PEP 621 metadata with setuptools backend:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "voxelcad"
dynamic = ["version"]
```

**Version** is single-sourced: defined in `src/voxelcad/__init__.py` as `__version__`, read by setuptools via `[tool.setuptools.dynamic]`.

**Dependencies**:

| Group | Contents | Install |
|-------|----------|---------|
| core | numpy, matplotlib | `pip install -e .` |
| viz | pyvista, vtk, pymeshfix | `pip install -e ".[viz]"` |
| dev | pytest, cython, pyvista | `pip install -e ".[dev]"` |

## setup.py

Exists solely for Cython extension building. It's guarded to only build when explicitly requested:

```python
ext_modules = []
if 'build_ext' in sys.argv:
    ext_modules = get_extensions()
setup(ext_modules=ext_modules)
```

This prevents `pip install -e .` from failing when numpy or Cython aren't installed yet (build isolation).

### OpenMP Detection

`get_openmp_flags()` detects the platform and returns appropriate compiler/linker flags:

| Platform | Compiler | OpenMP flags | Requirement |
|----------|----------|-------------|-------------|
| macOS | Apple clang | `-Xpreprocessor -fopenmp` | `brew install libomp` |
| Linux | gcc | `-fopenmp` | gcc with OpenMP (usually default) |

If OpenMP isn't available, kernels compile and run single-threaded.

### Cython Extensions

One extension module: `voxelcad._kernels._fused_parallel`

Source: `src/voxelcad/_kernels/_fused_parallel.pyx`

Compiler flags: `-O3` plus platform OpenMP flags.

The `_kernels/__init__.py` imports each kernel function with a try/except, setting unavailable functions to `None`. Primitive classes check for `None` before calling the Cython path.

### Missing Dependencies During Build

If numpy or Cython isn't installed when you run `python setup.py build_ext --inplace`:

- **Build time**: prints `WARNING: No module named 'Cython'. Skipping Cython extension build.` to stdout and exits successfully with no extensions compiled. This is a plain `print()`, not a Python warning — easy to miss in piped output.
- **Runtime**: the first time a primitive renders, it emits a `RuntimeWarning` (e.g., `"Sphere: Cython kernel unavailable, falling back to NumPy"`) and uses the 10-60x slower NumPy path.

To avoid surprises, use `make install` which installs `[dev]` dependencies (including Cython) before building.

## Makefile

```bash
make help         # list all targets
make              # pull + install + build (default)
make build        # compile Cython extensions
make install      # editable install with [dev] deps
make test         # pytest tests/
make benchmark    # pytest benchmarks/
make docs-images  # regenerate doc example PNGs
make clean        # remove build artifacts and .so files
```

## Adding a New Cython Kernel

1. Add your function to `_fused_parallel.pyx`
2. Add an import line to `_kernels/__init__.py` with a try/except fallback to `None`
3. Run `make build` to compile
4. Reference the function in your primitive's `_render_cython()` method

The kernel should accept `M4inv=None` as a keyword argument for transform support. Use `long long` (not `int`) for index computations — grids above 1024^3 overflow 32-bit indices.
