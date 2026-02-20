# Troubleshooting

## Build and Installation

### Cython extensions won't compile

**Symptom**: `python setup.py build_ext --inplace` fails.

**Common causes**:

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: numpy` | `pip install numpy` before building |
| `ModuleNotFoundError: Cython` | `pip install Cython` before building |
| `fatal error: 'omp.h' not found` (macOS) | `brew install libomp` |
| `-fopenmp: not found` (Linux) | Install gcc with OpenMP: `apt install gcc` |

VoxelCAD works without Cython - it falls back to NumPy. But renders will be 10-60x slower.

### `pip install -e .` doesn't compile Cython

Expected behavior. Editable installs don't trigger `build_ext`. Run it separately:

```bash
pip install -e .
python setup.py build_ext --inplace
```

### NumPy 2.0 deprecation warnings during build

Harmless. The build uses `NPY_NO_DEPRECATED_API` to suppress them. If you see them, the extensions still compiled correctly.

## Runtime Errors

### "Cython kernel unavailable, falling back to NumPy"

**RuntimeWarning** emitted when `ENV.use_cython=True` but the specific kernel function wasn't compiled. Usually means `build_ext` wasn't run or failed partially.

Fix: rebuild extensions.

```bash
python setup.py build_ext --inplace
```

### Out of memory / "Killed"

Bare "Killed" message (no Python traceback) means the OS terminated the process for memory exhaustion.

**Likely causes**:
- `voxel_size` too small for the model size
- Multiple large models in memory simultaneously
- Boolean operations on large non-matching grids (requires rendering both to union grid)

**Fix**: Increase `voxel_size`. A 1024^3 grid takes 128 MB packed; 2048^3 takes 1 GB.

### `plot()` shows nothing or crashes

VoxelCAD uses PyVista for visualization. Common issues:

| Problem | Fix |
|---------|-----|
| No display (headless server) | Use `model.export("out.stl")` instead |
| `ModuleNotFoundError: pyvista` | `pip install pyvista` |
| EGL/VTK warnings on Linux | Usually harmless; suppress with `export VTK_SILENCE_GET_VOID_POINTER_WARNINGS=1` |

### Boolean result looks wrong

Check that operands use the same `voxel_size`:

```python
a = Sphere(r=5, voxel_size=0.1)
b = Cube(size=8, voxel_size=0.2)   # different!
result = a & b                      # resampling may lose detail
```

When voxel sizes differ, the finer model is resampled to the coarser grid. Use matching sizes.

### Transform produces unexpected orientation

Transform order matters. Operations apply left-to-right:

```python
# Rotate THEN translate (rotate in place, then move):
model.rotate_z(45).translate([5, 0, 0])

# Translate THEN rotate (move, then orbit around origin):
model.translate([5, 0, 0]).rotate_z(45)
```

If the result isn't what you expected, reverse the operation order.

## STL Export

### Exported mesh has staircase artifacts

Voxel models are inherently blocky at low resolution. Decrease `voxel_size` for smoother surfaces:

```python
# Blocky
Sphere(r=5, voxel_size=0.5).export("rough.stl")

# Smooth
Sphere(r=5, voxel_size=0.05).export("smooth.stl")
```

### Export is slow

Mesh extraction uses marching cubes, which scales with grid volume. For a 1000^3 grid, expect a few seconds. The export itself (writing STL) is fast; it's the mesh generation that takes time.

### STL file is empty (0 bytes or no faces)

The model's bounding box might not contain any filled voxels. Check with:

```python
print(model.voxel_data.sum())  # should be > 0
model.plot()                   # visual check
```
