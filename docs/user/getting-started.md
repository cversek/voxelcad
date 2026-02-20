# Getting Started

## Installation

```bash
conda create -n voxelcad python=3.10
conda activate voxelcad
conda install -c conda-forge pyvista ipython tqdm cython numpy
pip install -e .
python setup.py build_ext --inplace
```

The last command compiles Cython extensions for 10-60x speedups. VoxelCAD works without them (pure NumPy fallback), but large models will be slow.

## Your First Model

```python
from voxelcad import Sphere

s = Sphere(r=5, voxel_size=0.1)
s.plot()
```

`r=5` sets the radius. `voxel_size=0.1` controls resolution: smaller values produce finer detail but use more memory. A good starting point is `voxel_size = size / 100`.

## Boolean Operations

Combine models with Python operators:

```python
from voxelcad import Sphere, Cube

s = Sphere(r=4, voxel_size=0.2)
c = Cube(size=6, voxel_size=0.2, center=True)

union        = s | c   # everything in either shape
intersection = s & c   # only where both shapes overlap
difference   = c - s   # cube with sphere carved out
xor          = s ^ c   # where exactly one shape exists
```

Both operands should use the same `voxel_size` for best performance.

## Transforms

Move, rotate, and scale models after construction:

```python
s = Sphere(r=3, voxel_size=0.2)

moved   = s.translate([0, 0, 5])     # shift along Z
rotated = s.rotate_z(45)             # degrees
scaled  = s.scale([2, 1, 0.5])       # per-axis scale factors
```

Transforms are lazy: they store a matrix and evaluate on demand. Chained transforms compose into a single matrix.

## Export to STL

```python
model = Sphere(r=5, voxel_size=0.1) & Cube(size=8, voxel_size=0.1, center=True)
model.export("output.stl")
```

The exported STL can be loaded into any slicer for 3D printing.

## Resolution and Memory

| voxel_size | Grid for r=5 sphere | Memory (packed) | Render time (Cython) |
|------------|---------------------|-----------------|---------------------|
| 0.5 | 20^3 | <1 KB | <1 ms |
| 0.1 | 100^3 | 122 KB | ~5 ms |
| 0.05 | 200^3 | 977 KB | ~30 ms |
| 0.01 | 1000^3 | 119 MB | ~1 s |

Start coarse (`voxel_size=0.5`) for rapid iteration. Increase resolution for final export.

## Next Steps

- [Geometry Catalog](geometry-catalog.md) -- all primitives and their parameters
- [Boolean Operations](boolean-operations.md) -- detailed guide with examples
- [Transforms](transforms.md) -- rotation, scaling, translation, composition
- [Performance Guide](performance-guide.md) -- memory management and Cython acceleration
