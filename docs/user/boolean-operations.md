# Boolean Operations

Combine models using Python operators. Both operands must be `VoxelModel` instances (primitives, CSG results, or transformed models).

## Operators

| Operator | Name | Result |
|----------|------|--------|
| `a \| b` | Union | Everything in either shape |
| `a & b` | Intersection | Only where both overlap |
| `a - b` | Difference | `a` with `b` carved out |
| `a ^ b` | XOR | Where exactly one shape exists |
| `~a` | Invert | Flip all voxels in the bounding box |

## Basic Usage

```python
from voxelcad import Sphere, Cube

s = Sphere(r=4, voxel_size=0.2)
c = Cube(size=6, voxel_size=0.2, center=True)

hollow = s - Sphere(r=3.5, voxel_size=0.2)   # hollow sphere
rounded_cube = s & c                           # sphere-clipped cube
combined = s | c                               # merged shape
```

## Chaining

Operations return models that support further operations:

```python
from voxelcad import Sphere, Cube, Cylinder

body = Cube(size=6, voxel_size=0.2, center=True)
hole = Cylinder(h=8, r=1.5, center=True, voxel_size=0.2)

# Cube with a cylindrical hole, intersected with a sphere
result = (body - hole) & Sphere(r=4, voxel_size=0.2)
```

Chained operations build a lazy CSG tree. Nothing renders until you call `plot()`, `export()`, or `render_volume()`.

## Grid Matching

For best performance, use the same `voxel_size` on all operands:

```python
# Fast: same voxel_size, byte-level bitwise ops
a = Sphere(r=5, voxel_size=0.1)
b = Cube(size=8, voxel_size=0.1, center=True)
result = a & b  # direct bitwise_and on packed arrays

# Slower: different voxel_size, requires resampling
a = Sphere(r=5, voxel_size=0.1)
b = Cube(size=8, voxel_size=0.2, center=True)
result = a & b  # renders both to common grid first
```

When operands share the same grid (same voxel_size and overlapping bounds), VoxelCAD uses byte-level bitwise operations on packed arrays - effectively instant. When grids differ, all operands render to a common union grid before combining.

## Difference vs XOR

Difference (`-`) is asymmetric: `a - b` removes `b` from `a`, keeping only parts of `a` that don't overlap with `b`.

XOR (`^`) is symmetric: `a ^ b` keeps everything that's in exactly one of the two shapes.

```python
s = Sphere(r=4, voxel_size=0.2)
c = Cube(size=6, voxel_size=0.2, center=True)

diff = s - c   # sphere with cube-shaped bite taken out
xor = s ^ c    # shell-like shape where they don't overlap
```

## Inversion

`~model` flips every voxel in the model's bounding box. Useful for creating negative molds:

```python
s = Sphere(r=3, voxel_size=0.2)
mold = Cube(size=8, voxel_size=0.2, center=True) & ~s  # cube with sphere-shaped cavity
```

Inversion is always eager (returns immediately) since it operates on a single model's packed data.
