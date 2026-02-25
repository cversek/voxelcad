# Optimization System

How VoxelCAD achieves fast boolean operations through tiered dispatch.

## Overview

Boolean operations (`|`, `&`, `^`, `-`) dispatch through three optimization tiers, chosen automatically based on operand state. Tier 1 is 100-1000x faster than Tier 3; the system selects the fastest available path.

## Tier 1: Same-Grid Byte-Level Ops

When both operands are materialized and share the same grid (origin, resolution, voxel size), boolean operations reduce to `numpy` byte ops on packed `uint8` arrays.

**Activation**: `self.voxel_data is not None and other.voxel_data is not None and self.grid.same_grid(other.grid)`

**Implementation** (`voxel_model.py:452-477`):

```python
def __or__(self, other):
    if (self.voxel_data is not None and other.voxel_data is not None
            and self.grid.same_grid(other.grid)):
        return self._same_grid_op(other, np.bitwise_or)
    return CSGModel(self, 'or', other, self.grid | other.grid)
```

`_same_grid_op` calls `_ensure_rendered()` on both operands, then applies the bitwise function directly on packed arrays. At 1024^3 resolution, this operates on ~128 MB of contiguous bytes. Typical time: ~10ms.

**Grid equality** (`voxel_grid.py:36-47`): requires `np.array_equal(res_vector)` and `np.allclose` on all limits and voxel sizes.

## Tier 2: CSG Tree with Query Planner

When operands don't share the same grid, the operation builds a lazy `CSGModel` node. At render time, the query planner optimizes execution.

**CSGModel** (`voxel_model.py:486-566`): stores `left`, `op`, `right`, and a pre-computed union grid.

**Query planning** (`voxel_model.py:524-535`):

1. Traverse the CSG tree in postfix order, collecting leaf models and operations
2. Compute the union grid (enclosing bounding box, finest voxel size)
3. Render all leaves onto the common grid via `render_on_grid()`
4. Combine results with stack-based postfix evaluation using byte-level ops

```python
# Bytewise operation map (voxel_model.py:493-498)
_BYTEWISE_OP_MAP = {
    'or':  np.bitwise_or,
    'and': np.bitwise_and,
    'xor': np.bitwise_xor,
    'sub': lambda a, b: np.bitwise_and(a, np.bitwise_not(b)),
}
```

This reduces multi-operand CSG trees to a single common grid, then all combinations are byte-level. The cost is rendering each leaf once; the benefit is that combination is instant.

**Union grid** (`voxel_grid.py:113-144`): takes the min/max bounds and the finer voxel size of the two operands.

## Tier 3: Geometry Evaluation (render_on_grid)

The lowest tier evaluates geometry or resamples data for each leaf.

**Entry point** (`voxel_model.py:84-103`):

```python
def render_on_grid(self, grid, M4inv=None):
    # Same-grid cache hit
    if M4inv is None and self.voxel_data is not None and self.grid.same_grid(grid):
        return self.voxel_data
    # Dispatch to implementation
    if ENV.use_cython:
        return self._render_cython(grid, M4inv)
    return self._render_numpy(grid, M4inv)
```

**For primitives** (Sphere, Cube, Cylinder, GyroidCube): `_render_cython` calls the fused Cython kernel, which evaluates geometry + thresholds + bit-packs in a single pass. Falls back to `_render_numpy` (per-slice iteration) when Cython is unavailable.

**For data-only models** (base VoxelModel with existing packed data): `_render_cython` calls `resample_and_pack`, a nearest-neighbor resampling kernel. Falls back to per-slice NumPy resampling.

## Transform Composition

`TransformedModel` (`voxel_model.py:569-622`) is a lazy wrapper that composes transform chains into a single 4x4 inverse matrix (`M4inv`). When rendered, it delegates to `source.render_on_grid(grid, M4inv)`.

### Why Inverse Transforms?

Rendering fills a destination grid: for each output voxel at position `(x, y, z)`, the system needs to determine whether that point is inside the geometry. When the geometry has been rotated, we don't move the geometry — we ask "where in the original (untransformed) geometry does this output point correspond to?"

The inverse transform `M4inv` maps destination coordinates back to source coordinates:

```
destination (x, y, z)  --->  M4inv  --->  source (xp, yp, zp)
```

The geometry test then runs on `(xp, yp, zp)` using the original, axis-aligned equations. This is why kernels apply `M4inv` before the geometry check:

```cython
# Transform destination coords to source space
xp = M4inv[0,0]*x + M4inv[0,1]*y + M4inv[0,2]*z + M4inv[0,3]
yp = M4inv[1,0]*x + M4inv[1,1]*y + M4inv[1,2]*z + M4inv[1,3]
zp = M4inv[2,0]*x + M4inv[2,1]*y + M4inv[2,2]*z + M4inv[2,3]
# Test against untransformed geometry
if xp*xp + yp*yp + zp*zp <= r_sq:
    set_bit(out, lin_idx)
```

The alternative — applying the forward transform to every source voxel — would scatter points into non-grid-aligned locations, requiring interpolation and producing a sparse, irregular output. The inverse approach guarantees every output voxel gets exactly one evaluation, producing a dense packed array with no gaps.

### Chain Composition

Multiple transforms compose into a single matrix multiplication, so chained operations cost nothing extra at render time:

```python
# Transform composition (voxel_model.py:601-610)
composed_M4inv = self.M4inv @ M4inv_new
return TransformedModel(self.source, composed_M4, composed_M4inv, new_grid)
```

A `sphere.rotate_x(30).rotate_z(45).translate([1,0,0])` produces one M4inv matrix — no intermediate volumes.

## Dispatch Decision Tree

```
A | B
  |
  +-- Both materialized + same_grid?
  |     YES --> Tier 1: np.bitwise_or(A.voxel_data, B.voxel_data)
  |     NO  --> CSGModel(A, 'or', B)
  |
  +-- CSGModel.render_volume()
        |
        +-- Plan: collect leaves, compute union grid
        +-- For each leaf: render_on_grid(union_grid, M4inv?)
        |     |
        |     +-- Same-grid cache hit? --> return cached
        |     +-- ENV.use_cython?
        |           YES --> _render_cython(grid, M4inv)
        |           NO  --> _render_numpy(grid, M4inv)
        |
        +-- Combine: postfix stack with bytewise ops
```

## Performance Characteristics

| Tier | Operation | 1024^3 Time | Memory |
|------|-----------|-------------|--------|
| 1 | Byte-level bitwise | ~10 ms | 128 MB (packed) |
| 2 | Render leaves + combine | 100-1000 ms | 128 MB per leaf |
| 3 | Geometry evaluation | 80-400 ms per primitive | 50 MB (Cython), 1 GB (NumPy) |

The gap between Tier 1 and Tier 3 grows super-linearly with resolution because Tier 1 is memory-bound (O(n)) while Tier 3 is compute-bound (O(n) with higher constant).
