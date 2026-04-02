# Query Planner

How VoxelCAD optimizes CSG tree evaluation.

## Problem

A CSG expression like `(A & B) | (C - D)` builds a tree of operation nodes. Naive evaluation would render and combine pairwise, materializing intermediate volumes at each step. The query planner flattens this tree into a single render-then-combine pass.

## CSGModel Tree Structure

Each boolean operation between non-same-grid operands creates a `CSGModel` node (`voxel_model.py:486-566`):

```python
class CSGModel(VoxelModel):
    def __init__(self, left, op, right, grid):
        self.left = left      # VoxelModel or CSGModel
        self.op = op          # 'or', 'and', 'xor', 'sub'
        self.right = right    # VoxelModel or CSGModel
        self.grid = grid      # Pre-computed union grid
```

CSGModel nodes are lazy — no rendering happens until `render_volume()` is called.

## Execution Planning

When `render_volume()` is called, the planner traverses the tree once (`voxel_model.py:524-535`):

```python
def _plan_execution(self):
    plan = ExecutionPlan()
    self._collect_leaves(plan.leaves, plan.operations)
    union_grid = plan.leaves[0].grid
    for leaf in plan.leaves[1:]:
        union_grid = union_grid | leaf.grid
    plan.common_grid = union_grid
    return plan
```

### Leaf Collection

`_collect_leaves` (`voxel_model.py:506-522`) walks the tree in postfix order:

- **Leaf nodes** (VoxelModel, TransformedModel, primitives): added to the leaf list with their index
- **Internal nodes** (CSGModel): recurse left and right, then record `(op, left_idx, right_idx)` in the operations list

The result is two parallel structures:
- `leaves`: flat list of renderable models
- `operations`: list of `(op, left_idx, right_idx)` tuples for stack evaluation

### Common Grid Computation

All leaves are rendered onto a single union grid — the bounding box that encloses every leaf, using the finest voxel size (`voxel_grid.py:113-144`):

```python
def __or__(self, other):  # Grid union
    xlim = (min(self.xlim[0], other.xlim[0]),
            max(self.xlim[1], other.xlim[1]))
    # ... same for ylim, zlim ...
    # Use finer voxel size to avoid data loss
    new_vsv = np.vstack((vsv1, vsv2)).min(axis=0)
    return cls(xlim, ylim, zlim, new_vsv)
```

## Execution

Rendering proceeds in two phases (`voxel_model.py:537-566`):

**Phase 1 — Render leaves**: Each leaf calls `render_on_grid(common_grid, M4inv)`. For primitives, this evaluates geometry via Cython kernels. For transformed models, the inverse transform matrix is passed through. All leaves produce packed `uint8` arrays on the same grid.

**Phase 2 — Combine**: A stack-based postfix evaluator applies byte-level operations:

```python
stack = list(rendered)
for op, left_idx, right_idx in plan.operations:
    result = self._BYTEWISE_OP_MAP[op](stack[left_idx], stack[right_idx])
    stack.append(result)
self.voxel_data = stack[-1]
```

Since all arrays share the same grid, combination is byte-level `numpy` ops — the same fast path as Tier 1.

## Operation Map

```python
_BYTEWISE_OP_MAP = {
    'or':  np.bitwise_or,
    'and': np.bitwise_and,
    'xor': np.bitwise_xor,
    'sub': lambda a, b: np.bitwise_and(a, np.bitwise_not(b)),
}
```

`sub` (difference) is implemented as `A AND (NOT B)` — two byte-level operations.

## Example

```python
result = (Sphere(3) & Cube(5, center=True)) | Cylinder(10, r=2)
```

Tree:

```
        OR
       /  \
     AND   Cylinder
    /   \
Sphere  Cube
```

Planner output:
- `leaves`: [Sphere, Cube, Cylinder]
- `operations`: [('and', 0, 1), ('or', 3, 2)]
- `common_grid`: union of all three bounding boxes

Execution:
1. Render Sphere on common grid → `packed[0]`
2. Render Cube on common grid → `packed[1]`
3. Render Cylinder on common grid → `packed[2]`
4. `packed[3] = bitwise_and(packed[0], packed[1])`
5. `packed[4] = bitwise_or(packed[3], packed[2])`
6. Result = `packed[4]`

Three render calls (expensive), two byte-level combinations (cheap). No intermediate volumes beyond the packed arrays.

## Design Constraints

- **Union grid grows with operand spread**: Widely separated objects produce a large common grid with many empty voxels. This is correct but wastes memory on the bounding region.
- **Voxel size mismatch**: The finer voxel size wins, which can significantly increase resolution when mixing coarse and fine models.
- **All leaves rendered**: The planner does not short-circuit (e.g., skipping a leaf if its contribution would be masked). Every leaf is fully rendered.
