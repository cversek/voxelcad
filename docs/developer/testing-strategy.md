# Testing Strategy

## Running Tests

```bash
make test                              # full suite
python -m pytest tests/ -v             # verbose output
python -m pytest tests/test_primitives.py -v  # single file
python -m pytest tests/ -k "sphere"    # filter by name
```

## Fixtures

Shared fixtures live in `tests/conftest.py`. All use low-resolution grids (32^3) for fast execution:

```python
LOW_RES_VS = 10.0 / 32  # ~0.3125 voxel size

@pytest.fixture
def cube32():
    return Cube(size=10, voxel_size=LOW_RES_VS, center=True)

@pytest.fixture
def sphere32():
    return Sphere(r=5, voxel_size=LOW_RES_VS)

@pytest.fixture
def cylinder32():
    return Cylinder(h=10, r=5, center=True, voxel_size=LOW_RES_VS)

@pytest.fixture
def gyroid32():
    return GyroidCube(size=10, voxel_size=LOW_RES_VS, center=True)
```

Use these fixtures rather than constructing models in each test. They keep tests consistent and fast.

## Test Categories

### Primitive rendering (`test_primitives.py`)

Verify that each primitive renders to packed uint8, produces non-empty output, and round-trips through unpack:

```python
def test_cube_render(cube32):
    data = cube32.render_volume()
    assert data.dtype == np.uint8
    assert cube32._voxel_shape == (32, 32, 32)
    assert data.sum() > 0

def test_unpack_roundtrip(cube32):
    cube32.render_volume()
    V = cube32._unpack_volume()
    assert V.dtype == np.bool_
    assert V.shape == (32, 32, 32)
```

### Boolean operations (`test_same_grid_boolean.py`, `test_lazy_csg.py`)

Same-grid tests verify byte-level fast path. CSG tests verify the lazy tree evaluation:

```python
def test_same_grid_union(sphere32, cube32):
    result = sphere32 | cube32
    result.render_volume()
    # Union should contain at least as many voxels as either operand
    assert result.voxel_data.sum() >= sphere32.voxel_data.sum()
```

### Transforms (`test_edge_cases.py`)

Verify that transforms compose correctly and that the inverse matrix is applied during rendering.

### Fallback paths (`test_fallback_paths.py`)

Test NumPy rendering when Cython is unavailable. These temporarily set `ENV.use_cython = False`.

### Endian regression (`test_endian_regression.py`)

Voxel data is packed with `np.packbits(..., order='F', bitorder='big')`. The Cython kernels must match this convention exactly. A mismatch produces scrambled geometry where ~50% of voxels are wrong (byte-aligned, systematic). These tests compare Cython and NumPy output via XOR to catch any packing disagreement:

```python
# XOR between implementations should be < 1% (surface rounding only)
xor_count = np.bitwise_xor(cython_data, numpy_data).sum()
total_voxels = np.prod(shape)
assert xor_count / total_voxels < 0.01
```

Small XOR counts (<1%) are expected at surface boundaries where `floor` vs `round-to-nearest` differ. Large counts (>10%) indicate an endianness or packing-order bug.

## Writing New Tests

1. Use existing fixtures from `conftest.py` when possible
2. Keep grids small (32^3) — tests should finish in under 1 second each
3. Test both Cython and NumPy paths if your code dispatches between them
4. For boolean operations, verify the result is non-empty and has plausible voxel counts
5. For transforms, verify the model renders without error and produces output

## Benchmarks

Benchmarks live in `benchmarks/` and use the `BenchmarkBase` class:

```bash
make benchmark                                     # run all
python -m pytest benchmarks/benchmark_render.py -v  # render only
```

Benchmark classes define `setup`, `run`, and `validate` methods:

```python
from super_utils.benchmarks import BenchmarkBase

class BenchmarkSphereRender(BenchmarkBase):
    name = "sphere_render"
    description = "Sphere.render_volume()"
    workload_type = "memory-bound"

    def setup(self):
        vs = 10.0 / RESOLUTIONS[self.size]
        self.model = Sphere(r=5, voxel_size=vs)

    def run(self):
        self.model.voxel_data = None
        self.model.render_volume()

    def validate(self):
        return self.model.voxel_data is not None
```

Sizes: `small` (32^3, CI), `medium` (256^3, dev), `large` (1024^3, profiling).
