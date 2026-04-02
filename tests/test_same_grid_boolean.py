"""Test same-grid boolean fast path.

IMPORTANT: Same-grid requires identical bounding boxes.
Cube(size=10, center=True) and GyroidCube(size=10, center=True) both have
bbox [-5,5]^3. Do NOT use Cube+Sphere (different bboxes — Phase 10.1 scope).
"""
import numpy as np
from voxelcad import Cube, GyroidCube
from voxelcad.voxel_model import VoxelModel, CSGModel
from tests.conftest import LOW_RES_VS


def _make_pair():
    """Two primitives with identical grids."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = GyroidCube(size=10, voxel_size=LOW_RES_VS, center=True)
    a.render_volume()
    b.render_volume()
    return a, b


def test_union_fast_path():
    a, b = _make_pair()
    result = a | b
    assert type(result) is VoxelModel, f"Expected VoxelModel, got {type(result).__name__}"
    assert result.voxel_data is not None
    assert result.voxel_data.dtype == np.uint8


def test_intersection_fast_path():
    a, b = _make_pair()
    result = a & b
    assert type(result) is VoxelModel
    assert result.voxel_data.sum() > 0


def test_xor_fast_path():
    a, b = _make_pair()
    result = a ^ b
    assert type(result) is VoxelModel
    # XOR of cube (all True) and gyroid (partial) should have voxels
    assert result.voxel_data.sum() > 0


def test_difference_fast_path():
    a, b = _make_pair()
    result = a - b
    assert type(result) is VoxelModel
    assert result.voxel_data.sum() > 0


def test_invert():
    a, _ = _make_pair()
    inv = ~a
    assert type(inv) is VoxelModel
    assert inv.voxel_data is not None
    # Double invert should recover original packed bytes
    double_inv = ~inv
    assert np.array_equal(double_inv.voxel_data, a.voxel_data)
