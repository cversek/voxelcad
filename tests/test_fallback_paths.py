"""Test fallback paths: CSG streaming, non-standard grids, edge geometry."""
import pytest
import numpy as np
from voxelcad import Cube, Sphere
from voxelcad.voxel_model import VoxelModel, CSGModel
from tests.conftest import LOW_RES_VS


def test_csg_streaming_produces_volume():
    """CSGModel with incompatible-grid children renders via streaming evaluate_slice."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS * 2)  # different voxel_size → Tier 3
    csg = a & b
    assert type(csg) is CSGModel
    csg.render_volume()
    V = csg._unpack_volume()
    assert V.any()


def test_tier2_intersection_correctness():
    """Tier 2 intersection: Cube[-5,5] & Sphere[r=4] should be subset of both."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    result = a & b
    assert type(result) is VoxelModel  # Tier 2
    V = result._unpack_volume()
    assert V.any()
    # Render parents on same grid to compare
    a_on_grid = a.render_on_grid(result.grid)
    assert V.sum() <= a_on_grid._unpack_volume().sum()


def test_non_power_of_2_grid():
    """Non-power-of-2 resolution from voxel_size=0.3 on size=10."""
    c = Cube(size=10, voxel_size=0.3, center=True)
    c.render_volume()
    rx, ry, rz = c._voxel_shape
    # ceil(10/0.3) = 34, not a power of 2
    assert rx == 34
    assert c.voxel_data is not None
    V = c._unpack_volume()
    assert V.shape == (34, 34, 34)
    assert V.all()  # Cube fills its own grid


def test_empty_intersection_compatible_grids():
    """Non-overlapping compatible grids: Tier 2 produces empty VoxelModel (no crash).

    With Tier 2, compatible grids render on the union grid and bitwise_and.
    Non-overlapping geometry produces all-zero voxel_data — no AssertionError.
    """
    a = Cube(size=2, voxel_size=0.5, center=True)   # bbox [-1,1]
    b = Cube(size=2, voxel_size=0.5)                 # bbox [0,2]
    b_far = b.translate([10, 10, 10])                # bbox [10,12]
    result = a & b_far
    assert type(result) is VoxelModel
    assert result._unpack_volume().sum() == 0  # empty intersection


def test_empty_intersection_incompatible_grids():
    """Non-overlapping incompatible grids still raise AssertionError (Tier 3).

    VoxelGrid.__and__ asserts xlim[0] < xlim[1], which fails for
    non-overlapping grids. This is a known limitation (voxelcad#1).
    """
    a = Cube(size=2, voxel_size=0.5, center=True)
    b = Cube(size=2, voxel_size=1.0)                 # different voxel_size
    b_far = b.translate([10, 10, 10])
    with pytest.raises(AssertionError):
        a & b_far


def test_unpack_slice_matches_full_unpack(cube32):
    """Per-slice unpack matches full volume unpack."""
    cube32.render_volume()
    V_full = cube32._unpack_volume()
    for k in range(cube32._voxel_shape[2]):
        V_slice = cube32._unpack_slice(k)
        np.testing.assert_array_equal(V_slice, V_full[:, :, k])
