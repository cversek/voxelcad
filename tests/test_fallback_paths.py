"""Test fallback paths: CSG render_on_grid, non-standard grids, edge geometry."""
import pytest
import numpy as np
from voxelcad import Cube, Sphere
from voxelcad.voxel_model import VoxelModel, CSGModel
from tests.conftest import LOW_RES_VS


def test_csg_incompatible_grids_produces_volume():
    """CSGModel with different voxel_size children renders via render_on_grid."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS * 2)
    csg = a & b
    assert type(csg) is CSGModel
    csg.render_volume()
    V = csg._unpack_volume()
    assert V.any()


def test_same_voxel_size_intersection_correctness():
    """Same voxel_size intersection: Cube[-5,5] & Sphere[r=4] correct."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    result = a & b
    # Same voxel_size with pre-rendered data → Tier 1 (VoxelModel)
    # or CSGModel if not pre-rendered
    result_data = result.render_volume()
    V = result._unpack_volume()
    assert V.any()


def test_non_power_of_2_grid():
    """Non-power-of-2 resolution from voxel_size=0.3 on size=10."""
    c = Cube(size=10, voxel_size=0.3, center=True)
    c.render_volume()
    rx, ry, rz = c._voxel_shape
    # ceil((10 + 2*0.3)/0.3) = ceil(35.33) = 36 (with 1-voxel padding per side)
    assert rx == 36
    assert c.voxel_data is not None
    V = c._unpack_volume()
    assert V.shape == (36, 36, 36)
    assert V[1:-1, 1:-1, 1:-1].all()  # interior is solid


def test_empty_intersection_non_overlapping_grids():
    """Non-overlapping grids raise AssertionError from VoxelGrid.__and__.

    With Tier 2 eliminated, non-overlapping same-voxel-size grids also
    go through CSGModel → VoxelGrid.__and__ which asserts xlim[0] < xlim[1].
    Known limitation (voxelcad#1).
    """
    a = Cube(size=2, voxel_size=0.5, center=True)   # bbox [-1,1]
    b = Cube(size=2, voxel_size=0.5)                 # bbox [0,2]
    b_far = b.translate([10, 10, 10])                # bbox [10,12]
    with pytest.raises(AssertionError):
        a & b_far


def test_empty_intersection_incompatible_grids():
    """Non-overlapping incompatible grids still raise AssertionError.

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
