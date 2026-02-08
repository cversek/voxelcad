"""Test fallback paths: CSG streaming, non-standard grids, edge geometry."""
import pytest
import numpy as np
from voxelcad import Cube, Sphere
from voxelcad.voxel_model import CSGModel
from tests.conftest import LOW_RES_VS


def test_csg_streaming_produces_volume():
    """CSGModel with different-grid children renders via streaming evaluate_slice."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    csg = a & b
    assert type(csg) is CSGModel
    csg.render_volume()
    V = csg._unpack_volume()
    # Intersection of cube[-5,5] and sphere[r=4] should have voxels
    assert V.any()
    # Should be smaller than either parent alone
    a.render_volume()
    assert V.sum() < a._unpack_volume().sum()


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


def test_empty_intersection():
    """Non-overlapping grids: VoxelGrid.__and__ raises AssertionError.

    When two grids don't overlap, the intersection bbox has maximin > minimax,
    producing invalid limits. VoxelGrid.__init__ asserts xlim[0] < xlim[1].
    This is a known limitation — see GitHub issue on VoxelCAD repo.
    Future fix: return an empty VoxelModel or raise a descriptive error.
    """
    a = Cube(size=2, voxel_size=0.5, center=True)   # bbox [-1,1]
    b = Cube(size=2, voxel_size=0.5)                 # bbox [0,2] — non-centered
    b_far = b.translate([10, 10, 10])                # bbox [10,12]
    with pytest.raises(AssertionError):
        a & b_far


def test_unpack_slice_matches_full_unpack(cube32):
    """Per-slice unpack matches full volume unpack."""
    cube32.render_volume()
    V_full = cube32._unpack_volume()
    for k in range(cube32._voxel_shape[2]):
        V_slice = cube32._unpack_slice(k)
        np.testing.assert_array_equal(V_slice, V_full[:, :, k])
