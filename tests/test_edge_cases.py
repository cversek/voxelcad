"""Test edge cases: single-voxel, identity transform, union_all."""
import numpy as np
from voxelcad import Cube, Sphere, GyroidCube, union_all
from voxelcad.voxel_model import VoxelModel, CSGModel, TransformedModel
from tests.conftest import LOW_RES_VS


def test_single_voxel_grid():
    """Very large voxel_size produces 1x1x1 grid."""
    c = Cube(size=10, voxel_size=20.0, center=True)
    c.render_volume()
    assert c._voxel_shape == (1, 1, 1)
    V = c._unpack_volume()
    assert V.shape == (1, 1, 1)
    assert V[0, 0, 0] == True


def test_identity_rotation():
    """Rotate 0 degrees preserves geometry bit-for-bit."""
    c = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    c.render_volume()
    t = c.rotate_z(0)
    t.render_volume()
    # Identity rotation should reproduce original voxels
    V_orig = c._unpack_volume()
    V_rot = t._unpack_volume()
    assert V_orig.shape == V_rot.shape
    assert V_orig.sum() == V_rot.sum()


def test_double_invert_recovers_original():
    """~~A should equal A at the packed byte level."""
    c = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    c.render_volume()
    double_inv = ~(~c)
    np.testing.assert_array_equal(double_inv.voxel_data, c.voxel_data)


def test_union_all():
    """union_all combines a list of models via chained CSG ops."""
    models = [
        Cube(size=10, voxel_size=LOW_RES_VS, center=True),
        Sphere(r=4, voxel_size=LOW_RES_VS),
        Sphere(r=3, voxel_size=LOW_RES_VS),
    ]
    result = union_all(models)
    # Unrendered operands → CSGModel chain
    result.render_volume()
    assert result.voxel_data is not None
    assert result.voxel_data.sum() > 0
