"""Test affine transform composition and lazy evaluation."""
import numpy as np
from voxelcad import Cube
from voxelcad.voxel_model import TransformedModel
from tests.conftest import LOW_RES_VS


def test_rotate_produces_transformed():
    c = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    t = c.rotate_z(45)
    assert type(t) is TransformedModel
    assert t.M4.shape == (4, 4)
    assert t.M4inv.shape == (4, 4)


def test_scale_produces_transformed():
    c = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    t = c.scale(2.0)
    assert type(t) is TransformedModel


def test_translate_produces_transformed():
    c = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    t = c.translate([5, 0, 0])
    assert type(t) is TransformedModel
    # Grid should shift by the translation vector
    assert t.grid.xlim[0] == c.grid.xlim[0] + 5
    assert t.grid.xlim[1] == c.grid.xlim[1] + 5


def test_chained_transforms_compose():
    """Chained transforms produce single TransformedModel, not nested."""
    c = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    t = c.rotate_z(45).scale(2.0).rotate_x(30)
    assert type(t) is TransformedModel
    # Source should be the original cube, not an intermediate TransformedModel
    assert t.source is c


def test_m4_times_m4inv_is_identity():
    """Forward and inverse transforms should compose to identity."""
    c = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    t = c.rotate_z(45).scale(2.0).translate([1, 2, 3])
    product = t.M4 @ t.M4inv
    np.testing.assert_allclose(product, np.eye(4), atol=1e-10)


def test_transformed_renders():
    """TransformedModel.render_volume() produces valid packed data."""
    c = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    c.render_volume()
    t = c.rotate_z(45)
    t.render_volume()
    assert t.voxel_data is not None
    assert t.voxel_data.dtype == np.uint8
    assert t.voxel_data.sum() > 0
