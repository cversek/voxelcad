"""Test lazy CSG evaluation with different-grid operands."""
import numpy as np
from voxelcad import Cube, Sphere
from voxelcad.voxel_model import CSGModel
from tests.conftest import LOW_RES_VS


def test_different_grid_produces_csg():
    """Cube(size=10) + Sphere(r=4) have different bboxes → CSGModel."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    result = a | b
    assert type(result) is CSGModel


def test_csg_is_lazy():
    """CSGModel should not materialize until render_volume() is called."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    result = a | b
    assert result.voxel_data is None


def test_csg_renders_correctly():
    """CSGModel.render_volume() produces valid packed data."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    result = a | b
    result.render_volume()
    assert result.voxel_data is not None
    assert result.voxel_data.dtype == np.uint8
    assert result.voxel_data.sum() > 0


def test_csg_depth_two():
    """Nested CSG: (A | B) & C produces CSGModel with CSGModel child."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    c = Sphere(r=3, voxel_size=LOW_RES_VS)
    ab = a | b
    abc = ab & c
    assert type(abc) is CSGModel
    assert type(abc.left) is CSGModel
    abc.render_volume()
    assert abc.voxel_data is not None


def test_csg_all_ops():
    """All four CSG operations produce CSGModel for different grids."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    for op_fn in [lambda x, y: x | y, lambda x, y: x & y,
                  lambda x, y: x ^ y, lambda x, y: x - y]:
        result = op_fn(a, b)
        assert type(result) is CSGModel
