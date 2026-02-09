"""Test CSG evaluation paths: Tier 2 (compatible grid) and Tier 3 (incompatible grid)."""
import numpy as np
from voxelcad import Cube, Sphere
from voxelcad.voxel_model import VoxelModel, CSGModel
from tests.conftest import LOW_RES_VS


# --- Tier 2: compatible grids (same voxel_size, different bbox) → VoxelModel ---

def test_compatible_grid_produces_voxelmodel():
    """Cube(size=10) + Sphere(r=4) share voxel_size → Tier 2 → VoxelModel."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    result = a | b
    assert type(result) is VoxelModel


def test_compatible_grid_is_materialized():
    """Tier 2 result has voxel_data immediately (not lazy)."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    result = a | b
    assert result.voxel_data is not None
    assert result.voxel_data.dtype == np.uint8
    assert result.voxel_data.sum() > 0


def test_compatible_grid_all_ops():
    """All four boolean ops produce VoxelModel for compatible grids."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    for op_fn in [lambda x, y: x | y, lambda x, y: x & y,
                  lambda x, y: x ^ y, lambda x, y: x - y]:
        result = op_fn(a, b)
        assert type(result) is VoxelModel


def test_compatible_grid_chains():
    """Chained Tier 2 ops: (A | B) & C all compatible → VoxelModel throughout."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    c = Sphere(r=3, voxel_size=LOW_RES_VS)
    ab = a | b
    assert type(ab) is VoxelModel
    abc = ab & c
    assert type(abc) is VoxelModel
    assert abc.voxel_data is not None


# --- Tier 3: incompatible grids (different voxel_size) → CSGModel ---

def test_incompatible_grid_produces_csg():
    """Different voxel_sizes → Tier 3 → CSGModel."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS * 2)  # 2x coarser
    result = a | b
    assert type(result) is CSGModel


def test_csg_is_lazy():
    """CSGModel (Tier 3) should not materialize until render_volume()."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS * 2)
    result = a | b
    assert result.voxel_data is None


def test_csg_renders_correctly():
    """CSGModel.render_volume() produces valid packed data."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS * 2)
    result = a | b
    result.render_volume()
    assert result.voxel_data is not None
    assert result.voxel_data.dtype == np.uint8
    assert result.voxel_data.sum() > 0
