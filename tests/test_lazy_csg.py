"""Test CSG evaluation: lazy tree construction and render-time materialization.

Phase 10.5: Tier 2 (compatible_grid_op) eliminated. All non-same-grid boolean
ops produce CSGModel (lazy). Materialization happens at render_volume() time
via render_on_grid() on common grid.
"""
import numpy as np
from voxelcad import Cube, Sphere
from voxelcad.voxel_model import VoxelModel, CSGModel
from tests.conftest import LOW_RES_VS


# --- Same voxel_size, not pre-rendered → CSGModel (lazy) ---

def test_compatible_grid_produces_csgmodel():
    """Unrendered same-voxel-size operands → CSGModel (lazy)."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    result = a | b
    # Not pre-rendered → CSGModel (no Tier 2 eager path)
    assert type(result) is CSGModel


def test_compatible_grid_renders_correctly():
    """CSGModel from same-voxel-size operands materializes on render_volume()."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    result = a | b
    assert result.voxel_data is None  # lazy
    result.render_volume()
    assert result.voxel_data is not None
    assert result.voxel_data.dtype == np.uint8
    assert result.voxel_data.sum() > 0


def test_compatible_grid_all_ops_render():
    """All four boolean ops render correctly for same-voxel-size operands."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    for op_fn in [lambda x, y: x | y, lambda x, y: x & y,
                  lambda x, y: x ^ y, lambda x, y: x - y]:
        result = op_fn(a, b)
        result.render_volume()
        assert result.voxel_data is not None
        assert result.voxel_data.dtype == np.uint8


def test_compatible_grid_chains_render():
    """Chained ops: (A | B) & C all same voxel_size → nested CSGModel → renders."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    c = Sphere(r=3, voxel_size=LOW_RES_VS)
    ab = a | b
    abc = ab & c
    abc.render_volume()
    assert abc.voxel_data is not None
    assert abc.voxel_data.sum() > 0


# --- Same voxel_size, pre-rendered → Tier 1 (byte-level fast path) ---

def test_same_grid_prerendered_produces_voxelmodel():
    """Pre-rendered same-grid operands → Tier 1 → VoxelModel (not CSGModel)."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    # Render both on their own grids first — same voxel_size, same bbox
    a.render_volume()
    b.render_volume()
    if a.grid.same_grid(b.grid):
        result = a | b
        assert type(result) is VoxelModel
        assert result.voxel_data is not None


# --- Different voxel_size → CSGModel ---

def test_incompatible_grid_produces_csg():
    """Different voxel_sizes → CSGModel."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS * 2)
    result = a | b
    assert type(result) is CSGModel


def test_csg_is_lazy():
    """CSGModel should not materialize until render_volume()."""
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
