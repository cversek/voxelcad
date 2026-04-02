"""Tests for CSG Tree Query Planner (Phase 10.5 — Unified Interface).

Validates that _plan_execution() correctly:
1. Collects all leaves in tree traversal order
2. Records operations in postfix order
3. Computes common (union) grid for all leaves
4. All leaves render on common grid via render_on_grid()
"""
import pytest
import numpy as np

from voxelcad.sphere import Sphere
from voxelcad.cube import Cube
from voxelcad.cylinder import Cylinder
from voxelcad.gyroid_cube import GyroidCube
from voxelcad.voxel_model import (
    VoxelModel, CSGModel, TransformedModel,
    LeafNode, ExecutionPlan,
)
import voxelcad.environment as ENV


# Use coarse voxel size for fast tests
VS = 1.0


class TestUnifiedInterface:
    """Test that all primitives implement _render_cython/_render_numpy."""

    def test_all_primitives_have_render_methods(self):
        """All primitives override _render_cython and _render_numpy."""
        primitives = [
            Sphere(r=5, voxel_size=VS),
            Cube(size=10, voxel_size=VS),
            Cylinder(h=10, r=5, voxel_size=VS),
            GyroidCube(size=10, voxel_size=VS, thresh1=0.5),
        ]
        for p in primitives:
            assert hasattr(p, '_render_cython'), f"{type(p).__name__} missing _render_cython"
            assert hasattr(p, '_render_numpy'), f"{type(p).__name__} missing _render_numpy"
            assert hasattr(p, 'render_on_grid'), f"{type(p).__name__} missing render_on_grid"

    def test_render_on_grid_returns_packed_array(self):
        """render_on_grid returns packed uint8 array."""
        s = Sphere(r=5, voxel_size=VS)
        result = s.render_on_grid(s.grid)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_render_on_grid_same_grid_shortcut(self):
        """After render_volume(), render_on_grid(own grid) returns cached data."""
        s = Sphere(r=5, voxel_size=VS)
        s.render_volume()  # populates voxel_data
        data1 = s.render_on_grid(s.grid)
        data2 = s.render_on_grid(s.grid)
        assert data1 is data2  # same object via same-grid shortcut

    def test_numpy_fallback_produces_result(self):
        """_render_numpy produces non-empty result for all primitives."""
        primitives = [
            Sphere(r=5, voxel_size=VS),
            Cube(size=10, voxel_size=VS),
            Cylinder(h=10, r=5, voxel_size=VS),
            GyroidCube(size=10, voxel_size=VS, thresh1=0.5),
        ]
        for p in primitives:
            result = p._render_numpy(p.grid)
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.uint8
            assert np.unpackbits(result, bitorder='big').sum() > 0, f"{type(p).__name__} produced empty result"

    def test_render_on_grid_with_foreign_grid(self):
        """Primitive can render on a different (foreign) grid."""
        s = Sphere(r=3, voxel_size=VS)
        # Foreign grid: larger, different origin
        from voxelcad.voxel_grid import VoxelGrid
        foreign = VoxelGrid(xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5), voxel_size=VS)
        result = s.render_on_grid(foreign)
        assert isinstance(result, np.ndarray)
        assert np.unpackbits(result, bitorder='big').sum() > 0


class TestQueryPlannerTreeWalk:
    """Test _plan_execution() tree walking and leaf collection."""

    def test_simple_binary_csg(self):
        """Simple A & B produces 2 leaves and 1 operation."""
        a = Sphere(r=5, voxel_size=VS)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        assert isinstance(csg, CSGModel)

        plan = csg._plan_execution()
        assert len(plan.leaves) == 2
        assert len(plan.operations) == 1
        assert plan.operations[0][0] == 'and'

    def test_depth_2_csg_tree(self):
        """(A | B) & C produces 3 leaves and 2 operations."""
        a = Sphere(r=5, voxel_size=VS)
        b = Cube(size=8, voxel_size=VS * 2)
        c = Cylinder(h=10, r=4, voxel_size=VS * 3)
        csg = (a | b) & c
        assert isinstance(csg, CSGModel)

        plan = csg._plan_execution()
        assert len(plan.leaves) == 3
        assert len(plan.operations) == 2
        assert plan.operations[0][0] == 'or'
        assert plan.operations[1][0] == 'and'

    def test_depth_3_csg_tree(self):
        """((A & B) | (C & D)) produces 4 leaves and 3 operations."""
        a = Sphere(r=5, voxel_size=VS)
        b = Cube(size=8, voxel_size=VS * 2)
        c = Cylinder(h=10, r=4, voxel_size=VS * 3)
        d = Sphere(r=3, voxel_size=VS * 4)
        csg = (a & b) | (c & d)
        assert isinstance(csg, CSGModel)

        plan = csg._plan_execution()
        assert len(plan.leaves) == 4
        assert len(plan.operations) == 3

    def test_common_grid_is_union(self):
        """Common grid is the union of all leaf grids."""
        a = Sphere(r=5, voxel_size=VS * 2)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        if isinstance(csg, CSGModel):
            plan = csg._plan_execution()
            assert plan.common_grid is not None
            assert plan.common_grid.xlim[0] <= min(a.grid.xlim[0], b.grid.xlim[0])
            assert plan.common_grid.xlim[1] >= max(a.grid.xlim[1], b.grid.xlim[1])


class TestExecutionPlanRepr:
    """Test ExecutionPlan string representation."""

    def test_repr_includes_key_info(self):
        """ExecutionPlan repr shows leaves and common_grid status."""
        a = Sphere(r=5, voxel_size=VS)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        assert isinstance(csg, CSGModel)

        plan = csg._plan_execution()
        r = repr(plan)
        assert "leaves=2" in r
        assert "common_grid=" in r


class TestCSGRenderVolume:
    """Test CSGModel.render_volume() with unified interface."""

    def test_csg_render_produces_volume(self):
        """CSGModel renders all leaves on common grid."""
        a = Sphere(r=5, voxel_size=VS * 2)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        if isinstance(csg, CSGModel):
            csg.render_volume()
            assert csg.voxel_data is not None
            assert csg.grid is not None

    def test_csg_incompatible_grids_renders(self):
        """CSGModel with different voxel_size leaves still renders."""
        a = Sphere(r=5, voxel_size=VS)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        assert isinstance(csg, CSGModel)
        csg.render_volume()
        assert csg.voxel_data is not None

    def test_csg_correct_intersection(self):
        """Planned execution produces geometrically correct results."""
        s = Sphere(r=3, voxel_size=VS * 2)
        c = Cube(size=10, center=True, voxel_size=VS * 2)
        csg = s & c
        if isinstance(csg, CSGModel):
            csg.render_volume()
            s_alone = Sphere(r=3, voxel_size=VS * 2)
            s_alone.render_volume()
            csg_sum = np.unpackbits(csg.voxel_data, bitorder='big').sum()
            s_sum = np.unpackbits(s_alone.voxel_data, bitorder='big').sum()
            assert abs(csg_sum - s_sum) / max(s_sum, 1) < 0.1

    def test_depth_2_csg_renders(self):
        """(A | B) & C renders correctly."""
        a = Sphere(r=3, voxel_size=VS * 2)
        b = Cube(size=6, center=True, voxel_size=VS * 2)
        c = Cylinder(h=10, r=5, center=True, voxel_size=VS * 2)
        csg = (a | b) & c
        if isinstance(csg, CSGModel):
            csg.render_volume()
            assert csg.voxel_data is not None
            assert np.unpackbits(csg.voxel_data, bitorder='big').sum() > 0

    def test_idempotent_render(self):
        """Calling render_volume() twice returns same data."""
        a = Sphere(r=5, voxel_size=VS * 2)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        if isinstance(csg, CSGModel):
            data1 = csg.render_volume()
            data2 = csg.render_volume()
            assert data1 is data2

    def test_transformed_in_csg(self):
        """Transformed primitive in CSG renders via render_on_grid(grid, M4inv)."""
        s = Sphere(r=5, voxel_size=VS * 2)
        t = s.rotate_z(45)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = t & b
        if isinstance(csg, CSGModel):
            csg.render_volume()
            assert csg.voxel_data is not None
            assert np.unpackbits(csg.voxel_data, bitorder='big').sum() > 0
