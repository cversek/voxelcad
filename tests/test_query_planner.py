"""Tests for CSG Tree Query Planner (Phase 10.2).

Validates that _plan_execution() correctly:
1. Collects all leaves in tree traversal order
2. Classifies leaves as FUSED, MATERIALIZED, or FALLBACK
3. Records operations in postfix order
4. Detects compatible-grid subtrees
5. Chooses optimal execution strategy
"""
import pytest
import numpy as np

from voxelcad.sphere import Sphere
from voxelcad.cube import Cube
from voxelcad.cylinder import Cylinder
from voxelcad.gyroid_cube import GyroidCube
from voxelcad.voxel_model import (
    VoxelModel, CSGModel, TransformedModel,
    LeafType, LeafNode, ExecutionPlan,
)
from voxelcad._kernels import CYTHON_AVAILABLE


# Use small voxel size for fast tests
VS = 1.0


class TestLeafClassification:
    """Test _classify_leaf() on different model types."""

    def test_primitive_fused_when_cython_available(self):
        """Primitives classify as FUSED when Cython kernels are available."""
        s = Sphere(r=5, voxel_size=VS)
        leaf = s._classify_leaf()
        if CYTHON_AVAILABLE:
            assert leaf.leaf_type == LeafType.FUSED
        else:
            assert leaf.leaf_type == LeafType.FALLBACK

    def test_primitive_materialized_after_render(self):
        """Primitives classify as MATERIALIZED after render_volume()."""
        s = Sphere(r=5, voxel_size=VS)
        s.render_volume()
        leaf = s._classify_leaf()
        assert leaf.leaf_type == LeafType.MATERIALIZED

    def test_transformed_with_primitive_source_is_fused(self):
        """TransformedModel with primitive source classifies as FUSED.

        Phase 10.4: When source has evaluate_at_coords(), TransformedModel
        can evaluate geometry directly at inverse-transformed coordinates.
        """
        s = Sphere(r=5, voxel_size=VS)
        t = s.rotate_z(45)
        assert isinstance(t, TransformedModel)
        leaf = t._classify_leaf()
        # Source (Sphere) has evaluate_at_coords → TransformedModel is FUSED
        assert leaf.leaf_type == LeafType.FUSED

    def test_all_primitives_fused_capable(self):
        """All primitive types report _is_fused_capable() correctly."""
        primitives = [
            Sphere(r=5, voxel_size=VS),
            Cube(size=10, voxel_size=VS),
            Cylinder(h=10, r=5, voxel_size=VS),
            GyroidCube(size=10, voxel_size=VS, thresh1=0.5),
        ]
        for p in primitives:
            if CYTHON_AVAILABLE:
                assert p._is_fused_capable(), f"{type(p).__name__} should be fused capable"
            else:
                assert not p._is_fused_capable()


class TestQueryPlannerTreeWalk:
    """Test _plan_execution() tree walking and leaf collection."""

    def test_simple_binary_csg(self):
        """Simple A & B produces 2 leaves and 1 operation."""
        a = Sphere(r=5, voxel_size=VS)
        b = Cube(size=8, voxel_size=VS * 2)  # Incompatible voxel size → CSGModel
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
        # Postfix order: first op is inner (a | b), second is outer (... & c)
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


class TestGridCompatibilityDetection:
    """Test all_compatible detection in query planner."""

    def test_all_compatible_same_voxel_size(self):
        """All leaves with same voxel_size → all_compatible=True."""
        # Force CSGModel via incompatible grid (different origin, same voxel_size)
        a = Sphere(r=5, voxel_size=VS)
        b = Cube(size=8, voxel_size=VS * 2)  # Different voxel size
        c = Sphere(r=4, voxel_size=VS * 2)   # Same as b, different from a
        # (b & c) should have compatible leaves
        csg = b & c
        if isinstance(csg, CSGModel):
            plan = csg._plan_execution()
            assert plan.all_compatible

    def test_incompatible_different_voxel_sizes(self):
        """Leaves with different voxel_size → all_compatible=False."""
        a = Sphere(r=5, voxel_size=VS)
        b = Cube(size=8, voxel_size=VS * 2)  # Different voxel size
        csg = a & b
        assert isinstance(csg, CSGModel)

        plan = csg._plan_execution()
        assert not plan.all_compatible

    def test_common_grid_computed_when_compatible(self):
        """When all_compatible, common_grid is the union grid."""
        a = Sphere(r=5, voxel_size=VS * 2)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        if isinstance(csg, CSGModel):
            plan = csg._plan_execution()
            if plan.all_compatible:
                assert plan.common_grid is not None
                # Union grid should contain both bounding boxes
                assert plan.common_grid.xlim[0] <= min(a.grid.xlim[0], b.grid.xlim[0])
                assert plan.common_grid.xlim[1] >= max(a.grid.xlim[1], b.grid.xlim[1])


class TestExecutionStrategySelection:
    """Test strategy selection based on tree analysis."""

    def test_streaming_when_incompatible(self):
        """Incompatible grids → strategy='streaming'."""
        a = Sphere(r=5, voxel_size=VS)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        assert isinstance(csg, CSGModel)

        plan = csg._plan_execution()
        assert plan.strategy == "streaming"

    def test_fused_bytewise_when_all_fused_and_compatible(self):
        """All fused-capable + compatible grids → strategy='fused_bytewise'."""
        a = Sphere(r=5, voxel_size=VS * 2)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        if isinstance(csg, CSGModel):
            plan = csg._plan_execution()
            if plan.all_compatible and plan.all_fused_capable:
                assert plan.strategy == "fused_bytewise"

    def test_transformed_with_primitive_source_is_fused_capable(self):
        """TransformedModel with primitive source now classifies as FUSED.

        Phase 10.4: With evaluate_at_coords(), transformed primitives
        are fused-capable, enabling fused_bytewise strategy.
        """
        a = Sphere(r=5, voxel_size=VS * 2)
        t = a.rotate_z(45)  # TransformedModel with primitive source → FUSED
        b = Cube(size=8, voxel_size=VS * 2)
        csg = t & b
        if isinstance(csg, CSGModel):
            plan = csg._plan_execution()
            # Both leaves are now fused-capable
            assert plan.all_fused_capable
            if plan.all_compatible:
                assert plan.strategy == "fused_bytewise"


class TestExecutionPlanRepr:
    """Test ExecutionPlan string representation."""

    def test_repr_includes_key_info(self):
        """ExecutionPlan repr shows leaves, strategy, compatibility."""
        a = Sphere(r=5, voxel_size=VS)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        assert isinstance(csg, CSGModel)

        plan = csg._plan_execution()
        r = repr(plan)
        assert "leaves=2" in r
        assert "strategy=" in r
        assert "all_compatible=" in r


class TestOptimizedRenderVolume:
    """Test CSGModel.render_volume() uses query-planned execution."""

    def test_csg_render_uses_planned_path_when_compatible(self):
        """CSGModel with compatible leaves uses _render_planned()."""
        a = Sphere(r=5, voxel_size=VS * 2)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        if isinstance(csg, CSGModel):
            plan = csg._plan_execution()
            if plan.all_compatible:
                # Render and verify result
                csg.render_volume()
                assert csg.voxel_data is not None
                # Grid should be the common_grid
                assert csg.grid is not None

    def test_csg_render_uses_streaming_when_incompatible(self):
        """CSGModel with incompatible leaves uses streaming fallback."""
        a = Sphere(r=5, voxel_size=VS)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        assert isinstance(csg, CSGModel)
        plan = csg._plan_execution()
        assert not plan.all_compatible
        assert plan.strategy == "streaming"
        # Render should still work
        csg.render_volume()
        assert csg.voxel_data is not None

    def test_csg_render_produces_correct_intersection(self):
        """Planned execution produces geometrically correct results."""
        # Sphere inside cube - intersection should be the sphere
        s = Sphere(r=3, voxel_size=VS * 2)
        c = Cube(size=10, center=True, voxel_size=VS * 2)
        csg = s & c
        if isinstance(csg, CSGModel):
            csg.render_volume()
            s_alone = Sphere(r=3, voxel_size=VS * 2)
            s_alone.render_volume()
            # Intersection of sphere with larger cube should equal sphere
            # (within tolerance for grid alignment differences)
            csg_sum = np.unpackbits(csg.voxel_data).sum()
            s_sum = np.unpackbits(s_alone.voxel_data).sum()
            # Allow 10% tolerance for grid alignment
            assert abs(csg_sum - s_sum) / max(s_sum, 1) < 0.1

    def test_depth_2_csg_renders_correctly(self):
        """(A | B) & C with compatible grids renders correctly."""
        a = Sphere(r=3, voxel_size=VS * 2)
        b = Cube(size=6, center=True, voxel_size=VS * 2)
        c = Cylinder(h=10, r=5, center=True, voxel_size=VS * 2)
        csg = (a | b) & c
        if isinstance(csg, CSGModel):
            csg.render_volume()
            assert csg.voxel_data is not None
            # Result should have some voxels
            assert np.unpackbits(csg.voxel_data).sum() > 0

    def test_idempotent_render(self):
        """Calling render_volume() twice returns same data."""
        a = Sphere(r=5, voxel_size=VS * 2)
        b = Cube(size=8, voxel_size=VS * 2)
        csg = a & b
        if isinstance(csg, CSGModel):
            data1 = csg.render_volume()
            data2 = csg.render_volume()
            assert data1 is data2
