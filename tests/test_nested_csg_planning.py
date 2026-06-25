"""Regression tests for nested CSG operand indexing (upstream issue #3).

A boolean operation nested inside another op must reference the result of
the inner op, not a stale leaf at the same numeric index. The planner's
result_idx is computed during recursive collection, so it must remain valid
even as later sibling subtrees keep appending leaves.

Failure mode prior to fix:
    For `core | (gc & cyl)` where core is itself a CSGModel (e.g. `cone | ext`),
    the inner `or` reports result_idx = 2 at emit time (2 leaves, 0 ops). After
    the sibling subtree adds two more leaves (gc at 2, cyl at 3), stack[2] is
    the gc leaf, not the inner OR result. The outer op then computes
    `gc | (gc & cyl) == gc` and the intersection vanishes.
"""
import numpy as np
import pytest

from voxelcad import Cube, Sphere
from voxelcad.voxel_model import CSGModel
from tests.conftest import LOW_RES_VS


# ---------------------------------------------------------------------------
# Planner-level: operand indices in the emitted execution plan
# ---------------------------------------------------------------------------

def test_plan_op_indices_valid_for_simple_nested_or_and():
    """`(A | B) | (C & D)` — outer op must reference inner OR's result,
    not stack[2] (which is leaf C after sibling collection)."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    c = Cube(size=8, voxel_size=LOW_RES_VS, center=True)
    d = Sphere(r=3, voxel_size=LOW_RES_VS)

    tree = (a | b) | (c & d)
    plan = tree._plan_execution()
    n = len(plan.leaves)
    assert n == 4

    # Each operand index in every op must refer either to a valid leaf
    # OR to the result of an EARLIER op in the plan. With the post-fix
    # encoding (>=0 → leaf, <0 → op result), validate that for both.
    for op_pos, (_op, left_ref, right_ref) in enumerate(plan.operations):
        for ref in (left_ref, right_ref):
            if ref >= 0:
                assert 0 <= ref < n, f"leaf index {ref} out of range (n={n})"
            else:
                op_idx = -ref - 1
                assert 0 <= op_idx < op_pos, (
                    f"op result reference -{op_idx + 1} forward-references "
                    f"op {op_idx} from op position {op_pos}"
                )


def test_plan_op_indices_valid_left_csg_right_leaf():
    """`(A | B) & C` — outer op left operand must reference inner OR,
    not leaf C which sits at the same slot as the stale result_idx."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    c = Sphere(r=3, voxel_size=LOW_RES_VS)

    tree = (a | b) & c
    plan = tree._plan_execution()
    assert len(plan.leaves) == 3
    assert len(plan.operations) == 2

    outer = plan.operations[1]
    # Left operand of outer op must be the result of operation 0
    # (encoded as -1 in the post-fix scheme), not leaf index 2 (c).
    assert outer[1] < 0, (
        f"outer op left operand should reference inner OR (op result), "
        f"got leaf index {outer[1]}"
    )


def test_plan_op_indices_valid_left_leaf_right_csg():
    """`A | (B & C)` — outer op right operand must reference inner AND."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    c = Sphere(r=3, voxel_size=LOW_RES_VS)

    tree = a | (b & c)
    plan = tree._plan_execution()
    assert len(plan.leaves) == 3
    outer = plan.operations[-1]
    assert outer[2] < 0, (
        f"outer op right operand should reference inner AND, "
        f"got leaf index {outer[2]}"
    )


# ---------------------------------------------------------------------------
# Behavioural: rendering correctness after nested ops
# ---------------------------------------------------------------------------

def _sum_packed(model):
    model.render_volume()
    return int(np.unpackbits(model.voxel_data).sum())


def test_nested_or_with_intersection_preserves_clip():
    """`core | (gc & cyl)` must produce the same `gc & cyl` clip as standalone.

    The clipped region's voxel count, contributed to the union, must be at
    most the standalone clip's count (a union can only grow by extra
    voxels from `core`).
    """
    # Use disjoint-ish geometry so clip != gyroid
    gc = Cube(size=10, voxel_size=LOW_RES_VS, center=True)  # bbox = 10
    cyl = Sphere(r=3, voxel_size=LOW_RES_VS)                # tight clip
    core = Sphere(r=1, voxel_size=LOW_RES_VS)               # small extra

    standalone = gc & cyl
    standalone_count = _sum_packed(standalone)

    # Rebuild fresh trees (rendering mutates internal state).
    gc = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    cyl = Sphere(r=3, voxel_size=LOW_RES_VS)
    core = Sphere(r=1, voxel_size=LOW_RES_VS)
    nested = core | (gc & cyl)
    nested_count = _sum_packed(nested)

    # If the bug were present, nested would render as ~gc (the full cube),
    # giving a count near 32**3 ≈ 32k. With the fix, nested ≈ clip + maybe
    # a few extra voxels from `core` (which is fully contained in cyl).
    gc_full = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    gc_count = _sum_packed(gc_full)

    assert nested_count < gc_count, (
        f"nested union of intersection ({nested_count}) reached full cube "
        f"count ({gc_count}) — inner clip was dropped (issue #3 regression)"
    )
    # The union should contain at least the standalone clip's voxels.
    assert nested_count >= standalone_count
    # And cannot exceed clip + core's voxel count.
    core_count = _sum_packed(Sphere(r=1, voxel_size=LOW_RES_VS))
    assert nested_count <= standalone_count + core_count + 1


def test_nested_intersection_with_union_preserves_clip():
    """`(core | extra) & cyl` — left subtree is a union, must not be dropped."""
    cyl = Sphere(r=3, voxel_size=LOW_RES_VS)
    core = Cube(size=8, voxel_size=LOW_RES_VS, center=True)
    extra = Sphere(r=1, voxel_size=LOW_RES_VS)

    # standalone: core & cyl
    standalone = Cube(size=8, voxel_size=LOW_RES_VS, center=True) & Sphere(r=3, voxel_size=LOW_RES_VS)
    standalone_count = _sum_packed(standalone)

    nested = (core | extra) & cyl
    nested_count = _sum_packed(nested)

    # With the bug: ('and', 2, 2) → C & C = C (just the sphere). With fix:
    # (core | extra) & cyl — at least as many voxels as standalone (since
    # core ⊆ (core ∪ extra)), but bounded above by cyl's count.
    cyl_count = _sum_packed(Sphere(r=3, voxel_size=LOW_RES_VS))
    assert nested_count >= standalone_count, (
        f"(core | extra) & cyl ({nested_count}) lost the union — got fewer "
        f"voxels than (core & cyl) standalone ({standalone_count})"
    )
    assert nested_count <= cyl_count, (
        f"(core | extra) & cyl ({nested_count}) exceeded cyl count "
        f"({cyl_count}) — clip not applied"
    )


def test_deeply_nested_alternating_ops():
    """`((A | B) & (C | D)) | E` — four leaves on the left subtree, then sibling."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    c = Cube(size=8, voxel_size=LOW_RES_VS, center=True)
    d = Sphere(r=3, voxel_size=LOW_RES_VS)
    e = Sphere(r=2, voxel_size=LOW_RES_VS)

    tree = ((a | b) & (c | d)) | e
    # Just rendering without crashing or producing all-zero output proves
    # the indices resolve to live results, not stale leaves.
    tree.render_volume()
    assert tree.voxel_data is not None
    nonzero = int(np.unpackbits(tree.voxel_data).sum())
    assert nonzero > 0


def test_workaround_translate_matches_direct_form():
    """The historic translate([0,0,0]) workaround should now equal the
    direct form (since the bug it papered over is fixed)."""
    a = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b = Sphere(r=4, voxel_size=LOW_RES_VS)
    c = Sphere(r=3, voxel_size=LOW_RES_VS)
    core = Sphere(r=1, voxel_size=LOW_RES_VS)

    direct = core | (a & b)
    direct_count = _sum_packed(direct)

    # Rebuild for the workaround form
    a2 = Cube(size=10, voxel_size=LOW_RES_VS, center=True)
    b2 = Sphere(r=4, voxel_size=LOW_RES_VS)
    core2 = Sphere(r=1, voxel_size=LOW_RES_VS)
    workaround = core2.translate([0, 0, 0]) | (a2 & b2)
    workaround_count = _sum_packed(workaround)

    # Allow at most 1 voxel difference from boundary rounding.
    assert abs(direct_count - workaround_count) <= 1, (
        f"direct form ({direct_count}) and translate-workaround "
        f"({workaround_count}) disagree — inner intersection still being dropped"
    )
