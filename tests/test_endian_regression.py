"""Regression tests for bit-packing endian consistency (Bug #58/#61).

Tests that Cython and NumPy rendering paths produce identical packed arrays
for both direct geometry evaluation AND transformed (M4inv) rendering.

The XOR-sum test: if two packed arrays represent the same boolean volume,
their XOR should be all zeros. Any non-zero XOR sum indicates a bit-packing
disagreement — the signature of wrong-endian access.
"""
import numpy as np
import pytest
import os
import sys

# Add src to path for in-tree testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from voxelcad import ENV
from voxelcad.sphere import Sphere
from voxelcad.cube import Cube
from voxelcad.cylinder import Cylinder


def xor_sum(packed_a, packed_b):
    """XOR two packed arrays and return the total number of differing bits."""
    xor = np.bitwise_xor(packed_a, packed_b)
    return int(np.unpackbits(xor).sum())


def render_both_paths(model, grid=None, M4inv=None):
    """Render a model via both Cython and NumPy, return both packed arrays."""
    if grid is None:
        grid = model.grid

    # Force Cython path
    ENV.use_cython = True
    cython_result = model._render_cython(grid, M4inv)

    # Force NumPy path
    ENV.use_cython = False
    numpy_result = model._render_numpy(grid, M4inv)

    # Restore default
    ENV.use_cython = True

    return cython_result, numpy_result


# ---------------------------------------------------------------------------
# Transform matrix helpers
# ---------------------------------------------------------------------------

def make_rotation_z(angle_deg):
    """Create 4x4 rotation matrix and its inverse."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    M4 = np.eye(4)
    M4[0, 0] = c;  M4[0, 1] = -s
    M4[1, 0] = s;  M4[1, 1] = c
    M4inv = np.eye(4)
    M4inv[0, 0] = c;  M4inv[0, 1] = s
    M4inv[1, 0] = -s; M4inv[1, 1] = c
    return M4, M4inv


def make_scale(sx, sy, sz):
    M4 = np.diag([sx, sy, sz, 1.0])
    M4inv = np.diag([1.0/sx, 1.0/sy, 1.0/sz, 1.0])
    return M4, M4inv


def make_translation(tx, ty, tz):
    M4 = np.eye(4)
    M4[:3, 3] = [tx, ty, tz]
    M4inv = np.eye(4)
    M4inv[:3, 3] = [-tx, -ty, -tz]
    return M4, M4inv


# ---------------------------------------------------------------------------
# Endian consistency: XOR-sum == 0 between Cython and NumPy paths
# ---------------------------------------------------------------------------

class TestEndianConsistency:
    """XOR-sum == 0 tests: Cython and NumPy must produce identical packed arrays."""

    # --- Direct geometry (no transform) ---

    def test_sphere_no_transform(self):
        s = Sphere(r=3, voxel_size=0.2)
        cy, npy = render_both_paths(s)
        diff = xor_sum(cy, npy)
        assert diff == 0, f"Sphere no-transform XOR diff = {diff}"

    def test_cube_no_transform(self):
        c = Cube(size=4, voxel_size=0.25)
        cy, npy = render_both_paths(c)
        diff = xor_sum(cy, npy)
        assert diff == 0, f"Cube no-transform XOR diff = {diff}"

    def test_cylinder_no_transform(self):
        cyl = Cylinder(r=2, h=5, voxel_size=0.25)
        cy, npy = render_both_paths(cyl)
        diff = xor_sum(cy, npy)
        assert diff == 0, f"Cylinder no-transform XOR diff = {diff}"

    # --- With M4inv transforms on primitives ---

    def test_sphere_rotate_z45(self):
        s = Sphere(r=3, voxel_size=0.2)
        _, M4inv = make_rotation_z(45)
        cy, npy = render_both_paths(s, M4inv=M4inv)
        diff = xor_sum(cy, npy)
        assert diff == 0, f"Sphere rotate Z45 XOR diff = {diff}"

    def test_cube_rotate_z30(self):
        c = Cube(size=4, voxel_size=0.25)
        _, M4inv = make_rotation_z(30)
        cy, npy = render_both_paths(c, M4inv=M4inv)
        diff = xor_sum(cy, npy)
        assert diff == 0, f"Cube rotate Z30 XOR diff = {diff}"

    def test_cylinder_scale(self):
        cyl = Cylinder(r=2, h=5, voxel_size=0.25)
        _, M4inv = make_scale(1.5, 0.5, 1.0)
        cy, npy = render_both_paths(cyl, M4inv=M4inv)
        diff = xor_sum(cy, npy)
        assert diff == 0, f"Cylinder scale XOR diff = {diff}"

    def test_sphere_translate(self):
        """Translation — the original Bug #58 trigger."""
        s = Sphere(r=3, voxel_size=0.2)
        _, M4inv = make_translation(1.0, 0.5, -0.3)
        cy, npy = render_both_paths(s, M4inv=M4inv)
        diff = xor_sum(cy, npy)
        assert diff == 0, f"Sphere translate XOR diff = {diff}"

    # --- CSG + transform (the ice cream scenario) ---

    def test_csg_leaf_consistency(self):
        """CSG leaf primitives render identically via both paths."""
        s = Sphere(r=3, voxel_size=0.25)
        cy, npy = render_both_paths(s)
        diff = xor_sum(cy, npy)
        assert diff == 0, f"CSG leaf (sphere) XOR diff = {diff}"

    def test_csg_with_transform_resampling(self):
        """Transformed CSGModel — materialize then resample_and_pack with M4inv.

        This is the ice cream transforms scenario:
        CSGModel.render_on_grid(grid, M4inv) -> render_volume() + resample_and_pack

        The Cython and NumPy paths compute M4inv coordinate transforms differently
        (scalar vs vectorized FP arithmetic), causing surface voxel disagreements.
        An endian bug would produce XOR >> 1% of total voxels (Bug #58: 63%).
        FP precision noise is << 1% (typically ~0.6% at this resolution).
        Threshold: 1% catches structural bugs while allowing FP surface noise.
        """
        s = Sphere(r=3, voxel_size=0.25)
        c = Cube(size=4, voxel_size=0.25)
        csg = s | c
        csg.render_volume()  # materialize

        _, M4inv = make_rotation_z(45)

        ENV.use_cython = True
        cy_result = csg._render_cython(csg.grid, M4inv)

        ENV.use_cython = False
        npy_result = csg._render_numpy(csg.grid, M4inv)

        ENV.use_cython = True

        rx, ry, rz = [int(r) for r in csg.grid.res_vector]
        total_voxels = rx * ry * rz
        diff = xor_sum(cy_result, npy_result)
        threshold = total_voxels * 0.01  # 1% — endian bugs are 50-60x this
        assert diff < threshold, (
            f"CSG resampling XOR diff = {diff} ({100*diff/total_voxels:.1f}% of "
            f"{total_voxels} voxels). Threshold: {threshold:.0f}. "
            f"If >> 1%, likely endian/structural bug, not FP precision."
        )

    def test_transformed_model_end_to_end(self):
        """Full TransformedModel.render_volume() — Cython vs NumPy."""
        s = Sphere(r=3, voxel_size=0.25)

        ENV.use_cython = True
        rotated_cy = s.rotate_z(45)
        cy_data = rotated_cy.render_volume()

        ENV.use_cython = False
        rotated_npy = s.rotate_z(45)
        npy_data = rotated_npy.render_volume()

        ENV.use_cython = True

        diff = xor_sum(cy_data, npy_data)
        assert diff == 0, f"TransformedModel rotate Z45 end-to-end XOR diff = {diff}"


# ---------------------------------------------------------------------------
# Pack/unpack round-trip
# ---------------------------------------------------------------------------

class TestPackUnpackRoundtrip:
    """Verify pack -> unpack round-trip preserves data."""

    def test_cython_pack_numpy_unpack(self):
        """Cython set_bit packing -> np.unpackbits should round-trip."""
        s = Sphere(r=3, voxel_size=0.2)
        ENV.use_cython = True
        packed = s._render_cython(s.grid)
        rx, ry, rz = [int(r) for r in s.grid.res_vector]
        n = rx * ry * rz
        unpacked = np.unpackbits(packed, bitorder='big')[:n]
        repacked = np.packbits(unpacked, bitorder='big')
        assert np.array_equal(packed[:len(repacked)], repacked[:len(packed)]), \
            "Cython pack -> unpack -> repack mismatch"

    def test_numpy_pack_unpack(self):
        """NumPy packbits -> unpackbits round-trip."""
        s = Sphere(r=3, voxel_size=0.2)
        ENV.use_cython = False
        packed = s._render_numpy(s.grid)
        rx, ry, rz = [int(r) for r in s.grid.res_vector]
        n = rx * ry * rz
        unpacked = np.unpackbits(packed, bitorder='big')[:n]
        repacked = np.packbits(unpacked, bitorder='big')
        ENV.use_cython = True
        assert np.array_equal(packed[:len(repacked)], repacked[:len(packed)]), \
            "NumPy pack -> unpack -> repack mismatch"
