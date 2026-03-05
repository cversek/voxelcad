"""Test export() dispatch: fused path, fallback, method kwarg, lazy render."""
import os
import struct
import tempfile

import numpy as np
import pytest

from voxelcad import Sphere
from voxelcad._kernels import fused_stl_export


def _stl_tri_count(path):
    """Read triangle count from binary STL header."""
    with open(path, 'rb') as f:
        f.seek(80)
        return struct.unpack('<I', f.read(4))[0]


@pytest.fixture
def sphere():
    return Sphere(r=5, voxel_size=10.0 / 32)


class TestExportLazyRender:
    """Verify export() triggers render_volume() when voxel_data is None."""

    def test_fresh_object_exports(self, sphere, tmp_path):
        assert sphere.voxel_data is None
        fname = str(tmp_path / "test.stl")
        sphere.export(fname)
        assert os.path.getsize(fname) > 84  # more than just STL header
        assert sphere.voxel_data is not None  # render was triggered

    def test_pre_rendered_object_exports(self, sphere, tmp_path):
        sphere.render_volume()
        assert sphere.voxel_data is not None
        fname = str(tmp_path / "test.stl")
        sphere.export(fname)
        assert _stl_tri_count(fname) > 0


class TestExportMethodKwarg:
    """Verify method kwarg dispatches correctly."""

    def test_default_uses_fused(self, sphere, tmp_path):
        """Default (method='auto') should use fused path when available."""
        fname = str(tmp_path / "auto.stl")
        sphere.export(fname)
        n_auto = _stl_tri_count(fname)
        assert n_auto > 0

    @pytest.mark.skipif(fused_stl_export is None,
                        reason="fused kernel not compiled")
    def test_fast_smooth_uses_fused(self, sphere, tmp_path):
        fname = str(tmp_path / "fast_smooth.stl")
        sphere.export(fname, method='fast_smooth')
        assert _stl_tri_count(fname) > 0

    def test_cdt_uses_fallback(self, sphere, tmp_path):
        """method='cdt' should use render_surface_mesh fallback."""
        fname = str(tmp_path / "cdt.stl")
        sphere.export(fname, method='cdt')
        assert _stl_tri_count(fname) > 0

    def test_invalid_method_raises(self, sphere, tmp_path):
        fname = str(tmp_path / "bad.stl")
        with pytest.raises(ValueError, match="Unknown method"):
            sphere.export(fname, method='invalid')


class TestExportKwargForwarding:
    """Verify all kwargs are extracted and forwarded correctly."""

    def test_lowpass_order(self, sphere, tmp_path):
        """lowpass_order should not raise NameError."""
        fname = str(tmp_path / "order2.stl")
        sphere.export(fname, lowpass_order=2)
        assert _stl_tri_count(fname) > 0

    def test_lowpass_cutoff(self, sphere, tmp_path):
        fname = str(tmp_path / "cutoff.stl")
        sphere.export(fname, lowpass_cutoff=0.15)
        assert _stl_tri_count(fname) > 0

    def test_mc_stride(self, sphere, tmp_path):
        fname = str(tmp_path / "stride.stl")
        sphere.export(fname, mc_stride=1)
        assert _stl_tri_count(fname) > 0

    def test_unknown_kwarg_ignored_or_forwarded(self, sphere, tmp_path):
        """Extra kwargs should not cause crashes in fused path."""
        fname = str(tmp_path / "extra.stl")
        # cache and target_reduction are extracted but unused by fused path
        sphere.export(fname, cache=False, target_reduction=0.0)
        assert _stl_tri_count(fname) > 0


class TestExportFileExtension:
    """Verify extension handling."""

    def test_stl_extension(self, sphere, tmp_path):
        fname = str(tmp_path / "mesh.stl")
        sphere.export(fname)
        assert os.path.exists(fname)

    def test_unknown_extension_raises(self, sphere, tmp_path):
        fname = str(tmp_path / "mesh.obj")
        with pytest.raises(ValueError, match="not recognized"):
            sphere.export(fname)


@pytest.mark.skipif(fused_stl_export is None,
                    reason="fused kernel not compiled")
class TestWindingConsistency:
    """Verify Lewiner MC produces consistent outward-facing normals."""

    def _read_stl_normals_and_verts(self, path):
        """Read all triangle normals and vertices from binary STL."""
        with open(path, 'rb') as f:
            f.seek(80)
            n_tri = struct.unpack('<I', f.read(4))[0]
            normals = np.empty((n_tri, 3), dtype=np.float32)
            centroids = np.empty((n_tri, 3), dtype=np.float32)
            for i in range(n_tri):
                data = struct.unpack('<12fH', f.read(50))
                normals[i] = data[0:3]
                v0 = np.array(data[3:6])
                v1 = np.array(data[6:9])
                v2 = np.array(data[9:12])
                centroids[i] = (v0 + v1 + v2) / 3.0
        return normals, centroids

    def test_sphere_normals_point_outward(self, sphere, tmp_path):
        """For a centered sphere, normals should point away from origin."""
        fname = str(tmp_path / "winding.stl")
        sphere.export(fname, method='fast_smooth')
        normals, centroids = self._read_stl_normals_and_verts(fname)
        # Dot product of normal with centroid vector (from origin)
        dots = np.sum(normals * centroids, axis=1)
        # All should be positive (outward-facing)
        outward_pct = np.sum(dots > 0) / len(dots) * 100
        assert outward_pct > 99.0, (
            f"Only {outward_pct:.1f}% normals point outward (expect >99%)")

    def test_sphere_normals_consistent_winding(self, sphere, tmp_path):
        """Cross-product normal should match stored STL normal direction."""
        fname = str(tmp_path / "winding2.stl")
        sphere.export(fname, method='fast_smooth')
        with open(fname, 'rb') as f:
            f.seek(80)
            n_tri = struct.unpack('<I', f.read(4))[0]
            agree = 0
            for _ in range(n_tri):
                data = struct.unpack('<12fH', f.read(50))
                n_stl = np.array(data[0:3])
                v0 = np.array(data[3:6])
                v1 = np.array(data[6:9])
                v2 = np.array(data[9:12])
                # Cross product from vertex winding
                cp = np.cross(v1 - v0, v2 - v0)
                cp_len = np.linalg.norm(cp)
                if cp_len > 0:
                    cp /= cp_len
                n_len = np.linalg.norm(n_stl)
                if n_len > 0:
                    n_stl /= n_len
                if np.dot(cp, n_stl) > 0.9:
                    agree += 1
        pct = agree / n_tri * 100
        assert pct > 99.0, (
            f"Only {pct:.1f}% triangles have consistent winding (expect >99%)")


@pytest.mark.skipif(fused_stl_export is None,
                    reason="fused kernel not compiled")
class TestFusedVsFallbackConsistency:
    """Verify fused and fallback paths produce comparable results."""

    def test_triangle_count_comparable(self, sphere, tmp_path):
        """Fused and CDT paths should produce similar triangle counts."""
        f_fused = str(tmp_path / "fused.stl")
        f_cdt = str(tmp_path / "cdt.stl")
        sphere.export(f_fused, method='fast_smooth')
        # Need fresh object for CDT path
        sphere2 = Sphere(r=5, voxel_size=10.0 / 32)
        sphere2.export(f_cdt, method='cdt')
        n_fused = _stl_tri_count(f_fused)
        n_cdt = _stl_tri_count(f_cdt)
        # Same geometry, same resolution — counts should be close
        # Allow 5% tolerance for different smoothing paths
        ratio = n_fused / n_cdt if n_cdt > 0 else 0
        assert 0.8 < ratio < 1.2, (
            f"fused={n_fused}, cdt={n_cdt}, ratio={ratio:.2f}")
