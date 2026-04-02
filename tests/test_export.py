"""Test export() dispatch: fused path, fallback, method kwarg, lazy render."""
import os
import struct
import tempfile

import numpy as np
import pytest

from voxelcad import Sphere, Cube, Cylinder, GyroidCube
from voxelcad._kernels import fused_stl_export, fused_mesh_export


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
        """For a sphere, normals should point away from mesh center."""
        fname = str(tmp_path / "winding.stl")
        sphere.export(fname, method='fast_smooth', compute_normals=True)
        normals, centroids = self._read_stl_normals_and_verts(fname)
        # Use mesh center — sphere grid origin is not at (0,0,0)
        mesh_center = centroids.mean(axis=0)
        radial = centroids - mesh_center
        dots = np.sum(normals * radial, axis=1)
        # All should be positive (outward-facing)
        outward_pct = np.sum(dots > 0) / len(dots) * 100
        assert outward_pct > 99.0, (
            f"Only {outward_pct:.1f}% normals point outward (expect >99%)")

    def test_sphere_normals_consistent_winding(self, sphere, tmp_path):
        """Cross-product normal should match stored STL normal direction."""
        fname = str(tmp_path / "winding2.stl")
        sphere.export(fname, method='fast_smooth', compute_normals=True)
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


def _read_stl_fast(path):
    """Read binary STL efficiently, return (normals, v0, v1, v2) as float32 arrays."""
    with open(path, 'rb') as f:
        f.read(80)
        n_tris = struct.unpack('<I', f.read(4))[0]
        raw = np.frombuffer(f.read(n_tris * 50), dtype=np.uint8).reshape(n_tris, 50)
        fl = np.frombuffer(raw[:, :48].tobytes(), dtype='<f4').reshape(n_tris, 12)
    return fl[:, 0:3], fl[:, 3:6], fl[:, 6:9], fl[:, 9:12]


def _signed_volume(v0, v1, v2):
    """Signed volume of closed mesh via divergence theorem."""
    return np.sum(v0 * np.cross(v1, v2)) / 6.0


def _radial_outward_pct(path):
    """Fraction of normals pointing away from mesh center (for convex shapes)."""
    normals, v0, v1, v2 = _read_stl_fast(path)
    centroids = (v0 + v1 + v2) / 3.0
    center = centroids.mean(axis=0)
    dots = np.sum(normals * (centroids - center), axis=1)
    return np.sum(dots > 0) / len(dots) * 100


@pytest.mark.skipif(fused_stl_export is None,
                    reason="fused kernel not compiled")
class TestWindingAllGeometries:
    """Verify winding correctness across all primitives and CSG operations."""

    VS = 10.0 / 48  # ~48^3 resolution for fast tests

    @pytest.mark.parametrize("model_fn,name", [
        (lambda vs: Sphere(r=5, voxel_size=vs), "Sphere"),
        (lambda vs: Cube(size=8, voxel_size=vs), "Cube"),
        (lambda vs: Cylinder(h=8, r=3, voxel_size=vs), "Cylinder"),
    ])
    def test_convex_radial_outward(self, model_fn, name, tmp_path):
        """Convex primitives: all normals should point away from mesh center."""
        m = model_fn(self.VS)
        fname = str(tmp_path / f"{name}.stl")
        m.export(fname, method='fast_smooth', compute_normals=True)
        pct = _radial_outward_pct(fname)
        assert pct > 99.0, f"{name}: only {pct:.1f}% outward (expect >99%)"

    @pytest.mark.parametrize("model_fn,name", [
        (lambda vs: GyroidCube(size=8, voxel_size=vs), "GyroidCube"),
        (lambda vs: Sphere(r=5, voxel_size=vs) | Cube(size=8, voxel_size=vs), "union"),
        (lambda vs: Sphere(r=6, voxel_size=vs) & Cube(size=8, center=True, voxel_size=vs), "intersect"),
        (lambda vs: Cube(size=10, center=True, voxel_size=vs) - Sphere(r=4, voxel_size=vs), "difference"),
        (lambda vs: Sphere(r=5, voxel_size=vs) ^ Cube(size=7, center=True, voxel_size=vs), "xor"),
    ])
    def test_signed_volume_positive(self, model_fn, name, tmp_path):
        """All geometries: signed volume must be positive (outward normals)."""
        m = model_fn(self.VS)
        fname = str(tmp_path / f"{name}.stl")
        m.export(fname, method='fast_smooth', compute_normals=True)
        _, v0, v1, v2 = _read_stl_fast(fname)
        vol = _signed_volume(v0, v1, v2)
        assert vol > 0, f"{name}: signed volume {vol:.1f} <= 0 (expect positive)"

    def test_stride2_sphere_outward(self, tmp_path):
        """Stride=2 should also produce correct winding."""
        s = Sphere(r=5, voxel_size=self.VS)
        fname = str(tmp_path / "sphere_s2.stl")
        s.export(fname, method='fast_smooth', mc_stride=2, compute_normals=True)
        pct = _radial_outward_pct(fname)
        assert pct > 99.0, f"Stride=2: only {pct:.1f}% outward"


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
        # Same geometry, same resolution — counts in same order of magnitude.
        # CDT (distance field) and scaled smoothing (binary indicator) produce
        # different Butterworth-smoothed contours, so allow wide tolerance.
        ratio = n_fused / n_cdt if n_cdt > 0 else 0
        assert 0.4 < ratio < 2.5, (
            f"fused={n_fused}, cdt={n_cdt}, ratio={ratio:.2f}")


@pytest.mark.skipif(fused_mesh_export is None,
                    reason="fused mesh kernel not compiled")
class TestFusedMeshExport:
    """Verify fused_mesh_export produces correct manifold meshes."""

    VS = 10.0 / 32  # ~32^3 resolution

    def _open_edge_count(self, mesh):
        edges = mesh.extract_feature_edges(
            boundary_edges=True, non_manifold_edges=True,
            feature_edges=False, manifold_edges=False)
        return edges.n_cells

    def test_sphere_manifold(self):
        """Fused mesh sphere should have 0 open edges."""
        s = Sphere(r=5, voxel_size=self.VS)
        m = s.render_surface_mesh(method='fast_smooth')
        assert m.n_points > 0
        assert m.n_cells > 0
        assert self._open_edge_count(m) == 0

    def test_gyroid_produces_mesh(self):
        """Fused mesh gyroid should produce valid mesh matching STL path."""
        g = GyroidCube(size=8, voxel_size=self.VS)
        m = g.render_surface_mesh(method='fast_smooth')
        assert m.n_points > 0
        assert m.n_cells > 0
        # Gyroid surface intersects grid boundary — open edges are
        # expected. LCC filter (#108) will remove floating fragments.

    def test_stride2(self):
        """Stride=2 should produce valid manifold mesh."""
        s = Sphere(r=5, voxel_size=self.VS)
        m = s.render_surface_mesh(method='fast_smooth', mc_stride=2)
        assert m.n_points > 0
        assert m.n_cells > 0
        assert self._open_edge_count(m) == 0

    @pytest.mark.parametrize("size,thresh1,thresh2,vs,desc", [
        (2, 1.0, None, 0.05, "solid_small_126vpp"),
        (3, 0.8, None, 0.1, "solid_mid_63vpp"),
        (10, 1.0, None, 0.1, "solid_large_63vpp"),
        (3, -0.5, 0.5, 0.1, "shell_thick_63vpp"),
        (3, -0.3, 0.3, 0.2, "shell_thin_31vpp"),
    ])
    def test_gyroid_manifold_good_params(self, size, thresh1, thresh2, vs, desc):
        """Known-good gyroid params should produce 0 open edges."""
        kw = dict(size=size, voxel_size=vs, thresh1=thresh1)
        if thresh2 is not None:
            kw['thresh2'] = thresh2
        g = GyroidCube(**kw)
        m = g.render_surface_mesh(method='fast_smooth')
        assert m.n_points > 0, f"{desc}: empty mesh"
        oe = self._open_edge_count(m)
        assert oe == 0, f"{desc}: {oe} open edges (expect 0)"

    @pytest.mark.parametrize("size,thresh1,vs,approx_oe,desc", [
        (10, 1.0, 0.3125, 3300, "20vpp_pathological"),
        (10, 1.0, 0.2, 18864, "31vpp_worst_case"),
        (3, 0.5, 0.1, 1008, "63vpp_low_thresh"),
    ])
    def test_gyroid_pathological_params(self, size, thresh1, vs, approx_oe, desc):
        """Known-pathological gyroid params — documents expected open edges.

        These configs have resolution/kernel interference that produces
        non-manifold output. The LCC filter (Task #108) will fix these.
        Until then, just verify the mesh is non-empty and open edges
        are in the expected ballpark (within 50% of known value).
        """
        g = GyroidCube(size=size, voxel_size=vs, thresh1=thresh1)
        m = g.render_surface_mesh(method='fast_smooth')
        assert m.n_points > 0, f"{desc}: empty mesh"
        oe = self._open_edge_count(m)
        assert oe > 0, f"{desc}: expected non-zero open edges, got 0"
        # Ballpark check — exact count varies with build/platform
        assert oe > approx_oe * 0.5, (
            f"{desc}: {oe} oe much less than expected ~{approx_oe}")
        assert oe < approx_oe * 1.5, (
            f"{desc}: {oe} oe much more than expected ~{approx_oe}")

    def test_only_largest_component_mesh(self):
        """only_largest_component should remove fragments from pathological gyroid."""
        g = GyroidCube(size=10, voxel_size=0.3125, thresh1=1.0)
        m_raw = g.render_surface_mesh(method='fast_smooth',
                                      only_largest_component=False, cache=False)
        m_lcc = g.render_surface_mesh(method='fast_smooth',
                                      only_largest_component=True, cache=False)
        assert m_lcc.n_points > 0
        assert m_lcc.n_cells < m_raw.n_cells, (
            f"LCC should remove fragments: {m_lcc.n_cells} >= {m_raw.n_cells}")

    def test_only_largest_component_stl(self, tmp_path):
        """STL export with only_largest_component should route through mesh+LCC."""
        g = GyroidCube(size=10, voxel_size=0.3125, thresh1=1.0)
        # Without LCC — should have open edges
        fname_raw = str(tmp_path / "raw.stl")
        g.export(fname_raw, only_largest_component=False)
        n_raw = _stl_tri_count(fname_raw)
        # With LCC — should have fewer tris (fragments removed)
        fname_lcc = str(tmp_path / "lcc.stl")
        g.export(fname_lcc, only_largest_component=True)
        n_lcc = _stl_tri_count(fname_lcc)
        assert n_lcc > 0, "LCC export produced empty STL"
        assert n_lcc < n_raw, (
            f"LCC should remove fragments: {n_lcc} >= {n_raw}")

    def test_matches_3stage_pipeline(self):
        """Fused mesh should match 3-stage pipeline output."""
        from voxelcad._kernels import fused_scale_convolve, sweep_mc_mesh
        from voxelcad.utils.spectral import compute_butterworth_kernel
        s = Sphere(r=5, voxel_size=self.VS)
        s.render_volume()
        rx, ry, rz = s.grid.res_vector
        kern = compute_butterworth_kernel(order=4, cutoff=0.25, radius=3)
        vsv = s.grid.voxel_size_vector
        # 3-stage path
        sf = fused_scale_convolve(s.voxel_data, rx, ry, rz, kern['int8'])
        sf = sf[1:-1, 1:-1, 1:-1].copy()
        v3, f3 = sweep_mc_mesh(sf, vsv[0], vsv[1], vsv[2])
        # Fused path
        vf, ff = fused_mesh_export(
            s.voxel_data, rx, ry, rz, kern['int8'],
            vsv[0], vsv[1], vsv[2])
        assert vf.shape[0] == v3.shape[0], (
            f"Vert count mismatch: fused={vf.shape[0]}, 3stage={v3.shape[0]}")
        assert ff.shape[0] == f3.shape[0], (
            f"Face count mismatch: fused={ff.shape[0]}, 3stage={f3.shape[0]}")
