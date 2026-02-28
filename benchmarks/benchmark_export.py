"""Benchmark: full STL export pipeline — meshfix vs EDT comparison.

Measures per-stage timing breakdown to quantify where export time is spent.
Compares legacy meshfix pipeline against the EDT + Butterworth + MC pipeline.
"""
import os
import time
import tempfile
import numpy as np
import psutil
from super_utils.benchmarks import BenchmarkBase
from voxelcad import Sphere, GyroidCube
from voxelcad._kernels import CYTHON_AVAILABLE

RESOLUTIONS = {
    "small": 64,
    "medium": 128,
    "large": 200,
}


def _rss_mb():
    """Current RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


class BenchmarkSphereExport(BenchmarkBase):
    name = "sphere_export"
    description = "Sphere: full STL export pipeline"
    workload_type = "mixed"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.model = Sphere(r=5, voxel_size=vs)
        self._stl_path = os.path.join(tempfile.gettempdir(), "bench_sphere.stl")
        self.stage_times = {}

    def run(self):
        # Stage 1: Voxel rendering
        self.model.voxel_data = None
        t0 = time.perf_counter()
        self.model.render_volume()
        t1 = time.perf_counter()
        self.stage_times['render'] = t1 - t0

        # Stage 2: Uniform grid + volume mesh (marching cubes)
        import pyvista as pv
        self.model.pv_grid = None
        self.model.pv_vol = None
        pv_grid = self.model.render_uniform_grid()
        t2 = time.perf_counter()
        self.stage_times['uniform_grid'] = t2 - t1

        pv_vol = pv_grid.threshold(0.5)
        t3 = time.perf_counter()
        self.stage_times['threshold'] = t3 - t2

        # Stage 3: Surface extraction
        pv_surf = pv_vol.extract_surface(algorithm='dataset_surface')
        t4 = time.perf_counter()
        self.stage_times['extract_surface'] = t4 - t3

        # Stage 4: Mesh repair
        import pymeshfix as mf
        meshfix = mf.MeshFix(pv_surf)
        meshfix.repair(joincomp=True)
        pv_surf = meshfix.mesh
        t5 = time.perf_counter()
        self.stage_times['meshfix'] = t5 - t4

        # Stage 5: Save STL
        pv_surf.save(self._stl_path)
        t6 = time.perf_counter()
        self.stage_times['save'] = t6 - t5

        self._surf = pv_surf

    def validate(self):
        return os.path.exists(self._stl_path) and os.path.getsize(self._stl_path) > 0

    def extra_results(self):
        total = sum(self.stage_times.values())
        breakdown = {}
        for stage, t in self.stage_times.items():
            breakdown[stage] = {
                "time_ms": round(t * 1000, 1),
                "pct": round(100 * t / total, 1) if total > 0 else 0,
            }
        breakdown["total_ms"] = round(total * 1000, 1)
        return {"stage_breakdown": breakdown}


class BenchmarkGyroidSphereExport(BenchmarkBase):
    name = "gyroid_sphere_export"
    description = "Gyroid & Sphere CSG: full STL export pipeline (meshfix)"
    workload_type = "mixed"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.sphere = Sphere(r=5, voxel_size=vs)
        self.gyroid = GyroidCube(12, voxel_size=vs, center=True,
                                 lattice_param=1.5, thresh1=-0.3, thresh2=0.3)
        self.model = self.sphere & self.gyroid
        self._stl_path = os.path.join(tempfile.gettempdir(), "bench_gyroid_sphere.stl")
        self.stage_times = {}

    def run(self):
        # Stage 1: CSG render (both primitives + boolean)
        t0 = time.perf_counter()
        self.model.render_volume()
        t1 = time.perf_counter()
        self.stage_times['render'] = t1 - t0

        # Stage 2: Uniform grid + volume mesh
        import pyvista as pv
        self.model.pv_grid = None
        self.model.pv_vol = None
        pv_grid = self.model.render_uniform_grid()
        t2 = time.perf_counter()
        self.stage_times['uniform_grid'] = t2 - t1

        pv_vol = pv_grid.threshold(0.5)
        t3 = time.perf_counter()
        self.stage_times['threshold'] = t3 - t2

        # Stage 3: Surface extraction
        pv_surf = pv_vol.extract_surface(algorithm='dataset_surface')
        t4 = time.perf_counter()
        self.stage_times['extract_surface'] = t4 - t3

        # Stage 4: Mesh repair
        import pymeshfix as mf
        meshfix = mf.MeshFix(pv_surf)
        meshfix.repair(joincomp=True)
        pv_surf = meshfix.mesh
        t5 = time.perf_counter()
        self.stage_times['meshfix'] = t5 - t4

        # Stage 5: Save STL
        pv_surf.save(self._stl_path)
        t6 = time.perf_counter()
        self.stage_times['save'] = t6 - t5

        self._surf = pv_surf

    def validate(self):
        return os.path.exists(self._stl_path) and os.path.getsize(self._stl_path) > 0

    def extra_results(self):
        total = sum(self.stage_times.values())
        breakdown = {}
        for stage, t in self.stage_times.items():
            breakdown[stage] = {
                "time_ms": round(t * 1000, 1),
                "pct": round(100 * t / total, 1) if total > 0 else 0,
            }
        breakdown["total_ms"] = round(total * 1000, 1)
        return {"stage_breakdown": breakdown}


# ─── EDT Pipeline Benchmarks ───────────────────────────────────────────


class _EDTBenchmarkMixin:
    """Shared EDT pipeline logic for Sphere and GyroidSphere benchmarks."""

    def _run_edt_pipeline(self, model):
        from scipy.ndimage import distance_transform_edt as edt
        from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq
        from voxelcad.voxel_grid import UniformGrid

        self.stage_times = {}
        self.mem_stages = {}
        mem0 = _rss_mb()

        # Stage 1: Voxel rendering
        model.voxel_data = None
        model.pv_surf = None
        t0 = time.perf_counter()
        model.render_volume()
        t1 = time.perf_counter()
        self.stage_times['render'] = t1 - t0

        # Stage 2: SDF (EDT interior - EDT exterior)
        V = model._unpack_volume()
        self.mem_stages['after_unpack'] = _rss_mb() - mem0
        t2 = time.perf_counter()
        dist = (edt(V) - edt(~V)).astype(np.float32)
        del V
        t3 = time.perf_counter()
        self.stage_times['sdf'] = t3 - t2
        self.mem_stages['after_sdf'] = _rss_mb() - mem0

        # Stage 3: Butterworth order-4 low-pass filter
        rx, ry, rz = dist.shape
        fx = fftfreq(rx)
        fy = fftfreq(ry)
        fz = rfftfreq(rz)
        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
        freq_mag = np.sqrt(FX**2 + FY**2 + FZ**2)
        H = (1.0 / (1.0 + (freq_mag / 0.25)**8)).astype(np.float32)
        del FX, FY, FZ, freq_mag
        D_fft = rfftn(dist)
        del dist
        D_fft *= H
        del H
        dist = irfftn(D_fft, s=(rx, ry, rz))
        del D_fft
        t4 = time.perf_counter()
        self.stage_times['fft_filter'] = t4 - t3
        self.mem_stages['after_fft'] = _rss_mb() - mem0

        # Stage 4: Marching cubes via VTK contour
        ugrid = UniformGrid()
        ugrid.dimensions = (rx, ry, rz)
        vs = model.grid.voxel_size_vector
        ugrid.spacing = vs
        ugrid.point_data['dist'] = dist.ravel(order='F').astype(np.float32)
        del dist
        pv_surf = ugrid.contour([0.0], scalars='dist')
        del ugrid
        if pv_surf.n_points > 0:
            pv_surf = pv_surf.extract_largest()
        t5 = time.perf_counter()
        self.stage_times['marching_cubes'] = t5 - t4

        # Stage 5: Save STL
        pv_surf.save(self._stl_path)
        t6 = time.perf_counter()
        self.stage_times['save'] = t6 - t5
        self.mem_stages['peak'] = _rss_mb() - mem0

        self._surf = pv_surf
        self._n_tris = pv_surf.n_cells

    def _edt_extra_results(self):
        total = sum(self.stage_times.values())
        breakdown = {}
        for stage, t in self.stage_times.items():
            breakdown[stage] = {
                "time_ms": round(t * 1000, 1),
                "pct": round(100 * t / total, 1) if total > 0 else 0,
            }
        breakdown["total_ms"] = round(total * 1000, 1)
        breakdown["triangles"] = self._n_tris
        breakdown["memory_mb"] = self.mem_stages
        return {"stage_breakdown": breakdown}


class BenchmarkSphereEDT(_EDTBenchmarkMixin, BenchmarkBase):
    name = "sphere_edt_export"
    description = "Sphere: EDT + Butterworth + MC pipeline"
    workload_type = "mixed"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.model = Sphere(r=5, voxel_size=vs)
        self._stl_path = os.path.join(tempfile.gettempdir(), "bench_sphere_edt.stl")

    def run(self):
        self._run_edt_pipeline(self.model)

    def validate(self):
        return os.path.exists(self._stl_path) and os.path.getsize(self._stl_path) > 0

    def extra_results(self):
        return self._edt_extra_results()


class BenchmarkGyroidSphereEDT(_EDTBenchmarkMixin, BenchmarkBase):
    name = "gyroid_sphere_edt_export"
    description = "Gyroid & Sphere CSG: EDT + Butterworth + MC pipeline"
    workload_type = "mixed"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.sphere = Sphere(r=5, voxel_size=vs)
        self.gyroid = GyroidCube(12, voxel_size=vs, center=True,
                                 lattice_param=1.5, thresh1=-0.3, thresh2=0.3)
        self.model = self.sphere & self.gyroid
        self._stl_path = os.path.join(
            tempfile.gettempdir(), "bench_gyroid_sphere_edt.stl")

    def run(self):
        self._run_edt_pipeline(self.model)

    def validate(self):
        return os.path.exists(self._stl_path) and os.path.getsize(self._stl_path) > 0

    def extra_results(self):
        return self._edt_extra_results()
