"""Benchmark: full STL export pipeline (render -> mesh -> repair -> save).

Measures per-stage timing breakdown to quantify where export time is spent.
Determines whether marching cubes / mesh repair warrant Cython optimization.
"""
import os
import time
import tempfile
import numpy as np
from super_utils.benchmarks import BenchmarkBase
from voxelcad import Sphere, GyroidCube
from voxelcad._kernels import CYTHON_AVAILABLE

RESOLUTIONS = {
    "small": 64,
    "medium": 128,
    "large": 256,
}


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
    description = "Gyroid & Sphere CSG: full STL export pipeline"
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
