"""Benchmark: STL export pipeline across resolutions.

Measures end-to-end export time including SDF computation, Butterworth
convolution, and marching cubes mesh extraction via the fused kernel path.
"""
import os
import tempfile

from super_utils.benchmarks import BenchmarkBase
from voxelcad import Sphere, GyroidCube

RESOLUTIONS = {
    "small": 64,
    "medium": 128,
    "large": 256,
}


class BenchmarkSphereExport(BenchmarkBase):
    name = "sphere_export"
    description = "Sphere.export() fused STL pipeline"
    workload_type = "memory-bound"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.model = Sphere(r=5, voxel_size=vs)
        self.model.render_volume()
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        self._tmpfile.close()

    def run(self):
        self.model.export(self._tmpfile.name)

    def validate(self):
        return os.path.getsize(self._tmpfile.name) > 0

    def teardown(self):
        try:
            os.unlink(self._tmpfile.name)
        except OSError:
            pass


class BenchmarkGyroidSphereExport(BenchmarkBase):
    name = "gyroid_sphere_export"
    description = "GyroidCube & Sphere intersection export"
    workload_type = "compute-bound"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        g = GyroidCube(size=10, voxel_size=vs, center=True)
        s = Sphere(r=4.5, voxel_size=vs)
        self.model = g & s
        self.model.render_volume()
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        self._tmpfile.close()

    def run(self):
        self.model.export(self._tmpfile.name)

    def validate(self):
        return os.path.getsize(self._tmpfile.name) > 0

    def teardown(self):
        try:
            os.unlink(self._tmpfile.name)
        except OSError:
            pass
