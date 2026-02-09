"""Benchmark: primitive render_volume() across resolutions.

Measures per-primitive rendering time and peak memory. Cython fused kernels
vs NumPy streaming fallback is determined by CYTHON_AVAILABLE at runtime.
"""
import numpy as np
from super_utils.benchmarks import BenchmarkBase
from voxelcad import Cube, Sphere, Cylinder, GyroidCube
from voxelcad._kernels import CYTHON_AVAILABLE

RESOLUTIONS = {
    "small": 64,
    "medium": 256,
    "large": 1024,
}

PRIMITIVES = {
    "cube": lambda vs: Cube(size=10, voxel_size=vs, center=True),
    "sphere": lambda vs: Sphere(r=5, voxel_size=vs),
    "cylinder": lambda vs: Cylinder(h=10, r=5, center=True, voxel_size=vs),
    "gyroid": lambda vs: GyroidCube(size=10, voxel_size=vs, center=True),
}


class BenchmarkCubeRender(BenchmarkBase):
    name = "cube_render"
    description = "Cube.render_volume()"
    workload_type = "memory-bound"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.model = Cube(size=10, voxel_size=vs, center=True)

    def run(self):
        self.model.voxel_data = None
        self.model.render_volume()

    def validate(self):
        return self.model.voxel_data is not None and self.model.voxel_data.sum() > 0


class BenchmarkSphereRender(BenchmarkBase):
    name = "sphere_render"
    description = "Sphere.render_volume()"
    workload_type = "compute-bound"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.model = Sphere(r=5, voxel_size=vs)

    def run(self):
        self.model.voxel_data = None
        self.model.render_volume()

    def validate(self):
        return self.model.voxel_data is not None and self.model.voxel_data.sum() > 0


class BenchmarkCylinderRender(BenchmarkBase):
    name = "cylinder_render"
    description = "Cylinder.render_volume()"
    workload_type = "compute-bound"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.model = Cylinder(h=10, r=5, center=True, voxel_size=vs)

    def run(self):
        self.model.voxel_data = None
        self.model.render_volume()

    def validate(self):
        return self.model.voxel_data is not None and self.model.voxel_data.sum() > 0


class BenchmarkGyroidRender(BenchmarkBase):
    name = "gyroid_render"
    description = "GyroidCube.render_volume() — trig-heavy"
    workload_type = "compute-bound"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.model = GyroidCube(size=10, voxel_size=vs, center=True)

    def run(self):
        self.model.voxel_data = None
        self.model.render_volume()

    def validate(self):
        return self.model.voxel_data is not None and self.model.voxel_data.sum() > 0
