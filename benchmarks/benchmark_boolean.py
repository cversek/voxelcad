"""Benchmark: boolean operations — Tier 1 (same-grid), Tier 2 (compatible), Tier 3 (CSG).

Tier 1: byte-level bitwise ops on pre-rendered packed arrays (same grid).
Tier 2: render_on_grid(union) + byte-level ops (compatible voxel_size).
Tier 3: CSG per-slice streaming (incompatible voxel_size).
"""
import numpy as np
from super_utils.benchmarks import BenchmarkBase
from voxelcad import Cube, Sphere, GyroidCube
from voxelcad.voxel_model import VoxelModel, CSGModel

RESOLUTIONS = {
    "small": 64,
    "medium": 256,
    "large": 1024,
}


class BenchmarkSameGridUnion(BenchmarkBase):
    name = "same_grid_union"
    description = "Cube | GyroidCube (byte-level bitwise_or)"
    workload_type = "memory-bound"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.a = Cube(size=10, voxel_size=vs, center=True)
        self.b = GyroidCube(size=10, voxel_size=vs, center=True)
        self.a.render_volume()
        self.b.render_volume()

    def run(self):
        self.result = self.a | self.b

    def validate(self):
        return (type(self.result) is VoxelModel and
                self.result.voxel_data is not None)


class BenchmarkSameGridIntersection(BenchmarkBase):
    name = "same_grid_intersection"
    description = "Cube & GyroidCube (byte-level bitwise_and)"
    workload_type = "memory-bound"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.a = Cube(size=10, voxel_size=vs, center=True)
        self.b = GyroidCube(size=10, voxel_size=vs, center=True)
        self.a.render_volume()
        self.b.render_volume()

    def run(self):
        self.result = self.a & self.b

    def validate(self):
        return (type(self.result) is VoxelModel and
                self.result.voxel_data.sum() > 0)


class BenchmarkCSGUnionRender(BenchmarkBase):
    name = "csg_union_render"
    description = "(Cube | Sphere).render_volume() — incompatible grids, CSG path"
    workload_type = "mixed"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.a = Cube(size=10, voxel_size=vs, center=True)
        self.b = Sphere(r=4, voxel_size=vs * 2)  # 2x coarser → Tier 3
        self.csg = self.a | self.b
        assert type(self.csg) is CSGModel

    def run(self):
        self.csg.voxel_data = None
        self.csg.render_volume()

    def validate(self):
        return self.csg.voxel_data is not None and self.csg.voxel_data.sum() > 0


class BenchmarkCSGDepth4Render(BenchmarkBase):
    name = "csg_depth4_render"
    description = "((A|B)&(C|D)).render_volume() — 4-primitive CSG tree"
    workload_type = "mixed"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        # Use incompatible voxel sizes to force Tier 3 CSG throughout
        a = Cube(size=10, voxel_size=vs, center=True)
        b = Sphere(r=4, voxel_size=vs * 2)
        c = Sphere(r=3, voxel_size=vs * 3)
        d = Sphere(r=5, voxel_size=vs * 1.5)
        self.tree = (a | b) & (c | d)

    def run(self):
        self.tree.voxel_data = None
        self.tree.render_volume()

    def validate(self):
        return self.tree.voxel_data is not None
