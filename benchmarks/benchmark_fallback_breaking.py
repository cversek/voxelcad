"""Benchmark: fallback-breaking scenarios that expose non-composability.

These benchmarks measure the gap between what the optimizer COULD do
(manual composition baseline) and what it actually does (CSGModel fallback).
Phase 10 should close these gaps.
"""
import numpy as np
from super_utils.benchmarks import BenchmarkBase
from voxelcad import Cube, Sphere, GyroidCube
from voxelcad.voxel_model import VoxelModel, CSGModel

RESOLUTIONS = {
    "small": 64,
    "medium": 256,
    "large": 512,  # 1024 too slow for CSG fallback in CI
}


class BenchmarkManualComposition(BenchmarkBase):
    """Performance ceiling: manual render + bitwise_and."""
    name = "manual_composition_baseline"
    description = "Manual: A.render(); B.render(); np.bitwise_and() — ceiling"
    workload_type = "memory-bound"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.a = Cube(size=10, voxel_size=vs, center=True)
        self.b = GyroidCube(size=10, voxel_size=vs, center=True)

    def run(self):
        self.a.voxel_data = None
        self.b.voxel_data = None
        self.a.render_volume()
        self.b.render_volume()
        self.result = np.bitwise_and(self.a.voxel_data, self.b.voxel_data)

    def validate(self):
        return self.result is not None and self.result.sum() > 0


class BenchmarkSameGridInCSG(BenchmarkBase):
    """Same-grid primitives forced through CSG path (different bbox)."""
    name = "same_grid_in_csg"
    description = "(Sphere & GyroidCube).render_volume() — same voxel_size, CSG path"
    workload_type = "mixed"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        # Sphere(r=4) has bbox [-4,4], GyroidCube(size=10) has bbox [-5,5]
        # Same voxel_size but different bbox → CSGModel (not fast path)
        self.a = Sphere(r=4, voxel_size=vs)
        self.b = GyroidCube(size=10, voxel_size=vs, center=True)
        self.csg = self.a & self.b
        assert type(self.csg) is CSGModel, "Expected CSGModel for different-bbox operands"

    def run(self):
        self.csg.voxel_data = None
        self.csg.render_volume()

    def validate(self):
        return self.csg.voxel_data is not None


class BenchmarkTransformedInCSG(BenchmarkBase):
    """TransformedModel in CSG tree — rotation forces source unpack."""
    name = "transformed_in_csg"
    description = "Cube.rotate(45) & Sphere — transform + CSG overhead"
    workload_type = "mixed"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        cube = Cube(size=10, voxel_size=vs, center=True)
        self.rotated = cube.rotate_z(45)
        self.sphere = Sphere(r=4, voxel_size=vs)
        self.csg = self.rotated & self.sphere

    def run(self):
        self.csg.voxel_data = None
        self.rotated.voxel_data = None
        self.rotated.source.voxel_data = None
        self.csg.render_volume()

    def validate(self):
        return self.csg.voxel_data is not None


class BenchmarkTransformOverhead(BenchmarkBase):
    """Isolated transform overhead: rotated render vs direct render."""
    name = "transform_overhead"
    description = "GyroidCube.rotate(30).render_volume() — transform cost"
    workload_type = "mixed"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        base = GyroidCube(size=10, voxel_size=vs, center=True)
        self.rotated = base.rotate_z(30)

    def run(self):
        self.rotated.voxel_data = None
        self.rotated.source.voxel_data = None
        self.rotated.render_volume()

    def validate(self):
        return self.rotated.voxel_data is not None
