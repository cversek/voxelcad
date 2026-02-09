"""Benchmark: fallback-breaking scenarios and Tier 2 gap closure.

Phase 9 showed a 19-35x gap between manual composition and CSG path
for compatible-grid operands. Phase 10.1 Tier 2 closes this gap.
These benchmarks measure both the closed gap and remaining CSG overhead.
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


class BenchmarkTier2CompatibleGrid(BenchmarkBase):
    """Tier 2: compatible-grid intersection (was 19-35x gap, now closed)."""
    name = "tier2_compatible_grid"
    description = "(Sphere & GyroidCube) — Tier 2 compatible grid path"
    workload_type = "mixed"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        self.a = Sphere(r=4, voxel_size=vs)
        self.b = GyroidCube(size=10, voxel_size=vs, center=True)

    def run(self):
        self.result = self.a & self.b

    def validate(self):
        return (type(self.result) is VoxelModel and
                self.result.voxel_data is not None)


class BenchmarkTransformedInCSG(BenchmarkBase):
    """TransformedModel in CSG tree — incompatible grids force Tier 3."""
    name = "transformed_in_csg"
    description = "Cube.rotate(45) & Sphere(vs*2) — transform + CSG overhead"
    workload_type = "mixed"

    def setup(self):
        res = RESOLUTIONS[self.size]
        vs = 10.0 / res
        cube = Cube(size=10, voxel_size=vs, center=True)
        self.rotated = cube.rotate_z(45)
        self.sphere = Sphere(r=4, voxel_size=vs * 2)  # incompatible → Tier 3
        self.csg = self.rotated & self.sphere
        assert type(self.csg) is CSGModel

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
