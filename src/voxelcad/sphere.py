import numpy as np

import voxelcad.environment as ENV

from voxelcad.voxel_model import VoxelModel
from voxelcad.voxel_grid  import VoxelGrid
from voxelcad._kernels import CYTHON_AVAILABLE, evaluate_and_pack_sphere

from voxelcad.debug import TIMING_START, TIMING_END

class Sphere(VoxelModel):
    def __init__(self, r, voxel_size=None, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        sv = np.array(2*r)*np.ones(3)
        sx,sy,sz = sv/2
        self.grid = VoxelGrid(xlim=(-sx,sx),
                              ylim=(-sy,sy),
                              zlim=(-sz,sz),
                              voxel_size=voxel_size)

    def evaluate_slice(self, X_2d, Y_2d, z_val):
        r = self.r
        cx,cy,cz = self.grid.compute_center_vector()
        return ((X_2d-cx)**2 + (Y_2d-cy)**2 + (z_val-cz)**2 <= r**2)

    def render_volume(self):
        if not CYTHON_AVAILABLE:
            return super().render_volume()
        TIMING_START("render_volume")
        xcc, ycc, zcc = self.grid.compute_cell_center_ranges()
        cx, cy, cz = self.grid.compute_center_vector()
        self.voxel_data = evaluate_and_pack_sphere(
            xcc, ycc, zcc, cx, cy, cz, self.r ** 2,
        )
        self._voxel_shape = (len(xcc), len(ycc), len(zcc))
        TIMING_END("render_volume")
        return self.voxel_data

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = Sphere(10,res=32)
    M.plot(show=True)
    M.export("test_model_sphere10.stl")
