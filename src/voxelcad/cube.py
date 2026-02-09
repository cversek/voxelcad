import numpy as np

import voxelcad.environment as ENV

from voxelcad.voxel_model import VoxelModel
from voxelcad.voxel_grid  import VoxelGrid
from voxelcad._kernels import CYTHON_AVAILABLE, evaluate_and_pack_cube

from voxelcad.debug import currentframe, DEBUG_TAG, DEBUG_EMBED, TIMING_START, TIMING_END, MEMORY_USAGE

class Cube(VoxelModel):
    def __init__(self, size, voxel_size=None, center=False, **kwargs):
        super().__init__(**kwargs)
        self.size = np.array(size)*np.ones(3)
        #set up grid dimensions
        if center:
            sx,sy,sz = self.size/2
            self.grid = VoxelGrid(xlim=(-sx,sx),
                                  ylim=(-sy,sy),
                                  zlim=(-sz,sz),
                                  voxel_size=voxel_size)
        else:
            sx,sy,sz = self.size
            self.grid = VoxelGrid(xlim=(0,sx),
                                  ylim=(0,sy),
                                  zlim=(0,sz),
                                  voxel_size=voxel_size)

    def evaluate_slice(self, X_2d, Y_2d, z_val):
        sx,sy,sz = self.size
        cx,cy,cz = self.grid.compute_center_vector()
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return (np.abs(X_2d-cx) <= sx/2) &\
               (np.abs(Y_2d-cy) <= sy/2) &\
               (np.abs(z_val-cz) <= sz/2)

    def evaluate_at_coords(self, X, Y, Z):
        """Evaluate cube geometry at arbitrary coordinates."""
        sx,sy,sz = self.size
        cx,cy,cz = self.grid.compute_center_vector()
        return (np.abs(X-cx) <= sx/2) & (np.abs(Y-cy) <= sy/2) & (np.abs(Z-cz) <= sz/2)

    def _is_fused_capable(self):
        """Cube has a Cython fused kernel when available."""
        return CYTHON_AVAILABLE and evaluate_and_pack_cube is not None

    def render_volume(self):
        if not CYTHON_AVAILABLE:
            return super().render_volume()
        TIMING_START("render_volume")
        xcc, ycc, zcc = self.grid.compute_cell_center_ranges()
        cx, cy, cz = self.grid.compute_center_vector()
        sx, sy, sz = self.size
        self.voxel_data = evaluate_and_pack_cube(
            xcc, ycc, zcc, cx, cy, cz,
            sx / 2.0, sy / 2.0, sz / 2.0,
        )
        self._voxel_shape = (len(xcc), len(ycc), len(zcc))
        TIMING_END("render_volume")
        return self.voxel_data

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = Cube(10,res=32)
    M.plot(show=True)
    M.export("test_model_cube10.stl")
