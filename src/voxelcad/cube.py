import numpy as np

import voxelcad.environment as ENV

from voxelcad.voxel_model import VoxelModel
from voxelcad.voxel_grid  import VoxelGrid
from voxelcad._kernels import evaluate_and_pack_cube

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

    def _render_cython(self, grid, M4inv=None):
        """Cython fused evaluate-and-pack for cube geometry."""
        if evaluate_and_pack_cube is None:
            if ENV.use_cython:
                import warnings
                warnings.warn(
                    "Cube: Cython kernel unavailable, falling back to NumPy",
                    RuntimeWarning, stacklevel=3,
                )
            return self._render_numpy(grid, M4inv)
        TIMING_START("cube_render_cython")
        xcc, ycc, zcc = grid.compute_cell_center_ranges()
        cx, cy, cz = self.grid.compute_center_vector()
        sx, sy, sz = self.size
        result = evaluate_and_pack_cube(
            xcc, ycc, zcc, cx, cy, cz,
            sx / 2.0, sy / 2.0, sz / 2.0,
            M4inv=M4inv,
        )
        TIMING_END("cube_render_cython")
        return result

    def _render_numpy(self, grid, M4inv=None):
        """NumPy per-slice geometry evaluation for cube."""
        TIMING_START("cube_render_numpy")
        cx, cy, cz = self.grid.compute_center_vector()
        sx, sy, sz = self.size
        hsx, hsy, hsz = sx/2, sy/2, sz/2
        rx, ry, rz = [int(r) for r in grid.res_vector]
        V = np.zeros((rx, ry, rz), dtype='bool')
        for X_2d, Y_2d, z_val, k in grid.iter_slices():
            if M4inv is not None:
                Z_2d = np.full_like(X_2d, z_val)
                Xp = M4inv[0,0]*X_2d + M4inv[0,1]*Y_2d + M4inv[0,2]*Z_2d + M4inv[0,3]
                Yp = M4inv[1,0]*X_2d + M4inv[1,1]*Y_2d + M4inv[1,2]*Z_2d + M4inv[1,3]
                Zp = M4inv[2,0]*X_2d + M4inv[2,1]*Y_2d + M4inv[2,2]*Z_2d + M4inv[2,3]
            else:
                Xp, Yp, Zp = X_2d, Y_2d, z_val
            V[:, :, k] = ((np.abs(Xp-cx) <= hsx) &
                          (np.abs(Yp-cy) <= hsy) &
                          (np.abs(Zp-cz) <= hsz))
        result = np.packbits(V.ravel(order='F'), bitorder='big')
        TIMING_END("cube_render_numpy")
        return result

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = Cube(10,res=32)
    M.plot(show=True)
    M.export("test_model_cube10.stl")
