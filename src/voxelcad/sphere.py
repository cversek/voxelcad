import numpy as np

import voxelcad.environment as ENV

from voxelcad.voxel_model import VoxelModel
from voxelcad.voxel_grid  import VoxelGrid
from voxelcad._kernels import evaluate_and_pack_sphere

from voxelcad.debug import TIMING_START, TIMING_END

class Sphere(VoxelModel):
    def __init__(self, r, voxel_size=None, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        sv = np.array(2*r)*np.ones(3)
        sx,sy,sz = sv/2
        # Pad grid by 1 voxel on each side so SDF zero-crossing
        # never touches the grid boundary (adds 2 voxels per axis)
        vs = voxel_size if voxel_size is not None else ENV.voxel_size
        pad = float(np.atleast_1d(vs)[0])
        self.grid = VoxelGrid(xlim=(-sx-pad,sx+pad),
                              ylim=(-sy-pad,sy+pad),
                              zlim=(-sz-pad,sz+pad),
                              voxel_size=voxel_size)

    def _render_cython(self, grid, M4inv=None):
        """Cython fused evaluate-and-pack for sphere geometry."""
        if evaluate_and_pack_sphere is None:
            if ENV.use_cython:
                import warnings
                warnings.warn(
                    "Sphere: Cython kernel unavailable, falling back to NumPy",
                    RuntimeWarning, stacklevel=3,
                )
            return self._render_numpy(grid, M4inv)
        TIMING_START("sphere_render_cython")
        xcc, ycc, zcc = grid.compute_cell_center_ranges()
        cx, cy, cz = self.grid.compute_center_vector()
        result = evaluate_and_pack_sphere(
            xcc, ycc, zcc, cx, cy, cz, self.r ** 2,
            M4inv=M4inv,
        )
        TIMING_END("sphere_render_cython")
        return result

    def _render_numpy(self, grid, M4inv=None):
        """NumPy per-slice geometry evaluation for sphere."""
        TIMING_START("sphere_render_numpy")
        cx, cy, cz = self.grid.compute_center_vector()
        r_sq = self.r ** 2
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
            V[:, :, k] = ((Xp-cx)**2 + (Yp-cy)**2 + (Zp-cz)**2 <= r_sq)
        result = np.packbits(V.ravel(order='F'), bitorder='big')
        TIMING_END("sphere_render_numpy")
        return result

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = Sphere(10,res=32)
    M.plot(show=True)
    M.export("test_model_sphere10.stl")
