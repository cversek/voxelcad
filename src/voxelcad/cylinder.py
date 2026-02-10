import numpy as np

import voxelcad.environment as ENV

from voxelcad.voxel_model import VoxelModel
from voxelcad.voxel_grid  import VoxelGrid
from voxelcad._kernels import evaluate_and_pack_cylinder

from voxelcad.debug import currentframe, DEBUG_TAG, DEBUG_EMBED, TIMING_START, TIMING_END

class Cylinder(VoxelModel):
    def __init__(self,h,r=None, r1=None, r2=None, center=False, voxel_size=None, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        if r is not None:
            assert(r1 is None and r2 is None)
            self.size_vector = np.array((2*r,2*r,h))
            self.r1 = r
            self.r2 = r
        else:
            assert(r1 is not None and r2 is not None)
            r_max = max(r1,r2)
            self.size_vector = np.array((2*r_max,2*r_max,h))
            self.r1 = r1
            self.r2 = r2
        self.center = center
        #set up grid dimensions
        # X and Y are always centered
        sv = self.size_vector
        if self.center:
            #center Z as well
            sx,sy,sz = sv/2
            self.grid = VoxelGrid(xlim=(-sx,sx),
                                  ylim=(-sy,sy),
                                  zlim=(-sz,sz),
                                  voxel_size=voxel_size)
        else:
            # start Z at zero
            sx,sy,sz = sv
            self.grid = VoxelGrid(xlim=(-sx,sx),
                                  ylim=(-sy,sy),
                                  zlim=(0,sz),
                                  voxel_size=voxel_size)

    def _render_cython(self, grid, M4inv=None):
        """Cython fused evaluate-and-pack for cylinder geometry."""
        if evaluate_and_pack_cylinder is None or M4inv is not None:
            return self._render_numpy(grid, M4inv)
        TIMING_START("cylinder_render_cython")
        xcc, ycc, zcc = grid.compute_cell_center_ranges()
        cx, cy, cz = self.grid.compute_center_vector()
        result = evaluate_and_pack_cylinder(
            xcc, ycc, zcc, cx, cy, cz,
            self.h, self.r1, self.r2,
        )
        TIMING_END("cylinder_render_cython")
        return result

    def _render_numpy(self, grid, M4inv=None):
        """NumPy per-slice geometry evaluation for cylinder."""
        TIMING_START("cylinder_render_numpy")
        cx, cy, cz = self.grid.compute_center_vector()
        h = self.h
        r1 = self.r1
        r2 = self.r2
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
            Xc = Xp - cx
            Yc = Yp - cy
            Zc = Zp - cz
            Pz = Zc/h + 0.5
            R = r1*(1.0-Pz) + r2*Pz
            V[:, :, k] = (Xc**2 + Yc**2 <= R**2) & ((0.0 <= Pz) & (Pz <= 1.0))
        result = np.packbits(V.ravel(order='F'), bitorder='big')
        TIMING_END("cylinder_render_numpy")
        return result

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = Cylinder(10,r1=5,r2=2,res=128)
