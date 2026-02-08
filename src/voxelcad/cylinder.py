import numpy as np

from voxelcad.voxel_model import VoxelModel
from voxelcad.voxel_grid  import VoxelGrid
from voxelcad._kernels import CYTHON_AVAILABLE, evaluate_and_pack_cylinder

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

    def evaluate_slice(self, X_2d, Y_2d, z_val):
        cx,cy,cz = self.grid.compute_center_vector()
        Xc = X_2d - cx
        Yc = Y_2d - cy
        Zc = z_val - cz
        h = self.h
        r1 = self.r1
        r2 = self.r2
        #parameterize Z: 0 at -h/2, 1 at h/2
        Pz = Zc/h + 0.5
        #interpolate radii along Z
        R = r1*(1.0-Pz) + r2*Pz
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return (Xc**2 + Yc**2 <= R**2) & ((0.0 <= Pz) & (Pz <= 1.0))

    def render_volume(self):
        if not CYTHON_AVAILABLE:
            return super().render_volume()
        TIMING_START("render_volume")
        xcc, ycc, zcc = self.grid.compute_cell_center_ranges()
        cx, cy, cz = self.grid.compute_center_vector()
        self.voxel_data = evaluate_and_pack_cylinder(
            xcc, ycc, zcc, cx, cy, cz,
            self.h, self.r1, self.r2,
        )
        self._voxel_shape = (len(xcc), len(ycc), len(zcc))
        TIMING_END("render_volume")
        return self.voxel_data

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = Cylinder(10,r1=5,r2=2,res=128)
