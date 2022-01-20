import numpy as np

from voxelcad.voxel_model import VoxelModel
from voxelcad.voxel_grid  import VoxelGrid

from voxelcad.debug import currentframe, DEBUG_TAG, DEBUG_EMBED

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
                                  
    def render_volume(self):
        super().render_volume()
        cx,cy,cz  = self.grid.compute_center_vector()
        X,Y,Z,V,m = self.grid.construct_mesh()
        # fill the cylindrical volume between the margins
        Xc = X-cx; Yc = Y-cy; Zc = Z-cz
        h = self.h
        r1 = self.r1
        r2 = self.r2
        #parameterize Z
        Pz = Zc/h + 0.5  # 0 at -h/2, 1 at h/2
        #interpolate radii along Z
        R = r1*(1.0-Pz) + r2*Pz
        V[m:-m,m:-m,m:-m] = (Xc**2 + Yc**2 <= R**2) & ((0.0 <= Pz) & (Pz <= 1.0))
        self.voxel_data = V
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return self.voxel_data

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = Cylinder(10,r1=5,r2=2,res=128)
    #M.plot()
    #M.export("test_model_sphere10.png")
    #M.export("test_model_sphere10.stl")
    #M.export("test_model_sphere10.nii")