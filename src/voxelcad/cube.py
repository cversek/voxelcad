import numpy as np

import voxelcad.environment as ENV

from voxelcad.voxel_model import VoxelModel
from voxelcad.voxel_grid  import VoxelGrid

from voxelcad.debug import currentframe, DEBUG_TAG, DEBUG_EMBED

class Cube(VoxelModel):
    def __init__(self, size, voxel_size=None, center=False, **kwargs):
        super().__init__(**kwargs)
        self.size = np.array(size)*np.ones(3)
        self.center = center
        #set up grid dimensions
        if self.center:
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
        
    def render_volume(self):
        super().render_volume() # will construct_grid if it is None
        # fill all of the cubic volume between the margins
        X,Y,Z,V,m = self.grid.construct_mesh()
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        sx,sy,sz = self.size
        cx,cy,cz = self.grid.compute_center_vector()
        V[m:-m,m:-m,m:-m] = (np.abs(X-cx) <= sx/2) &\
                            (np.abs(Y-cy) <= sy/2) &\
                            (np.abs(Z-cz) <= sz/2)
        self.voxel_data = V
        return self.voxel_data

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = Cube(10,res=32)
    M.plot(show=True)
    M.export("test_model_cube10.png")
    M.export("test_model_cube10.stl")
    M.export("test_model_cube10.nii")
    M = Cube([10,20,30],res=32)
    M.plot(show=True)
    M.export("test_model_cube10x20x30.png")
    M.export("test_model_cube10x20x30.stl")
    M.export("test_model_cube10x20x30.nii")