import numpy as np

import voxelcad.environment as ENV

from voxelcad.voxel_model import VoxelModel
from voxelcad.voxel_grid  import VoxelGrid

from voxelcad.debug import currentframe, DEBUG_TAG, DEBUG_EMBED

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

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = Cube(10,res=32)
    M.plot(show=True)
    M.export("test_model_cube10.stl")
