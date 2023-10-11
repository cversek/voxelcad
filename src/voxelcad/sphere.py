import numpy as np

import voxelcad.environment as ENV

from voxelcad.voxel_model import VoxelModel
from voxelcad.voxel_grid  import VoxelGrid

class Sphere(VoxelModel):
    def __init__(self, r, voxel_size=None, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        #construct_grid
        sv = np.array(2*r)*np.ones(3)
        sx,sy,sz = sv/2
        self.grid = VoxelGrid(xlim=(-sx,sx),
                              ylim=(-sy,sy),
                              zlim=(-sz,sz),
                              voxel_size=voxel_size)
        
    def render_volume(self):
        super().render_volume()
        r = self.r
        cx,cy,cz  = self.grid.compute_center_vector()
        X,Y,Z = self.grid.construct_mesh()
        # fill the spherical volume between the margins
        V = ((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2 <= r**2)
        self.voxel_data = V
        return self.voxel_data

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = Sphere(10,res=32)
    M.plot(show=True)
    M.export("test_model_sphere10.png")
    M.export("test_model_sphere10.stl")
    M.export("test_model_sphere10.nii")
