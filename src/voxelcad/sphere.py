import numpy as np
from numpy.lib.twodim_base import _trilu_dispatcher

from .voxel_model import VoxelModel
from .voxel_grid  import VoxelGrid

class Sphere(VoxelModel):
    def __init__(self, r, res=32, **kwargs):
        super().__init__(**kwargs)
        self.size_vector = np.array(2*r)*np.ones(3)
        self.res_vector  = np.array(res)*np.ones(3)
        self.r = r
        #construct_grid
        sx,sy,sz = self.size_vector/2
        self.grid = VoxelGrid(xlim=(-sx,sx),
                              ylim=(-sy,sy),
                              zlim=(-sz,sz),
                              res=self.res_vector)
        
    def render_volume(self):
        super().render_volume()
        r = self.r
        cx,cy,cz = self.grid.compute_center_vector()
        X,Y,Z = self.grid.construct_mesh()
        #the sphere is always centered in the mesh
        self.voxel_data = ((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2 <= r**2)
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