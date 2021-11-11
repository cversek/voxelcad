import numpy as np

from .environment import Environment as ENV
from .voxel_model import VoxelModel
from .voxel_grid  import VoxelGrid

class Cube(VoxelModel):
    def __init__(self, size, res=None, centered=False, **kwargs):
        super().__init__(**kwargs)
        self.size_vector = sv = np.array(size)*np.ones(3)
        if res is None:
            res = ENV.res
        self.res_vector  = np.array(res)*np.ones(3)
        self.centered = centered
        #construct_grid
        rx,ry,rz = self.res_vector
        sv = self.size_vector
        if self.centered:
            sx,sy,sz = sv/2
            self.grid = VoxelGrid(xlim=(-sx,sx),
                                  ylim=(-sy,sy),
                                  zlim=(-sz,sz),
                                  res=self.res_vector)
        else:
            sx,sy,sz = sv
            self.grid = VoxelGrid(xlim=(0,sx),
                                  ylim=(0,sy),
                                  zlim=(0,sz),
                                  res=self.res_vector)
        
    def render_volume(self):
        super().render_volume() #will construct_grid if it is None
        #fill all of the cubic volume
        X,Y,Z = self.grid.construct_mesh()
        self.voxel_data = (X == X)
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