import numpy as np
from numpy import sin, cos, pi

from .cube import Cube

class GyroidCube(Cube):
    def __init__(self, size, 
                 lattice_param = 1.0, 
                 structure_param=0.0, 
                 thresh1=0.0, 
                 thresh2=None, 
                 res=32, 
                 **kwargs):
        super().__init__(size,res, **kwargs)
        self.lattice_param = lattice_param
        self.structure_param = structure_param
        self.thresh1 = thresh1
        self.thresh2 = thresh2
    
    def render_volume(self):
        # REF https://forum.freecadweb.org/viewtopic.php?t=19819#p233282
        super().render_volume()
        X,Y,Z = self.grid.construct_mesh()
        a = pi*self.lattice_param
        I = cos(a*X)*sin(a*Y) + cos(a*Y)*sin(a*Z) + cos(a*Z)*sin(a*X) - self.structure_param
        #threshold to make solid
        if self.thresh1 is not None and self.thresh2 is not None:
            self.voxel_data = (self.thresh1 < I) & (I > self.thresh2) 
        elif self.thresh1 is not None:
            self.voxel_data = (I > self.thresh1)
        else:
            raise ValueError("Either or both thresh1, thresh2 should not be None")
        return self.voxel_data

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = GyroidCube(10,thresh1=-0.1,thresh2=0.1,res=32)
    M.plot(show=True)
    M.export("test_model_gyroidcube10_thresh-0p1_to_0p1.png")
    M.export("test_model_gyroidcube10_thresh-0p1_to_0p1.stl")
    M.export("test_model_gyroidcube10_thresh-0p1_to_0p1.nii")