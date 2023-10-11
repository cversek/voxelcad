import numpy as np
from numpy import sin, cos, tan, pi

from voxelcad.debug import currentframe, DEBUG_TAG, DEBUG_EMBED

from voxelcad.cube import Cube

class GyroidCube(Cube):
    def __init__(self, size, 
                 lattice_param = [1.0,1.0,1.0], 
                 structure_param=0.0,
                 phi = [0.0,0.0,0.0], 
                 thresh1=1.0, 
                 thresh2=None, 
                 **kwargs):
        super().__init__(size, **kwargs)
        self.lattice_param = np.array(lattice_param)*np.ones(3)
        self.structure_param = structure_param
        self.phi = phi
        self.thresh1 = thresh1
        self.thresh2 = thresh2
    
    def render_volume(self):
        # REF https://forum.freecadweb.org/viewtopic.php?t=19819#p233282
        super().render_volume()
        X,Y,Z = self.grid.construct_mesh()
        # the gyroid is defined as continuous function on the mesh
        a = pi*self.lattice_param
        X *= a[0]
        Y *= a[1]
        Z *= a[2]
        phi = self.phi
        F = cos(X + phi[0])*sin(Y + phi[1]) +\
            cos(Y + phi[1])*sin(Z + phi[2]) +\
            cos(Z + phi[2])*sin(X + phi[0]) - self.structure_param
        # threshold to make solid and fill space between margins
        if self.thresh1 is not None and self.thresh2 is not None:
            V =  ((F > self.thresh1) & (F < self.thresh2))
        elif self.thresh1 is not None:
            #V[m:-m,m:-m,m:-m] = (F > 0) & (F < self.thresh1)
            V = (F > 0) & (F < self.thresh1)
        else:
            raise ValueError("Either or both thresh1, thresh2 should not be None")
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        self.voxel_data = V
        return self.voxel_data

class WigglyGyroidCube(GyroidCube):
    def __init__(self, size,
                 w_freq   = 5,
                 w_expon  = 3,
                 w_amp    = 0.5, 
                 **kwargs):
        super().__init__(size,**kwargs)
        self.w_freq   = w_freq
        self.w_expon  = w_expon
        self.w_amp    = w_amp

    def render_volume(self):
        # REF https://forum.freecadweb.org/viewtopic.php?t=19819#p233282
        super().render_volume()
        X,Y,Z = self.grid.construct_mesh()
        # the gyroid is defined as continuous function on the mesh
        a = pi*self.lattice_param
        b = self.w_freq
        p = self.w_expon
        Aw = self.w_amp
        #precompute some useful quantities
        X *= a
        Y *= a
        Z *= a
        cosX = cos(X);cosY = cos(Y);cosZ = cos(Z)
        sinX = sin(X);sinY = sin(Y);sinZ = sin(Z)
        bX,bY,bZ = (b*X,b*Y,b*Z)
        gradX = cosZ*cosX - sinX*sinY
        gradY = cosX*cosY - sinY*sinZ
        gradZ = cosY*cosZ - sinZ*sinX
        #wiggle along the gradient direction (normal to surface)
        wx = Aw*(cos(bY)*sin(bZ))**p
        wy = Aw*(sin(bX)*cos(bZ))**p
        wz = Aw*(cos(bX)*sin(bY))**p
        Ffunc  = lambda x,y,z: cos(x)*sin(y) + cos(y)*sin(z) + cos(z)*sin(x) - self.structure_param
        Fw1 = Ffunc(X - wx*gradX,Y - wy*gradY,Z - wz*gradZ)
        Fw2 = Ffunc(X + wx*gradX,Y + wy*gradY,Z + wz*gradZ)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        # threshold to make solid and fill space between margins
        if self.thresh1 is not None and self.thresh2 is not None:
            V =  ((Fw1 > self.thresh1) & (Fw1 < self.thresh2)) | ((Fw2 > self.thresh1) & (Fw2 < self.thresh2))
        #elif self.thresh1 is not None:
        #    V = (F(X,Y,Z) > 0) | (F(Xw,Yw,Zw) > self.thresh1)
        else:
            raise ValueError("Either or both thresh1, thresh2 should not be None")
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        self.voxel_data = V
        return self.voxel_data


class HyperWigglyGyroidCube(GyroidCube):
    def __init__(self, size, 
                 w_freq   = 5,
                 w_expon  = 3,
                 w_amp    = 0.5, 
                 **kwargs):
        super().__init__(size,**kwargs)
        self.w_freq   = w_freq
        self.w_expon  = w_expon
        self.w_amp    = w_amp

    def render_volume(self):
        # REF https://forum.freecadweb.org/viewtopic.php?t=19819#p233282
        super().render_volume()
        X,Y,Z = self.grid.construct_mesh()
        # the gyroid is defined as continuous function on the mesh
        a = pi*self.lattice_param
        b = self.w_freq
        p = self.w_expon
        Aw = self.w_amp
        #precompute some useful quantities
        X *= a
        Y *= a
        Z *= a
        cosX = cos(X);cosY = cos(Y);cosZ = cos(Z)
        sinX = sin(X);sinY = sin(Y);sinZ = sin(Z)
        bX,bY,bZ = (b*X,b*Y,b*Z)
        gradX = cosZ*cosX - sinX*sinY
        gradY = cosX*cosY - sinY*sinZ
        gradZ = cosY*cosZ - sinZ*sinX
        #wiggle along the gradient direction (normal to surface)
        wx = Aw*(cos(bY)*sin(bZ))**p + 0.5*Aw*(cos(3*bY)*sin(3*bZ))**(p+1)
        wy = Aw*(sin(bX)*cos(bZ))**p + 0.5*Aw*(sin(3*bX)*cos(3*bZ))**(p+1)
        wz = Aw*(cos(bX)*sin(bY))**p + 0.5*Aw*(cos(3*bX)*sin(3*bY))**(p+1)
        Ffunc  = lambda x,y,z: cos(x)*sin(y) + cos(y)*sin(z) + cos(z)*sin(x) - self.structure_param
        Fw1 = Ffunc(X - wx*gradX,Y - wy*gradY,Z - wz*gradZ)
        Fw2 = Ffunc(X + wx*gradX,Y + wy*gradY,Z + wz*gradZ)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        # threshold to make solid and fill space between margins
        if self.thresh1 is not None and self.thresh2 is not None:
            V =  ((Fw1 > self.thresh1) & (Fw1 < self.thresh2)) | ((Fw2 > self.thresh1) & (Fw2 < self.thresh2))
        #elif self.thresh1 is not None:
        #    V = (F(X,Y,Z) > 0) | (F(Xw,Yw,Zw) > self.thresh1)
        else:
            raise ValueError("Either or both thresh1, thresh2 should not be None")
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        self.voxel_data = V
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
