import numpy as np
from numpy import sin, cos, tan, pi

from voxelcad.debug import currentframe, DEBUG_TAG, DEBUG_EMBED, TIMING_START, TIMING_END
from voxelcad._kernels import (
    CYTHON_AVAILABLE,
    evaluate_and_pack_gyroid,
    evaluate_and_pack_wiggly_gyroid,
    evaluate_and_pack_hyperwiggly_gyroid,
)

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

    def evaluate_slice(self, X_2d, Y_2d, z_val):
        # REF https://forum.freecadweb.org/viewtopic.php?t=19819#p233282
        a = pi*self.lattice_param
        Xa = X_2d * a[0]
        Ya = Y_2d * a[1]
        Za = z_val * a[2]
        phi = self.phi
        F = cos(Xa + phi[0])*sin(Ya + phi[1]) +\
            cos(Ya + phi[1])*sin(Za + phi[2]) +\
            cos(Za + phi[2])*sin(Xa + phi[0]) - self.structure_param
        if self.thresh1 is not None and self.thresh2 is not None:
            V = ((F > self.thresh1) & (F < self.thresh2))
        elif self.thresh1 is not None:
            V = (F > 0) & (F < self.thresh1)
        else:
            raise ValueError("Either or both thresh1, thresh2 should not be None")
        # intersect with cube bounds
        V &= super().evaluate_slice(X_2d, Y_2d, z_val)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return V

    def render_volume(self):
        if not CYTHON_AVAILABLE:
            return super().render_volume()
        TIMING_START("render_volume")
        xcc, ycc, zcc = self.grid.compute_cell_center_ranges()
        cx, cy, cz = self.grid.compute_center_vector()
        sx, sy, sz = self.size
        a = pi * self.lattice_param
        self.voxel_data = evaluate_and_pack_gyroid(
            xcc, ycc, zcc, cx, cy, cz,
            sx / 2.0, sy / 2.0, sz / 2.0,
            a[0], a[1], a[2],
            self.phi[0], self.phi[1], self.phi[2],
            self.structure_param,
            self.thresh1 if self.thresh1 is not None else 0.0,
            self.thresh2 if self.thresh2 is not None else 0.0,
            1 if (self.thresh1 is not None and self.thresh2 is not None) else 0,
        )
        self._voxel_shape = (len(xcc), len(ycc), len(zcc))
        TIMING_END("render_volume")
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

    def evaluate_slice(self, X_2d, Y_2d, z_val):
        a = pi*self.lattice_param
        b = self.w_freq
        p = self.w_expon
        Aw = self.w_amp
        Xa = X_2d * a[0]
        Ya = Y_2d * a[1]
        Za = z_val * a[2]
        cosX = cos(Xa);cosY = cos(Ya);cosZ = cos(Za)
        sinX = sin(Xa);sinY = sin(Ya);sinZ = sin(Za)
        bX = b*Xa; bY = b*Ya; bZ = b*Za
        gradX = cosZ*cosX - sinX*sinY
        gradY = cosX*cosY - sinY*sinZ
        gradZ = cosY*cosZ - sinZ*sinX
        wx = Aw*(cos(bY)*sin(bZ))**p
        wy = Aw*(sin(bX)*cos(bZ))**p
        wz = Aw*(cos(bX)*sin(bY))**p
        Ffunc = lambda x,y,z: cos(x)*sin(y) + cos(y)*sin(z) + cos(z)*sin(x) - self.structure_param
        Fw1 = Ffunc(Xa - wx*gradX, Ya - wy*gradY, Za - wz*gradZ)
        Fw2 = Ffunc(Xa + wx*gradX, Ya + wy*gradY, Za + wz*gradZ)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        if self.thresh1 is not None and self.thresh2 is not None:
            V = ((Fw1 > self.thresh1) & (Fw1 < self.thresh2)) | ((Fw2 > self.thresh1) & (Fw2 < self.thresh2))
        else:
            raise ValueError("Either or both thresh1, thresh2 should not be None")
        # intersect with cube bounds
        V &= Cube.evaluate_slice(self, X_2d, Y_2d, z_val)
        return V

    def render_volume(self):
        if not CYTHON_AVAILABLE:
            return super().render_volume()
        TIMING_START("render_volume")
        xcc, ycc, zcc = self.grid.compute_cell_center_ranges()
        cx, cy, cz = self.grid.compute_center_vector()
        sx, sy, sz = self.size
        a = pi * self.lattice_param
        self.voxel_data = evaluate_and_pack_wiggly_gyroid(
            xcc, ycc, zcc, cx, cy, cz,
            sx / 2.0, sy / 2.0, sz / 2.0,
            a[0], a[1], a[2],
            self.phi[0], self.phi[1], self.phi[2],
            self.structure_param, self.thresh1, self.thresh2,
            self.w_freq, self.w_amp, self.w_expon,
        )
        self._voxel_shape = (len(xcc), len(ycc), len(zcc))
        TIMING_END("render_volume")
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

    def evaluate_slice(self, X_2d, Y_2d, z_val):
        a = pi*self.lattice_param
        b = self.w_freq
        p = self.w_expon
        Aw = self.w_amp
        Xa = X_2d * a[0]
        Ya = Y_2d * a[1]
        Za = z_val * a[2]
        cosX = cos(Xa);cosY = cos(Ya);cosZ = cos(Za)
        sinX = sin(Xa);sinY = sin(Ya);sinZ = sin(Za)
        bX = b*Xa; bY = b*Ya; bZ = b*Za
        gradX = cosZ*cosX - sinX*sinY
        gradY = cosX*cosY - sinY*sinZ
        gradZ = cosY*cosZ - sinZ*sinX
        wx = Aw*(cos(bY)*sin(bZ))**p + 0.5*Aw*(cos(3*bY)*sin(3*bZ))**(p+1)
        wy = Aw*(sin(bX)*cos(bZ))**p + 0.5*Aw*(sin(3*bX)*cos(3*bZ))**(p+1)
        wz = Aw*(cos(bX)*sin(bY))**p + 0.5*Aw*(cos(3*bX)*sin(3*bY))**(p+1)
        Ffunc = lambda x,y,z: cos(x)*sin(y) + cos(y)*sin(z) + cos(z)*sin(x) - self.structure_param
        Fw1 = Ffunc(Xa - wx*gradX, Ya - wy*gradY, Za - wz*gradZ)
        Fw2 = Ffunc(Xa + wx*gradX, Ya + wy*gradY, Za + wz*gradZ)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        if self.thresh1 is not None and self.thresh2 is not None:
            V = ((Fw1 > self.thresh1) & (Fw1 < self.thresh2)) | ((Fw2 > self.thresh1) & (Fw2 < self.thresh2))
        else:
            raise ValueError("Either or both thresh1, thresh2 should not be None")
        # intersect with cube bounds
        V &= Cube.evaluate_slice(self, X_2d, Y_2d, z_val)
        return V

    def render_volume(self):
        if not CYTHON_AVAILABLE:
            return super().render_volume()
        TIMING_START("render_volume")
        xcc, ycc, zcc = self.grid.compute_cell_center_ranges()
        cx, cy, cz = self.grid.compute_center_vector()
        sx, sy, sz = self.size
        a = pi * self.lattice_param
        self.voxel_data = evaluate_and_pack_hyperwiggly_gyroid(
            xcc, ycc, zcc, cx, cy, cz,
            sx / 2.0, sy / 2.0, sz / 2.0,
            a[0], a[1], a[2],
            self.phi[0], self.phi[1], self.phi[2],
            self.structure_param, self.thresh1, self.thresh2,
            self.w_freq, self.w_amp, self.w_expon,
        )
        self._voxel_shape = (len(xcc), len(ycc), len(zcc))
        TIMING_END("render_volume")
        return self.voxel_data

################################################################################
# TEST CODE
################################################################################
if __name__ == "__main__":
    M = GyroidCube(10,thresh1=-0.1,thresh2=0.1,res=32)
    M.plot(show=True)
    M.export("test_model_gyroidcube10_thresh-0p1_to_0p1.stl")
