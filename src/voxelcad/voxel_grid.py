import numpy as np
import pyvista as pv

from voxelcad.debug import create_logger, currentframe, DEBUG_TAG, DEBUG_EMBED, MEMORY_USAGE

import voxelcad.environment as ENV

# PyVista 0.43+ renamed UniformGrid to ImageData
_PVGridBase = getattr(pv, 'ImageData', None) or pv.UniformGrid

class UniformGrid(_PVGridBase):
    #overload plots with some helpful defaults
    def plot(volume=True, opacity="sigmoid", cmap="coolwarm",*args,**kwargs):
        kwargs['volume']  = volume
        kwargs['opacity'] = opacity
        kwargs['cmap']    = cmap
        super().plot(*args,**kwargs)


class VoxelGrid:
    def __init__(self,xlim,ylim,zlim,voxel_size):
        assert(xlim[0] < xlim[1]);assert(ylim[0] < ylim[1]);assert(zlim[0] < zlim[1])
        self.xlim = np.array(xlim)
        self.ylim = np.array(ylim)
        self.zlim = np.array(zlim)
        assert(self.xlim.shape == self.ylim.shape == self.zlim.shape == (2,))
        #format the voxel size
        if voxel_size is None:
            voxel_size = ENV.voxel_size
        self.voxel_size_vector = vsv  = (np.array(voxel_size)*np.ones(3)).astype('float32')
        #derive the resolution vector to best approximate the voxel size
        sv = self.compute_size_vector()
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals(),exit=False)
        self.res_vector  = np.ceil(sv/vsv).astype('uint')

    def same_grid(self, other):
        """Check if another VoxelGrid has identical geometry.

        Returns True when origin, voxel_size, and resolution all match,
        meaning packed voxel_data arrays are directly compatible for
        byte-level boolean operations.
        """
        return (np.array_equal(self.res_vector, other.res_vector) and
                np.allclose(self.xlim, other.xlim) and
                np.allclose(self.ylim, other.ylim) and
                np.allclose(self.zlim, other.zlim) and
                np.allclose(self.voxel_size_vector, other.voxel_size_vector))

    def compatible_grid(self, other):
        """Check if another VoxelGrid has compatible voxel sizing.

        Returns True when voxel_size_vectors match (within tolerance),
        meaning both grids can be rendered onto a common union grid
        and combined with byte-level boolean operations.

        Note: same_grid() implies compatible_grid(), but compatible_grid()
        allows different bounding boxes.
        """
        return np.allclose(self.voxel_size_vector, other.voxel_size_vector)

    def compute_size_vector(self):
        x0,x1 = self.xlim; y0,y1 = self.ylim; z0,z1 = self.zlim
        return np.array((x1-x0,y1-y0,z1-z0))

    def compute_center_vector(self):
        x0,x1 = self.xlim; y0,y1 = self.ylim; z0,z1 = self.zlim
        return np.array(((x0+x1)/2,(y0+y1)/2,(z0+z1)/2))

    def compute_box_corner_vectors(self):
        x0,x1 = self.xlim
        y0,y1 = self.ylim
        z0,z1 = self.zlim
        C = np.array((
            (x0,y0,z0),
            (x1,y0,z0),
            (x0,y1,z0),
            (x1,y1,z0),
            (x0,y0,z1),
            (x1,y0,z1),
            (x0,y1,z1),
            (x1,y1,z1)
        ))
        return C

    def compute_cell_center_ranges(self):
        """Compute cell-center coordinate ranges for each axis.

        Returns (xcc, ycc, zcc) where each is a 1D array of cell centers.
        """
        rx,ry,rz    = self.res_vector
        vsx,vsy,vsz = self.voxel_size_vector
        x0,x1 = self.xlim; y0,y1 = self.ylim; z0,z1 = self.zlim
        xcc = np.linspace(x0+vsx/2.0, x1-vsx/2.0, int(rx))
        ycc = np.linspace(y0+vsy/2.0, y1-vsy/2.0, int(ry))
        zcc = np.linspace(z0+vsz/2.0, z1-vsz/2.0, int(rz))
        return xcc, ycc, zcc

    def iter_slices(self):
        """Yield (X_2d, Y_2d, z_val, k) for each Z-level.

        X_2d and Y_2d are 2D coordinate arrays of shape (rx, ry).
        z_val is the scalar z coordinate for this slice.
        k is the Z-index (0-based).

        Memory: only 2D arrays (~16 MB at 1024^2) instead of 3D (~24 GB at 1024^3).
        """
        xcc, ycc, zcc = self.compute_cell_center_ranges()
        # pre-allocate 2D coordinate grids (reused each iteration)
        X_2d, Y_2d = np.meshgrid(xcc, ycc, indexing='ij')
        for k, z_val in enumerate(zcc):
            yield X_2d, Y_2d, z_val, k

    def __or__(self, other): #union
        #minimin, maximax
        xlim = (min(self.xlim[0],other.xlim[0]),
                max(self.xlim[1],other.xlim[1]))
        ylim = (min(self.ylim[0],other.ylim[0]),
                max(self.ylim[1],other.ylim[1]))
        zlim = (min(self.zlim[0],other.zlim[0]),
                max(self.zlim[1],other.zlim[1]))
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals(),exit=False)
        return VoxelGrid._construct_new_bounding_grid(self,other,xlim,ylim,zlim)


    def __and__(self, other): #intersection
        #maximin, minimax
        xlim = (max(self.xlim[0],other.xlim[0]),
                min(self.xlim[1],other.xlim[1]))
        ylim = (max(self.ylim[0],other.ylim[0]),
                min(self.ylim[1],other.ylim[1]))
        zlim = (max(self.zlim[0],other.zlim[0]),
                min(self.zlim[1],other.zlim[1]))
        return VoxelGrid._construct_new_bounding_grid(self,other,xlim,ylim,zlim)

    @classmethod
    def _construct_new_bounding_grid(cls,grid1,grid2,xlim,ylim,zlim):
        #compute new size vector
        sv = np.array((xlim[1]-xlim[0],ylim[1]-ylim[0],zlim[1]-zlim[0]))
        #preserve voxel size of the finest elements
        vsv1 = grid1.voxel_size_vector
        vsv2 = grid2.voxel_size_vector
        new_vsv = np.vstack((vsv1,vsv2)).min(axis=0)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals(),exit=False)
        return cls(xlim,ylim,zlim,new_vsv)

    def __repr__(self):
        s = self
        return f"VoxelGrid(xlim={s.xlim},ylim={s.ylim},zlim={s.zlim},voxel_size={s.voxel_size_vector})"
