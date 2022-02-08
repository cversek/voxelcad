import numpy as np
import pyvista as pv

from voxelcad.debug import create_logger, currentframe, DEBUG_TAG, DEBUG_EMBED, MEMORY_USAGE

import voxelcad.environment as ENV

#subclass the pyvista UniformGrid class to provide convenient extensions
class UniformGrid(pv.UniformGrid):
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
        
    def construct_mesh(self, make_empty_voxels=False, voxel_dtype = 'bool'):   
        rx,ry,rz    = self.res_vector
        vsx,vsy,vsz = self.voxel_size_vector 
        x0,x1 = self.xlim; y0,y1 = self.ylim; z0,z1 = self.zlim
        #transform to center cell coords
        xcc0=x0+vsx/2.0; xcc1=x1-vsx/2.0
        ycc0=y0+vsy/2.0; ycc1=y1-vsy/2.0
        zcc0=z0+vsz/2.0; zcc1=z1-vsz/2.0
        X,Y,Z =  np.mgrid[xcc0:xcc1:rx*1j, ycc0:ycc1:ry*1j, zcc0:zcc1:rz*1j]
        if make_empty_voxels:
            #build an empty voxel array with a margin
            V = np.zeros(self.res_vector).astype(voxel_dtype)
            #V = np.packbits(V)
            return (X,Y,Z,V)
        else:
            return (X,Y,Z)

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
