import numpy as np

class VoxelGrid:
    def __init__(self, xlim,ylim,zlim,res):
        assert(xlim[0] < xlim[1]);assert(ylim[0] < ylim[1]);assert(zlim[0] < zlim[1])
        self.xlim = np.array(xlim)
        self.ylim = np.array(ylim)
        self.zlim = np.array(zlim)
        assert(self.xlim.shape == self.ylim.shape == self.zlim.shape == (2,))
        self.res_vector  = np.array(res)*np.ones(3)
        
    def compute_size_vector(self):
        x0,x1 = self.xlim; y0,y1 = self.ylim; z0,z1 = self.zlim
        return np.array((x1-x0,y1-y0,z1-z0))
        
    def compute_center_vector(self):
        x0,x1 = self.xlim; y0,y1 = self.ylim; z0,z1 = self.zlim
        return np.array(((x0+x1)/2,(y0+y1)/2,(z0+z1)/2))
        
    def construct_mesh(self):
        r0,r1,r2 = self.res_vector
        x0,x1 = self.xlim; y0,y1 = self.ylim; z0,z1 = self.zlim
        return np.mgrid[x0:x1:r0*1j, y0:y1:r1*1j, z0:z1:r2*1j]
        
    def __or__(self, other): #union
        #create a bounding cube of maximal resolution
        r0   = max(self.res_vector[0],other.res_vector[0])
        r1   = max(self.res_vector[0],other.res_vector[0])
        r2   = max(self.res_vector[0],other.res_vector[0])
        #minimin, maximax
        xlim = (min(self.xlim[0],other.xlim[0]),
                max(self.xlim[1],other.xlim[1]))
        ylim = (min(self.ylim[0],other.ylim[0]),
                max(self.ylim[1],other.ylim[1]))
        zlim = (min(self.zlim[0],other.zlim[0]),
                max(self.zlim[1],other.zlim[1]))
        return VoxelGrid(xlim,ylim,zlim,(r0,r1,r2))
        
    def __and__(self, other): #intersection
        #create a bounding cube of maximal resolution
        r0   = max(self.res_vector[0],other.res_vector[0])
        r1   = max(self.res_vector[0],other.res_vector[0])
        r2   = max(self.res_vector[0],other.res_vector[0])
        #maximin, minimax
        xlim = (max(self.xlim[0],other.xlim[0]),
                min(self.xlim[1],other.xlim[1]))
        ylim = (max(self.ylim[0],other.ylim[0]),
                min(self.ylim[1],other.ylim[1]))
        zlim = (max(self.zlim[0],other.zlim[0]),
                min(self.zlim[1],other.zlim[1]))
        return VoxelGrid(xlim,ylim,zlim,(r0,r1,r2))