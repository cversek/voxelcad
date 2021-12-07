import numpy as np

from voxelcad.debug import currentframe, DEBUG_TAG, DEBUG_EMBED


class VoxelGrid:
    def __init__(self, xlim,ylim,zlim,res,margin=2):
        assert(xlim[0] < xlim[1]);assert(ylim[0] < ylim[1]);assert(zlim[0] < zlim[1])
        self.xlim = np.array(xlim)
        self.ylim = np.array(ylim)
        self.zlim = np.array(zlim)
        assert(self.xlim.shape == self.ylim.shape == self.zlim.shape == (2,))
        self.res_vector  = (np.array(res)*np.ones(3)).astype('uint')
        #NOTE must have a 1 voxel margin at edges for surface algos to work properly
        margin = int(margin)
        assert(margin >= 1)
        self.margin = margin
        
    def compute_size_vector(self):
        x0,x1 = self.xlim; y0,y1 = self.ylim; z0,z1 = self.zlim
        return np.array((x1-x0,y1-y0,z1-z0))

    def compute_voxel_size_vector(self):
        sv = self.compute_size_vector()
        vsv = sv/self.res_vector
        return vsv
        
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
        
    def construct_mesh(self, make_empty_voxels=True, voxel_dtype = 'bool'):   
        rx,ry,rz = self.res_vector
        x0,x1 = self.xlim; y0,y1 = self.ylim; z0,z1 = self.zlim
        X,Y,Z =  np.mgrid[x0:x1:rx*1j, y0:y1:ry*1j, z0:z1:rz*1j]
        if make_empty_voxels:
            #build an empty voxel array with a margin
            m = self.margin
            V = np.zeros(self.res_vector+2*m).astype(voxel_dtype)
            return (X,Y,Z,V,m)
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
        #compute new resolution vectors
        rv1 = sv/grid1.compute_voxel_size_vector()
        rv2 = sv/grid2.compute_voxel_size_vector()
        #maximize resolution preserve voxel size of the finest elements
        rx   = max(rv1[0],rv2[0])
        ry   = max(rv1[1],rv2[1])
        rz   = max(rv1[2],rv2[2])
        return cls(xlim,ylim,zlim,(rx,ry,rz))

    def __repr__(self):
        s = self
        return f"VoxelGrid(xlim={s.xlim},ylim={s.ylim},zlim={s.zlim},res={s.res_vector})"
