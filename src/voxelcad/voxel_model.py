import os
from typing import OrderedDict
import numpy as np
import pyvista as pv

import voxelcad.environment as ENV

from voxelcad.debug import create_logger, currentframe, DEBUG_TAG, DEBUG_EMBED, MEMORY_USAGE
LOGGER = create_logger(__name__)

from voxelcad.voxel_grid import VoxelGrid, UniformGrid


class VoxelModel:
    def __init__(self, 
                 grid = None, 
                 voxel_data = None,
                 surface_data = None,
                 mesh_data = None,
                 pv_vol = None,
                 pv_surf = None,
                 ):
        self.grid = grid
        self.voxel_data   = voxel_data
        self.surface_data = surface_data
        self.mesh_data    = mesh_data
        self.pv_vol       = pv_vol
        self.pv_surf      = pv_surf
        
    def render_uniform_grid(self, volume_scale=255):
        #REF https://docs.pyvista.org/examples/00-load/create-uniform-grid.html
        #render the voxels first
        LOGGER.info(40*"*")
        LOGGER.info(f"{self.__class__} -> super().render_uniform_grid")
        mem0 = MEMORY_USAGE()
        LOGGER.info(f"TOTAL MEMORY USED: {mem0/2**30:0.2f} GB")
        LOGGER.info(40*"-")
        if self.voxel_data is None:
            self.render_volume()
        #get a numpy array of the grid points
        X,Y,Z = self.grid.construct_mesh(make_empty_voxels=False)
        m = self.grid.margin
        V = self.voxel_data[m:-m,m:-m,m:-m]
        rv  = self.grid.res_vector
        vsv = self.grid.voxel_size_vector
        ugrid = UniformGrid()
        # Set the grid dimensions: shape + 1 because we want to inject our values on
        #   the CELL data
        ugrid.dimensions = rv + 1
        # Edit the spatial reference
        #grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
        ugrid.spacing = vsv  # These are the cell sizes along each axis
        ugrid.cell_data['vol'] = volume_scale*V.flatten(order="F") #NOTE column-major (Fortran) order must be specified!
        LOGGER.info(f"END render_uniform_grid")
        mem = MEMORY_USAGE(offset=mem0)
        LOGGER.info(f"DELTA MEMORY USED: {mem/2**30:0.2f} GB")
        LOGGER.info(40*"*")
        return ugrid
            
    def render_volume(self):
        if self.grid is None:
            self.construct_grid()

    def render_volume_mesh(self, cache=True):
        #REF https://stackoverflow.com/questions/6030098/how-to-display-a-3d-plot-of-a-3d-array-isosurface-in-matplotlib-mplot3d-or-simil/35472146
        #get from cache if already computed
        LOGGER.info(40*"*")
        LOGGER.info(f"{self.__class__} -> super().render_volume_mesh")
        mem0 = MEMORY_USAGE()
        LOGGER.info(f"TOTAL MEMORY USED: {mem0/2**30:0.2f} GB")
        LOGGER.info(40*"-")
        if cache and self.pv_vol is not None:
            return self.pv_vol
        pv_grid = self.render_uniform_grid()
        pv_vol = pv_grid.threshold(0.5) #convert to unstructured grid of just the solid areas
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        if cache:
            self.pv_vol = pv_vol
        LOGGER.info(f"END render_volume_mesh")
        mem = MEMORY_USAGE(offset=mem0)
        LOGGER.info(f"DELTA MEMORY USED: {mem/2**30:0.2f} GB")
        LOGGER.info(40*"*")
        return pv_vol

    def render_surface_mesh(self, 
                            cache=True,
                            smooth_iters = 0,
                            downscale_times = 0,
                            only_largest_component = False,
                            ):
        #REF https://stackoverflow.com/questions/6030098/how-to-display-a-3d-plot-of-a-3d-array-isosurface-in-matplotlib-mplot3d-or-simil/35472146
        #REF https://docs.pyvista.org/examples/00-load/create-uniform-grid.html
        LOGGER.info(40*"*")
        LOGGER.info(f"{self.__class__} -> super().render_surface_mesh")
        mem0 = MEMORY_USAGE()
        LOGGER.info(f"TOTAL MEMORY USED: {mem0/2**30:0.2f} GB")
        LOGGER.info(40*"-")
        #get from cache if already computed
        if cache and self.pv_surf is not None:
            return self.pv_surf
        import pyvista as pv
        #render the volume mesh first
        pv_vol  = self.render_volume_mesh()
        pv_surf = pv_vol.extract_surface()
        #do filtering steps
        if smooth_iters > 0:
            pv_surf = pv_surf.smooth(n_iter=smooth_iters,progress_bar=True)
        if downscale_times > 0:
            #we use pyvista to clean up the mesh
            pv_surf = pv_surf.triangulate() #must be triangulated
            from voxelcad.utils.pyvista_tools import downscale_trimesh
            #cut the number of triangles down by 2**3 times
            pv_surf = downscale_trimesh(pv_surf,repeat=downscale_times,decimation_factor=0.5)
        if only_largest_component:
            pv_surf = pv_surf.extract_largest()
        if cache:
            self.pv_surf = pv_surf
        LOGGER.info(f"END render_surface_mesh")
        mem = MEMORY_USAGE(offset=mem0)
        LOGGER.info(f"DELTA MEMORY USED: {mem/2**30:0.2f} GB")
        LOGGER.info(40*"*")
        return pv_surf
        
    def plot(self, *args,**kwargs):
        vol_mesh = self.render_volume_mesh()
        vol_mesh.plot(*args,**kwargs)
        
    def export(self, filename, **kwargs):
        basepath, ext = os.path.splitext(filename)
        if ext == ".stl": #STL for 3d Printing
            surf_mesh = self.render_surface_mesh(**kwargs)
            surf_mesh.save(filename)
        else:
            raise ValueError(f"The filetype of extension '{ext}' is not recognized!")
            
    def test_points(self, X, Y, Z):
        """ test if points defined by mesh X, Y, Z are in the volume
        """
        LOGGER.info(40*"*")
        LOGGER.info(f"{self.__class__} -> super().test_points")
        mem0 = MEMORY_USAGE()
        LOGGER.info(f"TOTAL MEMORY USED: {mem0/2**30:0.2f} GB")
        LOGGER.info(40*"-")
        I,J,K = self.index_transform(X,Y,Z)
        #use indices to interpolate the voxel data between the margins
        if self.voxel_data is None:  #render the voxels first
            self.render_volume()
        m = self.grid.margin
        V = self.voxel_data[m:-m,m:-m,m:-m]
        in_volume = np.where((I >=0) & (J >=0) & (K >=0),V[I,J,K],False)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        LOGGER.info(f"END test_points")
        mem = MEMORY_USAGE(offset=mem0)
        LOGGER.info(f"DELTA MEMORY USED: {mem/2**30:0.2f} GB")
        LOGGER.info(40*"*")
        return in_volume

    def index_transform(self,X,Y,Z):
        LOGGER.info(40*"*")
        LOGGER.info(f"{self.__class__} -> super().index_transform")
        mem0 = MEMORY_USAGE()
        LOGGER.info(f"TOTAL MEMORY USED: {mem0/2**30:0.2f} GB")
        LOGGER.info(40*"-")
        LOGGER.debug(f"test_points: X.min()={X.min()},  X.max()={X.max()}")
        LOGGER.debug(f"test_points: Y.min()={Y.min()},  Y.max()={Y.max()}")
        LOGGER.debug(f"test_points: Z.min()={Z.min()},  Z.max()={Z.max()}")
        x0,x1 = self.grid.xlim; y0,y1 = self.grid.ylim; z0,z1 = self.grid.zlim
        rx, ry, rz = self.grid.res_vector
        #transform into data indices, giving dummy values (-1) to points outside bounds
        i_test = np.floor(rx*(X-x0)/(x1-x0)).astype('int')
        j_test = np.floor(ry*(Y-y0)/(y1-y0)).astype('int')
        k_test = np.floor(rz*(Z-z0)/(z1-z0)).astype('int')
        LOGGER.debug(f"test_points: i_test.min()={i_test.min()},  i_test.max()={i_test.max()}")
        LOGGER.debug(f"test_points: j_test.min()={j_test.min()},  j_test.max()={j_test.max()}")
        LOGGER.debug(f"test_points: k_test.min()={k_test.min()},  k_test.max()={k_test.max()}")
        #filter based on bounds and index checking
        # I = np.where(in_bounds & (0 <= i_test) & (i_test < rx),i_test,-1)
        # J = np.where(in_bounds & (0 <= j_test) & (j_test < ry),j_test,-1)
        # K = np.where(in_bounds & (0 <= k_test) & (k_test < rz),k_test,-1)
        I = np.where((0 <= i_test) & (i_test < rx),i_test,-1)
        J = np.where((0 <= j_test) & (j_test < ry),j_test,-1)
        K = np.where((0 <= k_test) & (k_test < rz),k_test,-1)
        LOGGER.debug(f"test_points: I.min()={I.min()},  I.max()={I.max()}")
        LOGGER.debug(f"test_points: J.min()={J.min()},  J.max()={J.max()}")
        LOGGER.debug(f"test_points: K.min()={K.min()},  K.max()={K.max()}")
        LOGGER.info(f"END index_transform")
        mem = MEMORY_USAGE(offset=mem0)
        LOGGER.info(f"DELTA MEMORY USED: {mem/2**30:0.2f} GB")
        LOGGER.info(40*"*")
        return I,J,K

    def translate(self, v):
        if self.voxel_data is None:  #render the voxels first
            self.render_volume()
        new_grid = VoxelGrid(xlim=self.grid.xlim + v[0],
                             ylim=self.grid.ylim + v[1],
                             zlim=self.grid.zlim + v[2],
                             voxel_size=self.grid.voxel_size_vector,
                             margin=self.grid.margin,
                             )
        vm = VoxelModel(grid=new_grid,voxel_data=self.voxel_data)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return vm


    def rotate_z(self, degrees):
        #construct the rotation matrix and its inverse
        theta = np.radians(degrees)
        c, s = np.cos(theta), np.sin(theta)
        R    = np.array(((c,-s,0),( s,c,0),(0,0,1.0)))
        Rinv = np.array(((c, s,0),(-s,c,0),(0,0,1.0))) #subs -theta in sin
        m = self.apply_transformation(R,Rinv)
        LOGGER.debug(f"rotate_z: m.shape: {m.voxel_data.shape}")
        LOGGER.debug(f"\tm.sum: {m.voxel_data.sum()}")
        LOGGER.debug(f"\tm.grid: {m.grid!r}")
        return m

    def apply_transformation(self, M, Minv):
        #rotate the bounding box corners
        C  = self.grid.compute_box_corner_vectors() #shape (8,3) (vectors, axes)
        Ct = np.dot(C,M.T) #apply transformation matrix
        #compute the limits of the new bounding box
        x0,x1 = xlim = (Ct[:,0].min(),Ct[:,0].max())
        y0,y1 = ylim = (Ct[:,1].min(),Ct[:,1].max())
        z0,z1 = zlim = (Ct[:,2].min(),Ct[:,2].max())
        #create a new grid preserving voxel size
        rx, ry, rz = self.grid.res_vector
        sv = self.grid.compute_size_vector()
        res = (np.round(rx*(x1-x0)/sv[0]),
               np.round(ry*(y1-y0)/sv[1]),
               np.round(rz*(z1-z0)/sv[2]))
        #res = max(rx,ry,rz)
        new_grid = VoxelGrid(xlim,ylim,zlim,res=res)
        Xt,Yt,Zt,Vt,m = new_grid.construct_mesh()
        #invert the rotation to map the mesh points back to the orginal data space
        X = Minv[0,0]*Xt + Minv[0,1]*Yt + Minv[0,2]*Zt
        Y = Minv[1,0]*Xt + Minv[1,1]*Yt + Minv[1,2]*Zt
        Z = Minv[2,0]*Xt + Minv[2,1]*Yt + Minv[2,2]*Zt
        #test if the new mesh points are contained in the volume
        #and fill within the margins
        Vt[m:-m,m:-m,m:-m] = self.test_points(X,Y,Z)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return VoxelModel(grid=new_grid,voxel_data=Vt)
        
    def __or__(self, other): #union
        if self.voxel_data is None:
            self.render_volume()
        if other.voxel_data is None:
            other.render_volume()
        #compute bounding voxel grid
        bounding_grid = self.grid | other.grid
        X,Y,Z,V,m = bounding_grid.construct_mesh()
        #test if the new mesh points are contained in either of the respective volumes 
        #and fill within the margins
        V1 = self.test_points(X,Y,Z)
        V2 = other.test_points(X,Y,Z)
        V[m:-m,m:-m,m:-m] = V1 | V2
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return VoxelModel(grid=bounding_grid,voxel_data=V)
         
    def __and__(self, other): #intersection
        if self.voxel_data is None:
            self.render_volume()
        if other.voxel_data is None:
            other.render_volume()
        #compute bounding voxel grid
        bounding_grid = self.grid & other.grid
        X,Y,Z,V,m = bounding_grid.construct_mesh()
        #test if the new mesh points are contained in both of the respective volumes 
        V[m:-m,m:-m,m:-m] = self.test_points(X,Y,Z) & other.test_points(X,Y,Z)
        return VoxelModel(grid=bounding_grid,voxel_data=V)

    def __sub__(self, other): #difference
        if self.voxel_data is None:
            self.render_volume()
        if other.voxel_data is None:
            other.render_volume()
        #compute bounding voxel grid
        X,Y,Z,V,m = self.grid.construct_mesh()
        #test if the new mesh points are contained in the first but not the second volume
        V[m:-m,m:-m,m:-m] = self.test_points(X,Y,Z) & ~other.test_points(X,Y,Z)
        return VoxelModel(grid=self.grid,voxel_data=V)

def union_all(models):
    u = models[0]
    print(f"union_all #{0}: u.grid.sv: {u.grid.compute_size_vector()}")
    for i,m in enumerate(models[1:]):
        #LOGGER.debug(f"union_all #{i}: u.grid.sv: {u.grid.compute_size_vector()}")
        u |= m
        print(f"union_all #{i+1}: u.grid.sv: {u.grid.compute_size_vector()}")
        
    return u