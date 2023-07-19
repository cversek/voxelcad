import os, time
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
        V = self.voxel_data
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
        t0 = time.time()
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
        t1 = time.time()
        LOGGER.info(f"END render_volume_mesh, time: {t1-t0:0.1f} s")
        mem = MEMORY_USAGE(offset=mem0)
        LOGGER.info(f"DELTA MEMORY USED: {mem/2**30:0.2f} GB")
        LOGGER.info(40*"*")
        return pv_vol

    def render_surface_mesh(self, 
                            cache=True,
                            use_meshfix = True,
                            smooth_iters = 0,
                            downscale_times = 0,
                            only_largest_component = False,
                            ):
        #REF https://stackoverflow.com/questions/6030098/how-to-display-a-3d-plot-of-a-3d-array-isosurface-in-matplotlib-mplot3d-or-simil/35472146
        #REF https://docs.pyvista.org/examples/00-load/create-uniform-grid.html
        t0 = time.time()
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
        pv_surf = pv_vol.extract_surface() #(nonlinear_subdivision=5)  #FIXME
        #do filtering steps
        if use_meshfix:
            _t0 = time.time()
            LOGGER.info(f"\trunning meshfix.repair()...")
            #repair holes in mesh
            import pymeshfix as mf
            meshfix = mf.MeshFix(pv_surf)
            meshfix.repair(verbose=True, joincomp=True)
            pv_surf = meshfix.mesh
            LOGGER.info(f"\t...completed in {time.time()-_t0:0.1f} s")
        if smooth_iters > 0:
            _t0 = time.time()
            LOGGER.info(f"\trunning smooth...")
            pv_surf = pv_surf.smooth(n_iter=smooth_iters,progress_bar=True)
            LOGGER.info(f"\t...completed in {time.time()-_t0:0.1f} s")
        if downscale_times > 0:
            _t0 = time.time()
            LOGGER.info(f"\trunning downscale_trimesh...")
            #we use pyvista to clean up the mesh
            pv_surf = pv_surf.triangulate() #must be triangulated
            from voxelcad.utils.pyvista_tools import downscale_trimesh
            #cut the number of triangles down by 2**3 times
            pv_surf = downscale_trimesh(pv_surf,smooth_iters=smooth_iters,repeat=downscale_times,decimation_factor=0.5)
            LOGGER.info(f"\t...completed in {time.time()-_t0:0.1f} s")
            if use_meshfix:
                _t0 = time.time()
                LOGGER.info(f"\trunning meshfix.repair() again after downscale...")
                #repair holes in mesh
                import pymeshfix as mf
                meshfix = mf.MeshFix(pv_surf)
                meshfix.repair(verbose=True, joincomp=True)
                pv_surf = meshfix.mesh
                LOGGER.info(f"\t...completed in {time.time()-_t0:0.1f} s")
        if only_largest_component:
            pv_surf = pv_surf.extract_largest()
        if cache:
            self.pv_surf = pv_surf
        t1 = time.time()
        LOGGER.info(f"END render_surface_mesh, time: {t1-t0:0.1f} s")
        mem = MEMORY_USAGE(offset=mem0)
        LOGGER.info(f"DELTA MEMORY USED: {mem/2**30:0.2f} GB")
        LOGGER.info(40*"*")
        return pv_surf
        
    def plot(self, *args,**kwargs):
        vol_mesh = self.render_volume_mesh()
        kwargs['color'] = kwargs.get('color','white') #provide default
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
        #transform into data indices, giving dummy values (-1) to points outside bounds
        I,J,K = self.index_transform(X,Y,Z)
        #use indices to interpolate the voxel data between the margins
        if self.voxel_data is None:  #render the voxels first
            self.render_volume()
        V = self.voxel_data
        in_volume = np.where((I >=0) & (J >=0) & (K >=0),V[I,J,K],False)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        LOGGER.info(f"END test_points")
        mem = MEMORY_USAGE(offset=mem0)
        LOGGER.info(f"DELTA MEMORY USED: {mem/2**30:0.2f} GB")
        LOGGER.info(40*"*")
        return in_volume

    def index_transform(self, X, Y, Z):
        LOGGER.info(40*"*")
        LOGGER.info(f"{self.__class__} -> super().index_transform")
        mem0 = MEMORY_USAGE()
        LOGGER.info(f"TOTAL MEMORY USED: {mem0/2**30:0.2f} GB")
        LOGGER.info(40*"-")
        LOGGER.debug(f"test_points: X.min()={X.min()},  X.max()={X.max()}")
        LOGGER.debug(f"test_points: Y.min()={Y.min()},  Y.max()={Y.max()}")
        LOGGER.debug(f"test_points: Z.min()={Z.min()},  Z.max()={Z.max()}")
        #we write the transormation as a function to better bound memory usage requirements
        def transform_coord_to_index(C,rc,c0,c1):
            i_test = np.floor(rc*(C-c0)/(c1-c0)).astype('int')
            #filter based on bounds and index checking
            I = np.where((0 <= i_test) & (i_test < rc),i_test,-1)
            LOGGER.debug(f"test_points: I.min()={I.min()},  I.max()={I.max()}")
            return I
        #transform into data indices, giving dummy values (-1) to points outside bounds
        x0,x1 = self.grid.xlim; y0,y1 = self.grid.ylim; z0,z1 = self.grid.zlim
        rx, ry, rz = self.grid.res_vector
        I = transform_coord_to_index(X,rx,x0,x1)
        J = transform_coord_to_index(Y,ry,y0,y1)
        K = transform_coord_to_index(Z,rz,z0,z1)
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
                             )
        vm = VoxelModel(grid=new_grid,voxel_data=self.voxel_data)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return vm
        
    def rotate_x(self, degrees, **kwargs):
        return self.rotate([1.0,0,0], degrees, **kwargs)

    def rotate_y(self, degrees, **kwargs):
        return self.rotate([0,1.0,0], degrees, **kwargs)

    def rotate_z(self, degrees, **kwargs):
        return self.rotate([0,0,1.0], degrees, **kwargs)

    def rotate(self, v, degrees, return_matrices=False):
        #REF: https://en.wikipedia.org/wiki/Rotation_matrix 
        #    "Rotation matrix from axis and angle"
        #normalize vector
        v = np.array(v)
        u = v/np.linalg.norm(v)
        ux,uy,uz = u
        #construct the rotation matrix and its inverse
        theta = np.radians(degrees)
        c, s = np.cos(theta), np.sin(theta)
        omc = 1 - c
        Uxx = ux*ux*omc
        Uyy = uy*uy*omc
        Uzz = uz*uz*omc
        Uxy = ux*uy*omc
        Uxz = ux*uz*omc
        Uyz = uy*uz*omc
        R    = np.array(((c  + Uxx, Uxy-uz*s, Uxz+uy*s),
                         (Uxy+uz*s, c  + Uyy, Uyz-ux*s),
                         (Uxz-uy*s, Uyz+ux*s, c  + Uzz)))
        #to get inverse, subs -theta in sin, thus s -> -s
        Rinv = np.array(((c  + Uxx, Uxy+uz*s, Uxz-uy*s),
                         (Uxy-uz*s, c  + Uyy, Uyz+ux*s),
                         (Uxz+uy*s, Uyz-ux*s, c  + Uzz)))
        if return_matrices:
            return (R,Rinv)
        else:
            m = self.apply_transformation(R,Rinv)
            LOGGER.debug(f"rotate: m.shape: {m.voxel_data.shape}")
            LOGGER.debug(f"\tm.sum: {m.voxel_data.sum()}")
            LOGGER.debug(f"\tm.grid: {m.grid!r}")
            return m

    def scale(self, v):
        v = v*np.ones(3,dtype="float32")
        #construct the scaling matrix and its inverse
        S    = np.array(((v[0],0,0),(0,v[1],0),(0,0,v[2])))
        Sinv = np.array(((1.0/v[0],0,0),(0,1.0/v[1],0),(0,0,1.0/v[2])))
        m = self.apply_transformation(S,Sinv)
        LOGGER.debug(f"scale: m.shape: {m.voxel_data.shape}")
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
        new_grid = VoxelGrid(xlim,ylim,zlim,voxel_size=self.grid.voxel_size_vector)
        Xt,Yt,Zt = new_grid.construct_mesh()
        #invert the rotation to map the mesh points back to the orginal data space
        X = Minv[0,0]*Xt + Minv[0,1]*Yt + Minv[0,2]*Zt
        Y = Minv[1,0]*Xt + Minv[1,1]*Yt + Minv[1,2]*Zt
        Z = Minv[2,0]*Xt + Minv[2,1]*Yt + Minv[2,2]*Zt
        #test if the new mesh points are contained in the volume
        Vt = self.test_points(X,Y,Z)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return VoxelModel(grid=new_grid,voxel_data=Vt)
        
    def __or__(self, other): #union
        if self.voxel_data is None:
            self.render_volume()
        if other.voxel_data is None:
            other.render_volume()
        #compute bounding voxel grid
        bounding_grid = self.grid | other.grid
        X,Y,Z = bounding_grid.construct_mesh(make_empty_voxels=False)
        #test if the new mesh points are contained in either of the respective volumes 
        #and fill within the margins
        V  = self.test_points(X,Y,Z)
        V |= other.test_points(X,Y,Z)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return VoxelModel(grid=bounding_grid,voxel_data=V)
         
    def __and__(self, other): #intersection
        if self.voxel_data is None:
            self.render_volume()
        if other.voxel_data is None:
            other.render_volume()
        #compute bounding voxel grid
        bounding_grid = self.grid & other.grid
        X,Y,Z = bounding_grid.construct_mesh()
        #test if the new mesh points are contained in both of the respective volumes 
        V  = self.test_points(X,Y,Z)
        V &= other.test_points(X,Y,Z)
        return VoxelModel(grid=bounding_grid,voxel_data=V)

    def __xor__(self, other): #exclusive or
        if self.voxel_data is None:
            self.render_volume()
        if other.voxel_data is None:
            other.render_volume()
        #compute bounding voxel grid, same as union
        bounding_grid = self.grid | other.grid
        X,Y,Z = bounding_grid.construct_mesh(make_empty_voxels=False)
        #test if the new mesh points are contained in either of the respective volumes 
        #and fill within the margins
        V  = self.test_points(X,Y,Z)
        V ^= other.test_points(X,Y,Z)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return VoxelModel(grid=bounding_grid,voxel_data=V)

    def __sub__(self, other): #difference
        if self.voxel_data is None:
            self.render_volume()
        if other.voxel_data is None:
            other.render_volume()
        #compute bounding voxel grid
        X,Y,Z = self.grid.construct_mesh()
        #test if the new mesh points are contained in the first but not the second volume
        V  =  self.test_points(X,Y,Z)
        V &= ~other.test_points(X,Y,Z)
        return VoxelModel(grid=self.grid,voxel_data=V)

def union_all(models):
    u = models[0]
    print(f"union_all #{0}: u.grid.sv: {u.grid.compute_size_vector()}")
    for i,m in enumerate(models[1:]):
        #LOGGER.debug(f"union_all #{i}: u.grid.sv: {u.grid.compute_size_vector()}")
        u |= m
        print(f"union_all #{i+1}: u.grid.sv: {u.grid.compute_size_vector()}")
        
    return u