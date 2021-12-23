from typing import OrderedDict
from matplotlib.pyplot import margins
import numpy as np

import voxelcad.environment as ENV

from voxelcad.debug import currentframe, DEBUG_TAG, DEBUG_EMBED
from voxelcad.voxel_grid import VoxelGrid

import logging
LOGGER = logging.getLogger(__name__)

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
        
    def construct_grid(self):
        raise NotImplementedError()
            
    def render_volume(self):
        if self.grid is None:
            self.construct_grid()
        
    def render_surface(self, cache = True, use_smoothing = False, lib=None):
        #get from cache if already computed
        if cache and self.surface_data is not None:
            return self.surface_data
        # REF: https://forum.freecadweb.org/viewtopic.php?t=19819#p233282
        #      https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html
        # REF: https://github.com/pmneila/PyMCubes
        #render the voxels first
        if self.voxel_data is None:
            self.render_volume()
        #use the marching cubes algo. to create a surface mesh
        sv = self.grid.compute_size_vector()
        rv = self.grid.res_vector
        spacing = tuple(sv/rv)
        V = self.voxel_data
        if use_smoothing and (lib is None or lib=="mcubes"):
            import mcubes
            V = mcubes.smooth(V)
            lib = "mcubes"
        elif use_smoothing:
            raise ValueError("can only use smoothing with lib='mcubes'")
        if lib is None:
            lib = "skimage" #default to faster implementation
        verts,faces,normals,values = (None,None,None,None)
        if lib == "mcubes":
            import mcubes
             # Extract the 0-levelset (the 0-levelset of the output of mcubes.smooth is the
            # smoothed version of the 0.5-levelset of the binary array).
            verts, faces = mcubes.marching_cubes(V, 0)
        elif lib == "skimage":
            from skimage.measure import marching_cubes
            verts, faces, normals, values = marching_cubes(self.voxel_data,level=0, spacing=spacing) # , normals, values <<need to add these after faces if newer version of skimage
        else:
            raise ValueError(f"lib={lib!r} is not a valid choice, try 'skimage' or 'mcubes")
        #compute new bounds
        x0,x1 = (verts[:,0].min(),verts[:,0].max())
        y0,y1 = (verts[:,1].min(),verts[:,1].max())
        z0,z1 = (verts[:,2].min(),verts[:,2].max())
        #compute center vector of the new bounds
        cv_bounds = np.array(((x0+x1)/2,(y0+y1)/2,(z0+z1)/2))
        cv_grid   = self.grid.compute_center_vector()
        #move points to the grid space
        verts   += (cv_grid - cv_bounds)
        if normals is not None:
            normals += (cv_grid - cv_bounds)
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        out_dict = OrderedDict()
        out_dict['verts']   = verts
        out_dict['faces']   = faces
        out_dict['normals'] = normals
        out_dict['values']  = values
        if cache:
            self.surface_data = out_dict
        return out_dict
        
    def render_mesh(self, cache=True):
        #get from cache if already computed
        if cache and self.mesh_data is not None:
            return self.mesh_data
        surface_data = self.render_surface()
        verts = surface_data['verts']
        faces = surface_data['faces']
        import stl
        from stl import mesh
        data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
        mesh_data = mesh.Mesh(data, remove_empty_areas=False)
        for i, f in enumerate(faces):
            for j in range(3):
                mesh_data.vectors[i][j] = verts[f[j],:]
        if cache:
            self.mesh_data = mesh_data
        return mesh_data

    def render_pyvista_volume_mesh(self, cache=True):
        #REF https://stackoverflow.com/questions/6030098/how-to-display-a-3d-plot-of-a-3d-array-isosurface-in-matplotlib-mplot3d-or-simil/35472146
        #REF https://docs.pyvista.org/examples/00-load/create-uniform-grid.html

        #get from cache if already computed
        if cache and self.pv_vol is not None:
            return self.pv_vol
        import pyvista as pv
        #render the voxels first
        if self.voxel_data is None:
            self.render_volume()
        #get a numpy array of the grid points
        X,Y,Z = self.grid.construct_mesh(make_empty_voxels=False)
        m = self.grid.margin
        V = self.voxel_data[m:-m,m:-m,m:-m]
        rv  = self.grid.res_vector
        vsv = self.grid.voxel_size_vector
        pv_grid = pv.UniformGrid()
        # Set the grid dimensions: shape + 1 because we want to inject our values on
        #   the CELL data
        pv_grid.dimensions = rv + 1
        # Edit the spatial reference
        #grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
        pv_grid.spacing = vsv  # These are the cell sizes along each axis
        pv_grid.cell_data['vol'] = 255.0*V.flatten(order="F") #NOTE column-major (Fortran) order must be specified!
        pv_vol = pv_grid.threshold(128) #convert to unstructured grid of just the solid areas
        DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        if cache:
            self.pv_vol = pv_vol
        return self.pv_vol

    def render_pyvista_surface_mesh(self, 
                                    cache=True,
                                    surf_render_lib='skimage',
                                    smooth_iters = 0,
                                    downscale_times = 0,
                                    only_largest_component = False,
                                    ):
        #REF https://stackoverflow.com/questions/6030098/how-to-display-a-3d-plot-of-a-3d-array-isosurface-in-matplotlib-mplot3d-or-simil/35472146
        #REF https://docs.pyvista.org/examples/00-load/create-uniform-grid.html

        #get from cache if already computed
        if cache and self.pv_surf is not None:
            return self.pv_surf
        import pyvista as pv
        #render the voxels first
        if self.voxel_data is None:
            self.render_volume()
        surf = self.render_surface(lib=surf_render_lib)
        verts = surf['verts']
        faces = surf['faces']
        #format triangular faces as [3,p1,p2,p4] for PolyData
        faces = np.vstack((3*np.ones(faces.shape[0],dtype='int32'),faces.T)).T
        pv_surf = pv.PolyData(verts,faces=faces)
        #do filtering steps
        if smooth_iters > 0:
            pv_surf = pv_surf.smooth(n_iter=smooth_iters,progress_bar=True)
        if downscale_times > 0:
            #we use pyvista to clean up the mesh
            from voxelcad.utils.pyvista_tools import downscale_trimesh
            #cut the number of triangles down by 2**3 times
            pv_surf = downscale_trimesh(pv_surf,repeat=downscale_times,decimation_factor=0.5)
        if only_largest_component:
            pv_surf = pv_surf.extract_largest()
        if cache:
            self.pv_surf = pv_surf
        return self.pv_surf
        
    def plot(self, style = None, axis = None, figure = None,**kwargs):
        if style is not None:
            self._plot_style(style,figure,axis)
        else:
            #try plotting in preference order, with fallback
            for style in ['vtk-mesh','trisurf','poly3d']:
                try:
                    figure = self._plot_style(style,figure,axis,**kwargs)
                    return figure
                except Exception as exc:
                    LOGGER.warning(f"plot: caught exception '{exc}', continuing...")

    def _plot_style(self,style,figure,axis,**kwargs):
        #plot in the chosen style
        if style == 'vtk-mesh':
            import vtkplotlib as vpl
            #render mesh data
            mesh_data = self.render_mesh()
            #setup vtkplot figure
            if figure is None:
                figure = vpl.figure()
            #plot the style
            vpl.mesh_plot(mesh_data)
            #show if requested
            if kwargs.get('show',True):
                vpl.show(block=kwargs.get('block',True)) #FIXME blocking is needed for interaction, limitation of vpl?
            return figure

        elif style in ['trisurf','poly3d']:
            import matplotlib.pyplot as plt
            #render surface data
            surface_data = self.render_surface()
            verts = surface_data['verts']
            faces = surface_data['faces']
            #setup matplotlib figure and axis
            if axis is None and figure is None:
                figure = plt.figure()
            if axis is None:
                axis = figure.add_subplot(111, projection='3d')
            #plot the style
            if style == 'trisurf':
                axis.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='ocean', lw=1)
            elif style == 'poly3d': #REF: https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                mesh = Poly3DCollection(verts[faces])
                mesh.set_edgecolor(edgecolor)
                axis.add_collection3d(mesh)
                axis.set_xlim(*self.grid.xlim)  
                axis.set_ylim(*self.grid.ylim)
                axis.set_zlim(*self.grid.zlim)   
            #finish formating axes
            if kwargs.get('show',True):
                plt.show()
            return figure
        
    def export(self, filename, pvfilter = False):
        if filename.endswith(".nii"): #NIfTi
            import nibabel as nib
            xform = np.eye(4) * 2
            img = nib.nifti1.Nifti1Image(1.0*self.voxel_data, xform)
            nib.save(img,filename)
        elif filename.endswith(".stl"): #STL for 3d Printing
            if not pvfilter:
                rendered_mesh = self.render_mesh()
                rendered_mesh.save(filename)
            else:
                pv_surf = self.render_pyvista_surface_mesh(filter=pvfilter)
               

        elif filename.endswith(".png"): #using matplotlib 
            fig = self.plot(show=False)
            fig.savefig(filename)
            plt.close(fig)
            
    def test_points(self, X, Y, Z):
        """ test if points defined by mesh X, Y, Z are in the volume
        """
        LOGGER.debug(f"{self.__class__} -> {__class__}.test_points")
        LOGGER.debug(f"test_points: X.min()={X.min()},  X.max()={X.max()}")
        LOGGER.debug(f"test_points: Y.min()={Y.min()},  Y.max()={Y.max()}")
        LOGGER.debug(f"test_points: Z.min()={Z.min()},  Z.max()={Z.max()}")
        x0,x1 = self.grid.xlim; y0,y1 = self.grid.ylim; z0,z1 = self.grid.zlim
        rx, ry, rz = self.grid.res_vector
        #first test if the points are within the bounding box of the grid
        in_bounds = (x0 < X) & (X < x1) & (y0 < Y) & (Y < y1) & (z0 < Z) & (Z < z1)
        #transform into data indices, giving dummy values (-1) to points outside bounds
        i_test = np.round(rx*(X-x0)/(x1-x0)).astype('int')
        j_test = np.round(ry*(Y-y0)/(y1-y0)).astype('int')
        k_test = np.round(rz*(Z-z0)/(z1-z0)).astype('int')
        LOGGER.debug(f"test_points: i_test.min()={i_test.min()},  i_test.max()={i_test.max()}")
        LOGGER.debug(f"test_points: j_test.min()={j_test.min()},  j_test.max()={j_test.max()}")
        LOGGER.debug(f"test_points: k_test.min()={k_test.min()},  k_test.max()={k_test.max()}")
        #filter based on bounds and index checking
        I = np.where(in_bounds & (0 <= i_test) & (i_test < rx),i_test,-1)
        J = np.where(in_bounds & (0 <= j_test) & (j_test < ry),j_test,-1)
        K = np.where(in_bounds & (0 <= k_test) & (k_test < rz),k_test,-1)
        LOGGER.debug(f"test_points: I.min()={I.min()},  I.max()={I.max()}")
        LOGGER.debug(f"test_points: J.min()={J.min()},  J.max()={J.max()}")
        LOGGER.debug(f"test_points: K.min()={K.min()},  K.max()={K.max()}")
        #use indices to interpolate the voxel data between the margins
        if self.voxel_data is None:  #render the voxels first
            self.render_volume()
        m = self.grid.margin
        V = self.voxel_data[m:-m,m:-m,m:-m]
        in_volume = np.where((I >=0) & (J >=0) & (K >=0),V[I,J,K],False)
        return in_volume

    def translate(self, v):
        if self.voxel_data is None:  #render the voxels first
            self.render_volume()
        new_grid = VoxelGrid(xlim=self.grid.xlim + v[0],
                             ylim=self.grid.ylim + v[1],
                             zlim=self.grid.zlim + v[2],
                             res=self.grid.res_vector,
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
        V[m:-m,m:-m,m:-m] = self.test_points(X,Y,Z) | other.test_points(X,Y,Z)
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
    for i,m in enumerate(models[1:]):
        LOGGER.debug(f"union_all #{i}: u.grid.sv: {u.grid.compute_size_vector()}")
        u |= m
        
    return u