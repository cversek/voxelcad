from typing import OrderedDict
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
                 pvmesh = None,
                 ):
        self.grid = grid
        self.voxel_data   = voxel_data
        self.surface_data = surface_data
        self.mesh_data    = mesh_data
        self.pvmesh       = pvmesh
        
    def construct_grid(self):
        raise NotImplementedError()
            
    def render_volume(self):
        if self.grid is None:
            self.construct_grid()
        
    def render_surface(self, cache = True):
        #get from cache if already computed
        if cache and self.surface_data is not None:
            return self.surface_data
        # REF: https://forum.freecadweb.org/viewtopic.php?t=19819#p233282
        #      https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html
        from skimage import measure
        #render the voxels first
        if self.voxel_data is None:
            self.render_volume()
        #use the marching cubes algo. to create a surface mesh
        sv = self.grid.compute_size_vector()
        rv = self.grid.res_vector
        spacing = tuple(sv/rv)
        verts, faces, normals, values = measure.marching_cubes(self.voxel_data, 0, spacing=spacing) # , normals, values <<need to add these after faces if newer version of skimage
        #compute new bounds
        x0,x1 = (verts[:,0].min(),verts[:,0].max())
        y0,y1 = (verts[:,1].min(),verts[:,1].max())
        z0,z1 = (verts[:,2].min(),verts[:,2].max())
        #compute center vector of the new bounds
        cv_bounds = np.array(((x0+x1)/2,(y0+y1)/2,(z0+z1)/2))
        cv_grid   = self.grid.compute_center_vector()
        #move points to the grid space
        verts   += (cv_grid - cv_bounds)
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

    def render_pyvista_mesh(self, cache=True):
        #REF https://stackoverflow.com/questions/6030098/how-to-display-a-3d-plot-of-a-3d-array-isosurface-in-matplotlib-mplot3d-or-simil/35472146
        #get from cache if already computed
        if cache and self.pvmesh is not None:
            return self.pvmesh
        import pyvista as pv
        #render the voxels first
        if self.voxel_data is None:
            self.render_volume()
        #get a numpy array of the grid points
        X,Y,Z = self.grid.construct_mesh(make_empty_voxels=False)
        m = self.grid.margin
        V = self.voxel_data[m:-m,m:-m,m:-m]
        grid = pv.StructuredGrid(X, Y, Z)
        grid.point_data['vol'] = V.flatten()
        DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        if cache:
            self.pvmesh = pvmesh
        return pvmesh
        
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
        
    def export(self, filename):
        if filename.endswith(".nii"): #NIfTi
            import nibabel as nib
            xform = np.eye(4) * 2
            img = nib.nifti1.Nifti1Image(1.0*self.voxel_data, xform)
            nib.save(img,filename)
        elif filename.endswith(".stl"): #STL for 3d Printing
            rendered_mesh = self.render_mesh()
            rendered_mesh.save(filename)
        elif filename.endswith(".png"): #using matplotlib 
            fig = self.plot(show=False)
            fig.savefig(filename)
            plt.close(fig)
            
    def test_points(self, X, Y, Z):
        """ test if points defined by mesh X, Y, Z are in the volume
        """
        #LOGGER.debug(f"{self.__class__} -> {__class__}.test_points")
        #LOGGER.debug(f"test_points: X.min()={X.min()},  X.max()={X.max()}")
        #LOGGER.debug(f"test_points: Y.min()={Y.min()},  Y.max()={Y.max()}")
        #LOGGER.debug(f"test_points: Z.min()={Z.min()},  Z.max()={Z.max()}")
        x0,x1 = self.grid.xlim; y0,y1 = self.grid.ylim; z0,z1 = self.grid.zlim
        rx, ry, rz = self.grid.res_vector
        #first test if the points are within the bounding box of the grid
        in_bounds = (x0 <= X) & (X <= x1) & (y0 <= Y) & (Y <= y1) & (z0 <= Z) & (Z <= z1)
        #transform into data indices, giving dummy values (-1) to points outside bounds
        i_test = np.round(rx*(X-x0)/(x1-x0)).astype('int')
        j_test = np.round(ry*(Y-y0)/(y1-y0)).astype('int')
        k_test = np.round(rz*(Z-z0)/(z1-z0)).astype('int')
        #LOGGER.debug(f"test_points: i_test.min()={i_test.min()},  i_test.max()={i_test.max()}")
        #LOGGER.debug(f"test_points: j_test.min()={j_test.min()},  j_test.max()={j_test.max()}")
        #LOGGER.debug(f"test_points: k_test.min()={k_test.min()},  k_test.max()={k_test.max()}")
        #filter based on bounds and index cheking
        I = np.where(in_bounds & (0 <= i_test) & (i_test < rx),i_test,-1)
        J = np.where(in_bounds & (0 <= j_test) & (j_test < ry),j_test,-1)
        K = np.where(in_bounds & (0 <= k_test) & (k_test < rz),k_test,-1)
        #LOGGER.debug(f"test_points: I.min()={I.min()},  I.max()={I.max()}")
        #LOGGER.debug(f"test_points: J.min()={J.min()},  J.max()={J.max()}")
        #LOGGER.debug(f"test_points: K.min()={K.min()},  K.max()={K.max()}")
        #use indices to interpolate the voxel data between the margins
        if self.voxel_data is None:  #render the voxels first
            self.render_volume()
        m = self.grid.margin
        V = self.voxel_data[m:-m,m:-m,m:-m]
        #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        in_volume = np.where((I >=0) & (J >=0) & (K >=0),V[I,J,K],False)
        return in_volume

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

def union_all(models):
    u = models[0]
    for i,m in enumerate(models[1:]):
        LOGGER.debug(f"union_all #{i}: u.grid.sv: {u.grid.compute_size_vector()}")
        u |= m
        
    return u



def voxel_fuzz(V, nseeds=1000, iters=5):
    #get list of all solid voxels
    I = np.argwhere(V == True)
    #choose solid voxels at random
    indices = np.random.choice(I.shape[0],size=nseeds)
    seed_locs = I[indices,:]
    #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
    for i in range(iters):
        next_seed_locs = []
        for seed_loc in seed_locs:
            try:
                x,y,z = seed_loc
                #look at nearest neighbors and find the normal direction
                Vnn = 1*V[x-1:x+2,y-1:y+2,z-1:z+2] #convert to 1 and 0
                dx = Vnn[2,1,1] - Vnn[0,1,1]
                dy = Vnn[1,2,1] - Vnn[1,0,1]
                dz = Vnn[1,1,2] - Vnn[1,1,0]
                dxy = Vnn[2,2,1] - Vnn[0,0,1]
                dxz = Vnn[2,1,2] - Vnn[0,1,0]
                dyz = Vnn[1,2,2] - Vnn[1,0,0]
                dxyz = Vnn[2,2,2] - Vnn[0,0,0]
                dn = -np.array((dx+dxy+dxz+dxyz,dy+dxy+dyz+dxyz,dz+dxz+dyz+dxyz))
                #grow the volume in the normal direction
                x += dn[0]
                y += dn[1]
                z += dn[2]
                V[x,y,z] = True
                next_seed_locs.append((x,y,z))
            except IndexError:
                pass #ignore boundry issues
        seed_locs = next_seed_locs
    #DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
    return V


