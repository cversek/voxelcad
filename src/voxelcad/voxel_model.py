import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from voxelcad.debug import currentframe, DEBUG_TAG, DEBUG_EMBED
from voxelcad.voxel_grid import VoxelGrid


class VoxelModel:
    def __init__(self, grid = None, voxel_data = None):
        self.grid = grid
        self.voxel_data = voxel_data
        
    def construct_grid(self):
        raise NotImplementedError()
            
    def render_volume(self):
        if self.grid is None:
            self.construct_grid()
        
    def render_surface(self):
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
        return (verts, faces, normals, values)
        
    def render_mesh(self):
        verts, faces, normals, values = self.render_surface()
        import stl
        from stl import mesh
        data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
        rendered_mesh = mesh.Mesh(data, remove_empty_areas=False)
        for i, f in enumerate(faces):
            for j in range(3):
                rendered_mesh.vectors[i][j] = verts[f[j],:]
        return rendered_mesh
        
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
            mesh = self.render_mesh()
            #setup vtkplot figure
            if figure is None:
                figure = vpl.figure()
            #plot the style
            vpl.mesh_plot(mesh)
            #show if requested
            if kwargs.get('show',True):
                vpl.show(block=kwargs.get('block',True)) #FIXME blocking is needed for interaction, limitation of vpl?
            return figure

        elif style in ['trisurf','poly3d']:
            import matplotlib.pyplot as plt
            #render surface data
            verts, faces, normals, values = self.render_surface()
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
        """ test if points defined by mesh X, Y, Z are in bounds
        """
        x0,x1 = self.grid.xlim; y0,y1 = self.grid.ylim; z0,z1 = self.grid.zlim
        r_x, r_y, r_z = self.res_vector
        #first test if the points are within the bounding box of the grid
        in_bounds = (x0 < X) & (X < x1) & (y0 < Y) & (Y < y1) & (z0 < Z) & (Z < z1)
        #transform into data indices, giving dummy values (-1) to points outside bounds
        I = np.where(in_bounds, np.floor(r_x*(X-x0)/(x1-x0)).astype('int'),-1)
        J = np.where(in_bounds, np.floor(r_y*(Y-y0)/(y1-y0)).astype('int'),-1)
        K = np.where(in_bounds, np.floor(r_z*(Z-z0)/(z1-z0)).astype('int'),-1)
        #use indices to interpolate the voxel data between the margins
        if self.voxel_data is None:  #render the voxels first
            self.render_volume()
        m = self.grid.margin
        V = self.voxel_data[m:-m,m:-m,m:-m]
        in_volume = np.where(in_bounds,V[I,J,K],False)
        return in_volume

    def rotate_z(self, degrees):
        #construct the rotation matrix and its inverse
        theta = np.radians(degrees)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((0,c,-s),(0,s,c),(0,0,1.0)))
        #rotate the bounding box corners
        x0,x1 = self.grid.xlim; y0,y1 = self.grid.ylim; z0,z1 = self.grid.zlim
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
        Cr = np.dot(C,R.T) #apply rotation matrix
        
        DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        #create a new rotated grid with maximal resolution
        r_x, r_y, r_z = self.res_vector
        r_max = max(r_x, r_y, r_z)
        xlim,ylim,zlim = (v0r[0],v1r[0]),(v0r[1],v1r[1]),(v0r[2],v1r[2])
        rot_grid = VoxelGrid(xlim,ylim,zlim,res=r_max)
        Xr,Yr,Zr = rot_grid.construct_mesh(make_empty_voxels=False)
        #invert the rotation to map back to the orginal data space
        c_inv, s_inv = np.cos(-theta), np.sin(-theta)
        X =  c_inv*Xr + s_inv*Yr  # "clockwise"
        Y = -s_inv*Xr + c_inv*Yr
        Z = Zr
        #test if the new mesh points are contained in the volume
        rot_voxel_data = self.test_points(X,Y,Z)
        DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals())
        return VoxelModel(grid=rot_grid,voxel_data=rot_voxel_data)
        


            
    def __or__(self, other): #union
        self.render_volume()  #FIXME this should only be done in necessary
        other.render_volume() #FIXME this should only be done in necessary
        #compute bounding voxel grid
        bounding_grid = self.grid | other.grid
        X,Y,Z = bounding_grid.construct_mesh(make_empty_voxels=False)
        #test if the new mesh points are contained in either of the respective volumes 
        voxel_data = self.test_points(X,Y,Z) | other.test_points(X,Y,Z)
        return VoxelModel(grid=bounding_grid,voxel_data=voxel_data)
        
        
    def __and__(self, other): #intersection
        self.render_volume()   #FIXME this should only be done in necessary
        other.render_volume()  #FIXME this should only be done in necessary
        #compute bounding voxel grid
        bounding_grid = self.grid & other.grid
        X,Y,Z = bounding_grid.construct_mesh(make_empty_voxels=False)
        #test if the new mesh points are contained in both of the respective volumes 
        voxel_data = self.test_points(X,Y,Z) & other.test_points(X,Y,Z)
        return VoxelModel(grid=bounding_grid,voxel_data=voxel_data)