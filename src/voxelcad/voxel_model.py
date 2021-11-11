import numpy as np

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
        from skimage import measure

        if self.voxel_data is None:
            self.render_volume()
        sv = self.grid.compute_size_vector()
        rv = self.grid.res_vector
        spacing = tuple(sv/rv)
        verts, faces, normals, values = measure.marching_cubes(self.voxel_data, 0, spacing=spacing) # , normals, values <<need to add these after faces if newer version of skimage
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
        
    def plot(self, show=True):
        verts, faces, normals, values = self.render_surface()
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='ocean', lw=1)
        if show:
            plt.show()
        return fig
        
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
        #use indices to interpolate the voxel data
        in_volume = np.where(in_bounds,self.voxel_data[I,J,K],False)
        return in_volume
            
    def __or__(self, other): #union
        self.render_volume()  #FIXME this should only be done in necessary
        other.render_volume() #FIXME this should only be done in necessary
        #compute bounding voxel grid
        bounding_grid = self.grid | other.grid
        X,Y,Z = bounding_grid.construct_mesh()
        #test if the new mmesh points are contain in either of the respective volumes 
        voxel_data = self.test_points(X,Y,Z) | other.test_points(X,Y,Z)
        return VoxelModel(grid=bounding_grid,voxel_data=voxel_data)
        
        
    def __and__(self, other): #intersection
        self.render_volume()   #FIXME this should only be done in necessary
        other.render_volume()  #FIXME this should only be done in necessary
        #compute bounding voxel grid
        bounding_grid = self.grid & other.grid
        X,Y,Z = bounding_grid.construct_mesh()
        #test if the new mmesh points are contain in both of the respective volumes 
        voxel_data = self.test_points(X,Y,Z) & other.test_points(X,Y,Z)
        return VoxelModel(grid=bounding_grid,voxel_data=voxel_data)