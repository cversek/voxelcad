import os, time
from dataclasses import dataclass, field
from typing import OrderedDict, List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import pyvista as pv

import voxelcad.environment as ENV

from voxelcad.debug import create_logger, currentframe, DEBUG_TAG, DEBUG_EMBED, MEMORY_USAGE, TIMING_START, TIMING_END
LOGGER = create_logger(__name__)

from voxelcad.voxel_grid import VoxelGrid, UniformGrid
from voxelcad._kernels import CYTHON_AVAILABLE


class LeafType(Enum):
    """Classification of CSG tree leaf nodes for execution planning."""
    FUSED = "fused"           # Primitive with Cython kernel available
    MATERIALIZED = "materialized"  # Already has packed voxel_data
    FALLBACK = "fallback"     # TransformedModel or requires per-slice streaming


@dataclass
class LeafNode:
    """A leaf in the CSG tree with its classification and grid info."""
    model: Any  # VoxelModel subclass
    leaf_type: LeafType
    grid: 'VoxelGrid'

    @property
    def voxel_size(self) -> np.ndarray:
        return self.grid.voxel_size_vector


@dataclass
class ExecutionPlan:
    """Query plan for optimized CSG tree execution.

    Analyzes the CSG tree structure and determines:
    - Which leaves can use fused Cython kernels
    - Which subtrees share compatible grids (can use byte-level ops)
    - The optimal execution order

    Attributes:
        leaves: All leaf nodes in tree traversal order
        operations: Internal CSG operations in post-order
        all_compatible: True if all leaves share compatible voxel_size
        all_fused_capable: True if all leaves can use Cython kernels
        common_grid: Union grid if all_compatible, else None
        strategy: 'fused_bytewise' | 'mixed' | 'streaming'
    """
    leaves: List[LeafNode] = field(default_factory=list)
    operations: List[Tuple[str, int, int]] = field(default_factory=list)  # (op, left_idx, right_idx)
    all_compatible: bool = False
    all_fused_capable: bool = False
    common_grid: Optional['VoxelGrid'] = None
    strategy: str = "streaming"  # Default fallback

    def __repr__(self):
        return (f"ExecutionPlan(leaves={len(self.leaves)}, "
                f"strategy={self.strategy!r}, "
                f"all_compatible={self.all_compatible}, "
                f"all_fused_capable={self.all_fused_capable})")


class VoxelModel:
    def __init__(self,
                 grid = None,
                 voxel_data = None,
                 _voxel_shape = None,
                 surface_data = None,
                 mesh_data = None,
                 pv_vol = None,
                 pv_surf = None,
                 ):
        self.grid = grid
        # Auto-pack bool arrays for 8x storage reduction (F-order: Z-slices contiguous)
        if voxel_data is not None and voxel_data.dtype == np.bool_:
            self._voxel_shape = voxel_data.shape
            self.voxel_data = np.packbits(voxel_data.ravel(order='F'))
        else:
            self.voxel_data   = voxel_data
            self._voxel_shape = _voxel_shape
        self.surface_data = surface_data
        self.mesh_data    = mesh_data
        self.pv_vol       = pv_vol
        self.pv_surf      = pv_surf

    def _unpack_volume(self):
        """Unpack stored uint8 data back to bool array with original shape (F-order)."""
        n = int(np.prod(self._voxel_shape))
        return np.unpackbits(self.voxel_data)[:n].reshape(self._voxel_shape, order='F').view(np.bool_)

    def _unpack_slice(self, k):
        """Extract a single Z-slice from packed storage (F-order: Z-slices contiguous).

        Args:
            k: Z-index of the slice to extract

        Returns:
            2D boolean array of shape (rx, ry)
        """
        rx, ry, rz = self._voxel_shape
        slice_size = rx * ry
        start = k * slice_size
        byte_start = start // 8
        byte_end = (start + slice_size + 7) // 8
        bits = np.unpackbits(self.voxel_data[byte_start:byte_end])
        bit_offset = start - byte_start * 8
        return bits[bit_offset:bit_offset + slice_size].reshape(rx, ry, order='F').view(np.bool_)

    def evaluate_slice(self, X_2d, Y_2d, z_val):
        """Evaluate this model's volume for a single Z-slice.

        Subclasses override this to implement their geometry.
        Returns a 2D boolean array of shape (rx, ry).

        Args:
            X_2d: 2D array of X coordinates, shape (rx, ry)
            Y_2d: 2D array of Y coordinates, shape (rx, ry)
            z_val: scalar Z coordinate for this slice
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate_slice()"
        )

    def _has_evaluate_slice(self):
        """Check if this model has a real geometry evaluation function."""
        return type(self).evaluate_slice is not VoxelModel.evaluate_slice

    def _has_evaluate_at_coords(self):
        """Check if this model can evaluate geometry at arbitrary coordinates."""
        return type(self).evaluate_at_coords is not VoxelModel.evaluate_at_coords

    def evaluate_at_coords(self, X, Y, Z):
        """Evaluate geometry at arbitrary coordinate arrays.

        Unlike evaluate_slice() which takes a scalar z_val, this method
        accepts Z as an array of the same shape as X and Y. Enables
        TransformedModel to evaluate source primitives at inverse-transformed
        coordinates without intermediate volume allocation.

        Subclasses (primitives) should override this with their geometry formula.
        Default implementation falls back to per-voxel indexing into rendered volume.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate_at_coords()"
        )

    def _is_fused_capable(self):
        """Check if this model can use a Cython fused kernel.

        Override in primitive subclasses that have Cython kernels.
        Returns True only if CYTHON_AVAILABLE and primitive has a kernel.
        """
        return False

    def _classify_leaf(self) -> LeafNode:
        """Classify this model as a CSG tree leaf node.

        Returns:
            LeafNode with appropriate type classification
        """
        if self.voxel_data is not None:
            return LeafNode(self, LeafType.MATERIALIZED, self.grid)
        if self._is_fused_capable():
            return LeafNode(self, LeafType.FUSED, self.grid)
        return LeafNode(self, LeafType.FALLBACK, self.grid)

    def _get_slice_for_coords(self, X_2d, Y_2d, z_val):
        """Get a boolean slice for the given coordinates.

        If the model has evaluate_slice (primitives), uses it directly.
        If the model only has packed data (transformed models), renders
        if needed, then maps coordinates to indices and extracts bits
        from packed storage without full unpack.

        Args:
            X_2d: 2D array of X coordinates
            Y_2d: 2D array of Y coordinates
            z_val: scalar Z coordinate

        Returns:
            2D boolean array
        """
        if self._has_evaluate_slice():
            return self.evaluate_slice(X_2d, Y_2d, z_val)
        # Fallback: extract Z-slice from packed storage, then index with X,Y
        if self.voxel_data is None:
            self.render_volume()
        x0, x1 = self.grid.xlim
        y0, y1 = self.grid.ylim
        z0, z1 = self.grid.zlim
        rx, ry, rz = self.grid.res_vector
        k = int(np.floor(rz * (z_val - z0) / (z1 - z0)))
        if k < 0 or k >= rz:
            return np.zeros(X_2d.shape, dtype='bool')
        V_slice = self._unpack_slice(k)
        I = np.floor(rx * (X_2d - x0) / (x1 - x0)).astype('int')
        J = np.floor(ry * (Y_2d - y0) / (y1 - y0)).astype('int')
        valid = (I >= 0) & (I < rx) & (J >= 0) & (J < ry)
        I_safe = np.where(valid, I, 0)
        J_safe = np.where(valid, J, 0)
        return np.where(valid, V_slice[I_safe, J_safe], False)

    def render_volume(self):
        """Render volume using streaming Z-slice evaluation.

        Iterates over Z-slices, calling evaluate_slice() for each,
        and accumulates into a pre-allocated 3D boolean array.
        Returns existing data if already rendered.
        """
        if self.voxel_data is not None:
            return self.voxel_data
        if self.grid is None:
            self.construct_grid()
        TIMING_START("render_volume")
        LOGGER.info(40*"*")
        LOGGER.info(f"{self.__class__} -> render_volume (streaming)")
        mem0 = MEMORY_USAGE()
        LOGGER.info(f"TOTAL MEMORY USED: {mem0/2**30:0.2f} GB")
        LOGGER.info(40*"-")
        rx, ry, rz = self.grid.res_vector
        V = np.zeros((int(rx), int(ry), int(rz)), dtype='bool')
        for X_2d, Y_2d, z_val, k in self.grid.iter_slices():
            V[:, :, k] = self.evaluate_slice(X_2d, Y_2d, z_val)
        self._voxel_shape = V.shape
        self.voxel_data = np.packbits(V.ravel(order='F'))
        LOGGER.info(f"END render_volume")
        mem = MEMORY_USAGE(offset=mem0)
        LOGGER.info(f"DELTA MEMORY USED: {mem/2**30:0.2f} GB")
        LOGGER.info(40*"*")
        TIMING_END("render_volume")
        return self.voxel_data

    def render_uniform_grid(self, volume_scale=255):
        #REF https://docs.pyvista.org/examples/00-load/create-uniform-grid.html
        TIMING_START("render_uniform_grid")
        LOGGER.info(40*"*")
        LOGGER.info(f"{self.__class__} -> super().render_uniform_grid")
        mem0 = MEMORY_USAGE()
        LOGGER.info(f"TOTAL MEMORY USED: {mem0/2**30:0.2f} GB")
        LOGGER.info(40*"-")
        if self.voxel_data is None:
            self.render_volume()
        V = self._unpack_volume()
        rv  = self.grid.res_vector
        vsv = self.grid.voxel_size_vector
        ugrid = UniformGrid()
        # Set the grid dimensions: shape + 1 because we want to inject our values on
        #   the CELL data
        ugrid.dimensions = rv + 1
        # Edit the spatial reference
        ugrid.spacing = vsv  # These are the cell sizes along each axis
        # .view(uint8) avoids int64 upcast (8 GB at 1024^3); stays in uint8
        ugrid.cell_data['vol'] = V.flatten(order="F").view(np.uint8) * volume_scale
        LOGGER.info(f"END render_uniform_grid")
        mem = MEMORY_USAGE(offset=mem0)
        LOGGER.info(f"DELTA MEMORY USED: {mem/2**30:0.2f} GB")
        LOGGER.info(40*"*")
        TIMING_END("render_uniform_grid")
        return ugrid

    def render_volume_mesh(self, cache=True):
        #REF https://stackoverflow.com/questions/6030098/how-to-display-a-3d-plot-of-a-3d-array-isosurface-in-matplotlib-mplot3d-or-simil/35472146
        TIMING_START("render_volume_mesh")
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
        TIMING_END("render_volume_mesh")
        return pv_vol

    def render_surface_mesh(self,
                            cache=True,
                            use_meshfix = True,
                            smooth_iters = 0,
                            downscale_times = 0,
                            only_largest_component = False,
                            ):
        TIMING_START("render_surface_mesh")
        t0 = time.time()
        LOGGER.info(40*"*")
        LOGGER.info(f"{self.__class__} -> super().render_surface_mesh")
        mem0 = MEMORY_USAGE()
        LOGGER.info(f"TOTAL MEMORY USED: {mem0/2**30:0.2f} GB")
        LOGGER.info(40*"-")
        if cache and self.pv_surf is not None:
            return self.pv_surf
        import pyvista as pv
        pv_vol  = self.render_volume_mesh()
        pv_surf = pv_vol.extract_surface()
        if use_meshfix:
            _t0 = time.time()
            LOGGER.info(f"\trunning meshfix.repair()...")
            import pymeshfix as mf
            meshfix = mf.MeshFix(pv_surf)
            meshfix.repair(joincomp=True)
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
            pv_surf = pv_surf.triangulate()
            from voxelcad.utils.pyvista_tools import downscale_trimesh
            pv_surf = downscale_trimesh(pv_surf,smooth_iters=smooth_iters,repeat=downscale_times,decimation_factor=0.5)
            LOGGER.info(f"\t...completed in {time.time()-_t0:0.1f} s")
            if use_meshfix:
                _t0 = time.time()
                LOGGER.info(f"\trunning meshfix.repair() again after downscale...")
                import pymeshfix as mf
                meshfix = mf.MeshFix(pv_surf)
                meshfix.repair(joincomp=True)
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
        TIMING_END("render_surface_mesh")
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
        """Test if points defined by arrays X, Y, Z are in the volume."""
        LOGGER.info(40*"*")
        LOGGER.info(f"{self.__class__} -> super().test_points")
        mem0 = MEMORY_USAGE()
        LOGGER.info(f"TOTAL MEMORY USED: {mem0/2**30:0.2f} GB")
        LOGGER.info(40*"-")
        I,J,K = self.index_transform(X,Y,Z)
        if self.voxel_data is None:
            self.render_volume()
        V = self._unpack_volume()
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
        def transform_coord_to_index(C,rc,c0,c1):
            i_test = np.floor(rc*(C-c0)/(c1-c0)).astype('int')
            I = np.where((0 <= i_test) & (i_test < rc),i_test,-1)
            LOGGER.debug(f"test_points: I.min()={I.min()},  I.max()={I.max()}")
            return I
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
        v = np.asarray(v, dtype='float64')
        M4 = np.eye(4)
        M4[:3, 3] = v
        M4inv = np.eye(4)
        M4inv[:3, 3] = -v
        new_grid = VoxelGrid(xlim=self.grid.xlim + v[0],
                             ylim=self.grid.ylim + v[1],
                             zlim=self.grid.zlim + v[2],
                             voxel_size=self.grid.voxel_size_vector)
        return TransformedModel(self, M4, M4inv, new_grid)

    def rotate_x(self, degrees, **kwargs):
        return self.rotate([1.0,0,0], degrees, **kwargs)

    def rotate_y(self, degrees, **kwargs):
        return self.rotate([0,1.0,0], degrees, **kwargs)

    def rotate_z(self, degrees, **kwargs):
        return self.rotate([0,0,1.0], degrees, **kwargs)

    def rotate(self, v, degrees, return_matrices=False):
        #REF: https://en.wikipedia.org/wiki/Rotation_matrix
        #    "Rotation matrix from axis and angle"
        v = np.array(v)
        u = v/np.linalg.norm(v)
        ux,uy,uz = u
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
        Rinv = np.array(((c  + Uxx, Uxy+uz*s, Uxz-uy*s),
                         (Uxy-uz*s, c  + Uyy, Uyz+ux*s),
                         (Uxz+uy*s, Uyz-ux*s, c  + Uzz)))
        if return_matrices:
            return (R,Rinv)
        else:
            m = self.apply_transformation(R,Rinv)
            LOGGER.debug(f"rotate: type={type(m).__name__}, grid={m.grid!r}")
            return m

    def scale(self, v):
        v = v*np.ones(3,dtype="float32")
        S    = np.array(((v[0],0,0),(0,v[1],0),(0,0,v[2])))
        Sinv = np.array(((1.0/v[0],0,0),(0,1.0/v[1],0),(0,0,1.0/v[2])))
        m = self.apply_transformation(S,Sinv)
        LOGGER.debug(f"scale: type={type(m).__name__}, grid={m.grid!r}")
        return m

    def apply_transformation(self, M, Minv):
        """Return a lazy TransformedModel (deferred materialization).

        Args:
            M: 3x3 forward transform matrix
            Minv: 3x3 inverse transform matrix
        """
        # Embed 3x3 into 4x4 homogeneous
        M4 = np.eye(4)
        M4[:3, :3] = M
        M4inv = np.eye(4)
        M4inv[:3, :3] = Minv
        C = self.grid.compute_box_corner_vectors()
        Ct = np.dot(C, M.T)
        xlim = (Ct[:,0].min(), Ct[:,0].max())
        ylim = (Ct[:,1].min(), Ct[:,1].max())
        zlim = (Ct[:,2].min(), Ct[:,2].max())
        new_grid = VoxelGrid(xlim, ylim, zlim,
                             voxel_size=self.grid.voxel_size_vector)
        return TransformedModel(self, M4, M4inv, new_grid)

    def render_on_grid(self, target_grid):
        """Evaluate this model's geometry on an arbitrary target grid.

        Returns a new VoxelModel with packed data on target_grid.
        Uses evaluate_slice() for primitives/CSG, or _get_slice_for_coords()
        for coordinate remapping of packed data models.
        """
        if self.voxel_data is not None and self.grid.same_grid(target_grid):
            return VoxelModel(grid=target_grid, voxel_data=self.voxel_data,
                              _voxel_shape=self._voxel_shape)
        rx, ry, rz = target_grid.res_vector
        V = np.zeros((int(rx), int(ry), int(rz)), dtype='bool')
        for X_2d, Y_2d, z_val, k in target_grid.iter_slices():
            V[:, :, k] = self._get_slice_for_coords(X_2d, Y_2d, z_val)
        return VoxelModel(grid=target_grid, voxel_data=V)

    def _ensure_rendered(self):
        """Ensure voxel_data is populated (render if needed)."""
        if self.voxel_data is None:
            self.render_volume()

    def _same_grid_op(self, other, np_op):
        """Tier 1: byte-level boolean op for same-grid operands."""
        self._ensure_rendered()
        other._ensure_rendered()
        packed = np_op(self.voxel_data, other.voxel_data)
        return VoxelModel(grid=self.grid, voxel_data=packed,
                          _voxel_shape=self._voxel_shape)

    def _compatible_grid_op(self, other, np_op):
        """Tier 2: render both on union grid, then byte-level op."""
        union_grid = self.grid | other.grid
        left = self.render_on_grid(union_grid)
        right = other.render_on_grid(union_grid)
        packed = np_op(left.voxel_data, right.voxel_data)
        return VoxelModel(grid=union_grid, voxel_data=packed,
                          _voxel_shape=left._voxel_shape)

    def __or__(self, other): #union
        # Tier 1: same grid with existing data → byte-level
        if (self.voxel_data is not None and other.voxel_data is not None
                and self.grid.same_grid(other.grid)):
            return self._same_grid_op(other, np.bitwise_or)
        # Tier 2: compatible grids → render on union grid + byte-level
        if self.grid.compatible_grid(other.grid):
            return self._compatible_grid_op(other, np.bitwise_or)
        # Tier 3: incompatible → CSGModel lazy evaluation
        bounding_grid = self.grid | other.grid
        return CSGModel(self, 'or', other, bounding_grid)

    def __and__(self, other): #intersection
        # Tier 1: same grid with existing data → byte-level
        if (self.voxel_data is not None and other.voxel_data is not None
                and self.grid.same_grid(other.grid)):
            return self._same_grid_op(other, np.bitwise_and)
        # Tier 2: compatible grids → render on union grid + byte-level
        if self.grid.compatible_grid(other.grid):
            return self._compatible_grid_op(other, np.bitwise_and)
        # Tier 3: incompatible → CSGModel lazy evaluation
        bounding_grid = self.grid & other.grid
        return CSGModel(self, 'and', other, bounding_grid)

    def __xor__(self, other): #exclusive or
        # Tier 1: same grid with existing data → byte-level
        if (self.voxel_data is not None and other.voxel_data is not None
                and self.grid.same_grid(other.grid)):
            return self._same_grid_op(other, np.bitwise_xor)
        # Tier 2: compatible grids → render on union grid + byte-level
        if self.grid.compatible_grid(other.grid):
            return self._compatible_grid_op(other, np.bitwise_xor)
        # Tier 3: incompatible → CSGModel lazy evaluation
        bounding_grid = self.grid | other.grid
        return CSGModel(self, 'xor', other, bounding_grid)

    def __sub__(self, other): #difference
        _sub_op = lambda a, b: np.bitwise_and(a, np.bitwise_not(b))
        # Tier 1: same grid with existing data → byte-level
        if (self.voxel_data is not None and other.voxel_data is not None
                and self.grid.same_grid(other.grid)):
            packed = _sub_op(self.voxel_data, other.voxel_data)
            return VoxelModel(grid=self.grid, voxel_data=packed,
                              _voxel_shape=self._voxel_shape)
        # Tier 2: compatible grids → render on union grid + byte-level
        if self.grid.compatible_grid(other.grid):
            return self._compatible_grid_op(other, _sub_op)
        # Tier 3: incompatible → CSGModel lazy evaluation
        return CSGModel(self, 'sub', other, self.grid)

    def __invert__(self): #bitwise NOT
        self._ensure_rendered()
        packed = np.bitwise_not(self.voxel_data)
        return VoxelModel(grid=self.grid, voxel_data=packed,
                          _voxel_shape=self._voxel_shape)

class CSGModel(VoxelModel):
    """Lazy boolean combination — defers materialization until consumption.

    Stores operand references and operation type without allocating
    intermediate volumes. evaluate_slice() evaluates both children
    per-slice and combines on the fly.
    """
    _OP_MAP = {
        'or':  lambda L, R: L | R,
        'and': lambda L, R: L & R,
        'xor': lambda L, R: L ^ R,
        'sub': lambda L, R: L & ~R,
    }

    def __init__(self, left, op, right, grid):
        super().__init__(grid=grid)
        self.left = left
        self.op = op
        self.right = right

    def evaluate_slice(self, X_2d, Y_2d, z_val):
        L = self.left._get_slice_for_coords(X_2d, Y_2d, z_val)
        R = self.right._get_slice_for_coords(X_2d, Y_2d, z_val)
        return self._OP_MAP[self.op](L, R)

    def _collect_leaves(self, leaves: List[LeafNode], ops: List[Tuple[str, int, int]]) -> int:
        """Recursively collect leaves and operations from this CSG tree.

        Args:
            leaves: List to append leaf nodes to
            ops: List to append operations to (postfix order)

        Returns:
            Index of this node's result in the evaluation stack
        """
        # Process left subtree
        if isinstance(self.left, CSGModel):
            left_idx = self.left._collect_leaves(leaves, ops)
        else:
            left_idx = len(leaves)
            leaves.append(self.left._classify_leaf())

        # Process right subtree
        if isinstance(self.right, CSGModel):
            right_idx = self.right._collect_leaves(leaves, ops)
        else:
            right_idx = len(leaves)
            leaves.append(self.right._classify_leaf())

        # Record this operation (postfix: left, right, then op)
        result_idx = len(leaves) + len(ops)
        ops.append((self.op, left_idx, right_idx))
        return result_idx

    def _plan_execution(self) -> ExecutionPlan:
        """Analyze CSG tree and create an optimized execution plan.

        Walks the tree to:
        1. Collect all leaf nodes with their classifications
        2. Record operations in postfix order
        3. Determine if all leaves share compatible grids
        4. Choose optimal execution strategy

        Returns:
            ExecutionPlan with leaves, operations, and strategy
        """
        plan = ExecutionPlan()

        # Collect tree structure
        self._collect_leaves(plan.leaves, plan.operations)

        if not plan.leaves:
            return plan

        # Check grid compatibility across all leaves
        first_vs = plan.leaves[0].voxel_size
        plan.all_compatible = all(
            np.allclose(leaf.voxel_size, first_vs)
            for leaf in plan.leaves
        )

        # Check if all leaves can use fused kernels
        plan.all_fused_capable = all(
            leaf.leaf_type in (LeafType.FUSED, LeafType.MATERIALIZED)
            for leaf in plan.leaves
        )

        # Compute common grid if all compatible
        if plan.all_compatible:
            # Union of all leaf grids
            union_grid = plan.leaves[0].grid
            for leaf in plan.leaves[1:]:
                union_grid = union_grid | leaf.grid
            plan.common_grid = union_grid

        # Determine execution strategy
        if plan.all_compatible and plan.all_fused_capable:
            plan.strategy = "fused_bytewise"
        elif plan.all_compatible:
            plan.strategy = "mixed"
        else:
            plan.strategy = "streaming"

        return plan

    def _classify_leaf(self) -> LeafNode:
        """CSGModel is not a leaf — it's an internal node.

        This should not be called on CSGModel directly, but if it is
        (e.g., nested CSG that wasn't flattened), treat as fallback.
        """
        return LeafNode(self, LeafType.FALLBACK, self.grid)

    # Byte-level operation map for packed arrays
    _BYTEWISE_OP_MAP = {
        'or':  np.bitwise_or,
        'and': np.bitwise_and,
        'xor': np.bitwise_xor,
        'sub': lambda a, b: np.bitwise_and(a, np.bitwise_not(b)),
    }

    def render_volume(self):
        """Render CSG tree using query-planned execution.

        Analyzes the tree structure and chooses optimal strategy:
        - fused_bytewise/mixed: render leaves to common grid, combine with byte-level ops
        - streaming: per-slice evaluation (fallback)
        """
        if self.voxel_data is not None:
            return self.voxel_data

        plan = self._plan_execution()

        # Use optimized path when all leaves share compatible grids
        if plan.strategy in ("fused_bytewise", "mixed") and plan.common_grid is not None:
            return self._render_planned(plan)

        # Fallback: streaming per-slice evaluation
        return self._render_streaming()

    def _render_planned(self, plan: ExecutionPlan):
        """Execute CSG tree using query plan with byte-level combination.

        Renders all leaves onto the common grid, then applies operations
        in postfix order using byte-level bitwise ops on packed arrays.
        """
        TIMING_START("render_volume_planned")
        common_grid = plan.common_grid

        # Render all leaves onto the common grid
        rendered = []
        for leaf in plan.leaves:
            vm = leaf.model.render_on_grid(common_grid)
            rendered.append(vm.voxel_data)

        # Apply operations in postfix order
        # Stack holds intermediate results (packed arrays)
        stack = list(rendered)  # Start with leaf results

        for op, left_idx, right_idx in plan.operations:
            left_data = stack[left_idx]
            right_data = stack[right_idx]
            result = self._BYTEWISE_OP_MAP[op](left_data, right_data)
            stack.append(result)

        # Final result is the last item on the stack
        self.voxel_data = stack[-1]
        self._voxel_shape = (int(common_grid.res_vector[0]),
                             int(common_grid.res_vector[1]),
                             int(common_grid.res_vector[2]))
        self.grid = common_grid
        TIMING_END("render_volume_planned")
        return self.voxel_data

    def _render_streaming(self):
        """Fallback: render using per-slice streaming evaluation."""
        TIMING_START("render_volume_streaming")
        if self.grid is None:
            raise ValueError("CSGModel requires a grid")
        rx, ry, rz = self.grid.res_vector
        V = np.zeros((int(rx), int(ry), int(rz)), dtype='bool')
        for X_2d, Y_2d, z_val, k in self.grid.iter_slices():
            V[:, :, k] = self.evaluate_slice(X_2d, Y_2d, z_val)
        self._voxel_shape = V.shape
        self.voxel_data = np.packbits(V.ravel(order='F'))
        TIMING_END("render_volume_streaming")
        return self.voxel_data


class TransformedModel(VoxelModel):
    """Lazy affine transform — defers materialization until consumption.

    Uses 4x4 homogeneous matrices for full affine transforms (rotation,
    scale, translation, and arbitrary compositions). Chained transforms
    compose into a single matrix pair.
    """

    def __init__(self, source, M4, M4inv, grid):
        super().__init__(grid=grid)
        self.source = source
        self.M4 = M4        # 4x4 forward transform
        self.M4inv = M4inv  # 4x4 inverse transform

    def evaluate_slice(self, X_2d, Y_2d, z_val):
        """Evaluate by inverse-transforming coordinates into source space.

        If source has evaluate_at_coords(), evaluates geometry directly
        at inverse-transformed coordinates (no intermediate volume).
        Otherwise falls back to rendering source and indexing.
        """
        Minv = self.M4inv
        Z_2d = np.full_like(X_2d, z_val)

        # Inverse transform: target coords → source coords
        X_orig = Minv[0,0]*X_2d + Minv[0,1]*Y_2d + Minv[0,2]*Z_2d + Minv[0,3]
        Y_orig = Minv[1,0]*X_2d + Minv[1,1]*Y_2d + Minv[1,2]*Z_2d + Minv[1,3]
        Z_orig = Minv[2,0]*X_2d + Minv[2,1]*Y_2d + Minv[2,2]*Z_2d + Minv[2,3]

        # If source can evaluate at arbitrary coords, use it directly
        if self.source._has_evaluate_at_coords():
            return self.source.evaluate_at_coords(X_orig, Y_orig, Z_orig)

        # Fallback: render source and index into packed data
        if self.source.voxel_data is None:
            self.source.render_volume()
        V_src = self.source._unpack_volume()
        src_rx, src_ry, src_rz = self.source.grid.res_vector
        sx0, sx1 = self.source.grid.xlim
        sy0, sy1 = self.source.grid.ylim
        sz0, sz1 = self.source.grid.zlim
        I = np.floor(src_rx * (X_orig - sx0) / (sx1 - sx0)).astype('int')
        J = np.floor(src_ry * (Y_orig - sy0) / (sy1 - sy0)).astype('int')
        K = np.floor(src_rz * (Z_orig - sz0) / (sz1 - sz0)).astype('int')
        valid = (I >= 0) & (I < src_rx) & (J >= 0) & (J < src_ry) & (K >= 0) & (K < src_rz)
        I_safe = np.where(valid, I, 0)
        J_safe = np.where(valid, J, 0)
        K_safe = np.where(valid, K, 0)
        return np.where(valid, V_src[I_safe, J_safe, K_safe], False)

    def _transform_corners(self, source, M4):
        """Compute new bounding grid from transformed source corners."""
        C = source.grid.compute_box_corner_vectors()
        # Apply 4x4 to homogeneous coordinates
        Ch = np.hstack([C, np.ones((C.shape[0], 1))])
        Ct = (M4 @ Ch.T).T[:, :3]
        xlim = (Ct[:,0].min(), Ct[:,0].max())
        ylim = (Ct[:,1].min(), Ct[:,1].max())
        zlim = (Ct[:,2].min(), Ct[:,2].max())
        return VoxelGrid(xlim, ylim, zlim,
                         voxel_size=source.grid.voxel_size_vector)

    def apply_transformation(self, M, Minv):
        """Compose: convert 3x3 to 4x4, then compose with existing."""
        M4_new = np.eye(4)
        M4_new[:3, :3] = M
        M4inv_new = np.eye(4)
        M4inv_new[:3, :3] = Minv
        composed_M4 = M4_new @ self.M4
        composed_M4inv = self.M4inv @ M4inv_new
        new_grid = self._transform_corners(self.source, composed_M4)
        return TransformedModel(self.source, composed_M4, composed_M4inv, new_grid)

    def translate(self, v):
        """Compose translation into the transform chain."""
        T = np.eye(4)
        T[:3, 3] = v
        Tinv = np.eye(4)
        Tinv[:3, 3] = -np.array(v)
        composed_M4 = T @ self.M4
        composed_M4inv = self.M4inv @ Tinv
        new_grid = self._transform_corners(self.source, composed_M4)
        return TransformedModel(self.source, composed_M4, composed_M4inv, new_grid)

    def _classify_leaf(self) -> LeafNode:
        """Classify TransformedModel based on source's evaluation capability.

        If source has evaluate_at_coords(), TransformedModel can evaluate
        directly without intermediate volume → promotes to FUSED.
        Otherwise remains FALLBACK (requires source render + index).
        """
        if self.source._has_evaluate_at_coords():
            return LeafNode(self, LeafType.FUSED, self.grid)
        return LeafNode(self, LeafType.FALLBACK, self.grid)


def union_all(models):
    u = models[0]
    print(f"union_all #{0}: u.grid.sv: {u.grid.compute_size_vector()}")
    for i,m in enumerate(models[1:]):
        u |= m
        print(f"union_all #{i+1}: u.grid.sv: {u.grid.compute_size_vector()}")
    return u
