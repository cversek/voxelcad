import os, time
from dataclasses import dataclass, field
from typing import OrderedDict, List, Dict, Any, Optional, Tuple
import numpy as np
import pyvista as pv

import voxelcad.environment as ENV

from voxelcad.debug import create_logger, currentframe, DEBUG_TAG, DEBUG_EMBED, MEMORY_USAGE, TIMING_START, TIMING_END
LOGGER = create_logger(__name__)

from voxelcad.voxel_grid import VoxelGrid, UniformGrid


@dataclass
class LeafNode:
    """A leaf in the CSG tree with its grid info."""
    model: Any  # VoxelModel subclass
    grid: 'VoxelGrid'


@dataclass
class ExecutionPlan:
    """Query plan for CSG tree execution.

    All leaves render on a common grid via render_on_grid(),
    then results are combined with byte-level bitwise ops.
    """
    leaves: List[LeafNode] = field(default_factory=list)
    operations: List[Tuple[str, int, int]] = field(default_factory=list)
    common_grid: Optional['VoxelGrid'] = None

    def __repr__(self):
        return (f"ExecutionPlan(leaves={len(self.leaves)}, "
                f"common_grid={'set' if self.common_grid else 'None'})")


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
            self.voxel_data = np.packbits(voxel_data.ravel(order='F'), bitorder='big')
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
        return np.unpackbits(self.voxel_data, bitorder='big')[:n].reshape(self._voxel_shape, order='F').view(np.bool_)

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
        bits = np.unpackbits(self.voxel_data[byte_start:byte_end], bitorder='big')
        bit_offset = start - byte_start * 8
        return bits[bit_offset:bit_offset + slice_size].reshape(rx, ry, order='F').view(np.bool_)

    def render_on_grid(self, grid, M4inv=None):
        """THE unified interface for geometry evaluation.

        Returns packed uint8 array (F-order) representing the model
        evaluated on the given grid coordinates.

        Base class default: nearest-neighbor resampling of existing packed data.
        Primitives override _render_cython/_render_numpy with geometry evaluation.

        Args:
            grid: Target VoxelGrid to evaluate on
            M4inv: Optional 4x4 inverse transform matrix
        """
        # Same-grid shortcut: no work needed
        if M4inv is None and self.voxel_data is not None and self.grid.same_grid(grid):
            return self.voxel_data
        # Dispatch to Cython or NumPy implementation
        if ENV.use_cython:
            return self._render_cython(grid, M4inv)
        return self._render_numpy(grid, M4inv)

    def _render_cython(self, grid, M4inv=None):
        """Cython implementation. Base class: nearest-neighbor resampling.

        Primitives override with geometry-specific Cython kernels.
        """
        from voxelcad._kernels import resample_and_pack
        if resample_and_pack is None:
            import warnings
            if ENV.use_cython:
                warnings.warn(
                    "ENV.use_cython=True but Cython resample_and_pack kernel is not "
                    "available. Falling back to NumPy resampling. Build Cython "
                    "extensions with: python setup.py build_ext --inplace",
                    RuntimeWarning, stacklevel=3)
            return self._render_numpy(grid, M4inv)
        src_data = self.render_volume()
        src_rx, src_ry, src_rz = [int(r) for r in self.grid.res_vector]
        src_xcc, src_ycc, src_zcc = self.grid.compute_cell_center_ranges()
        dst_xcc, dst_ycc, dst_zcc = grid.compute_cell_center_ranges()
        return resample_and_pack(
            src_xcc, src_ycc, src_zcc, src_data,
            src_rx, src_ry, src_rz,
            dst_xcc, dst_ycc, dst_zcc, M4inv)

    def _render_numpy(self, grid, M4inv=None):
        """NumPy implementation. Base class: nearest-neighbor resampling.

        Primitives override with geometry-specific NumPy evaluation.
        """
        src_data = self.render_volume()  # ensure source is materialized
        src_rx, src_ry, src_rz = [int(r) for r in self.grid.res_vector]
        # Use cell centers + spacing (matches Cython resample_and_pack)
        src_xcc, src_ycc, src_zcc = self.grid.compute_cell_center_ranges()
        src_x0 = float(src_xcc[0])
        src_y0 = float(src_ycc[0])
        src_z0 = float(src_zcc[0])
        src_dx = float(src_xcc[1] - src_xcc[0]) if src_rx > 1 else 1.0
        src_dy = float(src_ycc[1] - src_ycc[0]) if src_ry > 1 else 1.0
        src_dz = float(src_zcc[1] - src_zcc[0]) if src_rz > 1 else 1.0

        dst_rx, dst_ry, dst_rz = [int(r) for r in grid.res_vector]
        V = np.zeros((dst_rx, dst_ry, dst_rz), dtype='bool')

        for X_2d, Y_2d, z_val, k in grid.iter_slices():
            if M4inv is not None:
                Z_2d = np.full_like(X_2d, z_val)
                Xp = M4inv[0,0]*X_2d + M4inv[0,1]*Y_2d + M4inv[0,2]*Z_2d + M4inv[0,3]
                Yp = M4inv[1,0]*X_2d + M4inv[1,1]*Y_2d + M4inv[1,2]*Z_2d + M4inv[1,3]
                Zp = M4inv[2,0]*X_2d + M4inv[2,1]*Y_2d + M4inv[2,2]*Z_2d + M4inv[2,3]
            else:
                Xp, Yp = X_2d, Y_2d
                Zp = np.full_like(X_2d, z_val)

            # Nearest-neighbor index into source grid (cell-center-based,
            # matches Cython resample_and_pack: (long long)((x-x0)/dx + 0.5))
            I = np.floor((Xp - src_x0) / src_dx + 0.5).astype('int')
            J = np.floor((Yp - src_y0) / src_dy + 0.5).astype('int')
            K_idx = np.floor((Zp - src_z0) / src_dz + 0.5).astype('int')
            valid = ((I >= 0) & (I < src_rx) &
                     (J >= 0) & (J < src_ry) &
                     (K_idx >= 0) & (K_idx < src_rz))
            I_safe = np.where(valid, I, 0)
            J_safe = np.where(valid, J, 0)
            K_safe = np.where(valid, K_idx, 0)

            # Bit lookup in F-order packed array (MSB-first, matching np.packbits)
            lin_idx = I_safe + J_safe * src_rx + K_safe * src_rx * src_ry
            byte_idx = lin_idx >> 3
            bit_idx = 7 - (lin_idx & 7)
            bits = (src_data[byte_idx] >> bit_idx) & 1
            V[:, :, k] = np.where(valid, bits.astype(bool), False)

        return np.packbits(V.ravel(order='F'), bitorder='big')

    def render_volume(self):
        """Render on own grid and cache. Returns packed uint8."""
        if self.voxel_data is not None:
            return self.voxel_data
        if self.grid is None:
            self.construct_grid()
        TIMING_START("render_volume")
        LOGGER.info(f"{self.__class__.__name__} -> render_volume")
        self.voxel_data = self.render_on_grid(self.grid)
        rx, ry, rz = [int(r) for r in self.grid.res_vector]
        self._voxel_shape = (rx, ry, rz)
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
        pv_surf = pv_vol.extract_surface(algorithm='dataset_surface')
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
            pv_surf = pv_surf.smooth(n_iter=smooth_iters,progress_bar=ENV.progress_bar)
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

    def render_surface_mesh_edt(self, isovalue=0.0, lowpass_cutoff=0.25,
                                cache=True, only_largest_component=False,
                                target_reduction=0.0):
        """Extract smooth surface mesh using a signed distance field.

        Computes EDT on both interior and exterior of the binary volume
        to build a signed distance field (SDF = interior - exterior),
        applies a Butterworth low-pass filter in the frequency domain
        to remove voxel staircase artifacts, then runs marching cubes.

        Args:
            isovalue: SDF threshold for isosurface extraction.
                0.0 = exact boundary (default), positive = erode inward.
            lowpass_cutoff: Cutoff frequency as fraction of Nyquist
                (0.0-0.5 cycles/voxel).  Removes staircase artifacts
                above this frequency.  0 disables filtering.
            cache: If True, cache the result in self.pv_surf.
            only_largest_component: If True, keep only the largest
                connected component.
            target_reduction: Fraction of triangles to remove (0.0-0.95).
                0.0 = no decimation (default), 0.9 = reduce to 10%.
                Uses quadric error metric decimation (single pass).

        Returns:
            PyVista PolyData surface mesh.
        """
        TIMING_START("render_surface_mesh_edt")
        t0 = time.time()
        LOGGER.info(f"{self.__class__.__name__} -> render_surface_mesh_edt")
        mem0 = MEMORY_USAGE()
        if cache and self.pv_surf is not None:
            if target_reduction > 0 and self.pv_surf.n_cells > 0:
                _t0 = time.time()
                n_before = self.pv_surf.n_cells
                LOGGER.info(f"\tdecimating cached mesh ({n_before} tris, "
                            f"target_reduction={target_reduction})...")
                result = self.pv_surf.decimate(target_reduction)
                LOGGER.info(f"\t...decimated to {result.n_cells} tris "
                            f"in {time.time()-_t0:.2f} s")
                return result
            return self.pv_surf
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            import warnings
            warnings.warn(
                "scipy not installed — falling back to render_surface_mesh(). "
                "Install scipy for smooth EDT-based mesh extraction: "
                "pip install scipy",
                RuntimeWarning, stacklevel=2)
            TIMING_END("render_surface_mesh_edt")
            return self.render_surface_mesh(
                cache=cache, only_largest_component=only_largest_component)
        # Ensure volume is rendered
        if self.voxel_data is None:
            self.render_volume()
        # Unpack to bool and compute signed distance field
        V = self._unpack_volume()
        _t0 = time.time()
        LOGGER.info(f"\tcomputing SDF on {V.shape} volume...")
        dist = distance_transform_edt(V) - distance_transform_edt(~V)
        del V  # free bool volume
        dist = dist.astype(np.float32)
        LOGGER.info(f"\t...SDF completed in {time.time()-_t0:.1f} s")
        # Butterworth low-pass filter in frequency domain
        if lowpass_cutoff > 0:
            _t0 = time.time()
            LOGGER.info(f"\tFFT low-pass filter (cutoff={lowpass_cutoff})...")
            from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq
            rx, ry, rz = dist.shape
            fx = fftfreq(rx)
            fy = fftfreq(ry)
            fz = rfftfreq(rz)  # half-spectrum for real FFT
            FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
            freq_mag = np.sqrt(FX**2 + FY**2 + FZ**2)
            # Butterworth order-4: sharper transition than ord-2, better
            # staircase suppression without Gibbs ringing artifacts
            H = (1.0 / (1.0 + (freq_mag / lowpass_cutoff)**8)).astype(np.float32)
            del FX, FY, FZ, freq_mag
            D_fft = rfftn(dist)
            del dist
            D_fft *= H
            del H
            dist = irfftn(D_fft, s=(rx, ry, rz))
            del D_fft
            LOGGER.info(f"\t...filtering completed in {time.time()-_t0:.1f} s")
        # Build PyVista uniform grid from SDF (point data for contour)
        rv = self.grid.res_vector
        vsv = self.grid.voxel_size_vector
        ugrid = UniformGrid()
        ugrid.dimensions = rv  # point grid matches SDF array shape
        ugrid.spacing = vsv
        ugrid.point_data['dist'] = dist.ravel(order='F').astype(np.float32)
        del dist
        # Marching cubes via contour on the signed distance field
        _t0 = time.time()
        LOGGER.info(f"\textracting isosurface at isovalue={isovalue}...")
        pv_surf = ugrid.contour([isovalue], scalars='dist')
        del ugrid  # free grid
        LOGGER.info(f"\t...contour completed in {time.time()-_t0:.1f} s")
        if only_largest_component and pv_surf.n_points > 0:
            _t0 = time.time()
            LOGGER.info(f"\textracting largest component "
                        f"({pv_surf.n_cells} tris)...")
            pv_surf = pv_surf.extract_largest()
            LOGGER.info(f"\t...extracted ({pv_surf.n_cells} tris) "
                        f"in {time.time()-_t0:.2f} s")
        # Cache the full-resolution mesh before decimation
        if cache:
            self.pv_surf = pv_surf
        # Decimate after caching so cached mesh stays full-res
        if target_reduction > 0 and pv_surf.n_cells > 0:
            _t0 = time.time()
            n_before = pv_surf.n_cells
            LOGGER.info(f"\tdecimating mesh ({n_before} tris, "
                        f"target_reduction={target_reduction})...")
            pv_surf = pv_surf.decimate(target_reduction)
            LOGGER.info(f"\t...decimated to {pv_surf.n_cells} tris "
                        f"in {time.time()-_t0:.2f} s")
        t1 = time.time()
        LOGGER.info(f"END render_surface_mesh_edt, time: {t1-t0:.1f} s")
        mem = MEMORY_USAGE(offset=mem0)
        LOGGER.info(f"DELTA MEMORY USED: {mem/2**30:.2f} GB")
        TIMING_END("render_surface_mesh_edt")
        return pv_surf

    def analyze_spectrum(self, lowpass_cutoff=0.25, plot=True,
                         save_path=None, n_bins=None):
        """Analyze the SDF's frequency content with optional plots.

        Computes the signed distance field, then analyzes its radial
        power spectrum to estimate bandwidth and safe stride values.

        Parameters
        ----------
        lowpass_cutoff : float
            Butterworth cutoff for comparison. Default 0.25.
        plot : bool
            If True, generate diagnostic PNG plots.
        save_path : str, optional
            Directory to save plots. Defaults to current directory.
        n_bins : int, optional
            Radial frequency bins.

        Returns
        -------
        dict
            Keys: ``freq_bins``, ``power``, ``bandwidth_99``,
            ``bandwidth_95``, ``recommended_cutoff``, ``safe_stride``,
            ``sdf_raw`` (the raw SDF array for further analysis).
        """
        from scipy.ndimage import distance_transform_edt as edt
        from .utils.spectral import (
            radial_power_spectrum, estimate_bandwidth,
            safe_stride, recommend_cutoff,
            plot_radial_spectrum, plot_filter_effect,
            plot_spectrum_slices, plot_cumulative_energy,
        )

        if self.voxel_data is None:
            self.render_volume()

        V = self._unpack_volume()
        LOGGER.info(f"analyze_spectrum: computing SDF on {V.shape} volume...")
        dist = edt(V) - edt(~V)
        del V
        dist = dist.astype(np.float32)

        freq_bins, power = radial_power_spectrum(dist, n_bins=n_bins)
        bw99 = estimate_bandwidth(freq_bins, power, 0.99)
        bw95 = estimate_bandwidth(freq_bins, power, 0.95)
        rec_cutoff = recommend_cutoff(freq_bins, power, method='energy')
        stride = safe_stride(bw99)

        LOGGER.info(f"  bandwidth_99 = {bw99:.4f} cyc/vox")
        LOGGER.info(f"  bandwidth_95 = {bw95:.4f} cyc/vox")
        LOGGER.info(f"  recommended_cutoff = {rec_cutoff:.4f}")
        LOGGER.info(f"  safe_stride = {stride}")

        result = dict(
            freq_bins=freq_bins, power=power,
            bandwidth_99=bw99, bandwidth_95=bw95,
            recommended_cutoff=rec_cutoff, safe_stride=stride,
            sdf_raw=dist,
        )

        if plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import os

            save_dir = save_path or '.'
            name = self.__class__.__name__
            res = self.grid.res_vector[0]

            # 1. Radial power spectrum
            fig = plot_radial_spectrum(
                freq_bins, power, cutoff=lowpass_cutoff,
                bandwidth=bw99,
                title=f'{name} {res}^3 — Radial Power Spectrum')
            path = os.path.join(save_dir,
                                f'spectrum_{name}_{res}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            LOGGER.info(f"  saved: {path}")

            # 2. Cumulative energy
            markers = {
                f'cutoff ({lowpass_cutoff})': lowpass_cutoff,
                f'bw99 ({bw99:.3f})': bw99,
            }
            fig = plot_cumulative_energy(
                freq_bins, power, markers=markers,
                title=f'{name} {res}^3 — Cumulative Energy')
            path = os.path.join(save_dir,
                                f'cumulative_{name}_{res}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            LOGGER.info(f"  saved: {path}")

            # 3. 2D spectrum slices
            fig = plot_spectrum_slices(
                dist, title_prefix=f'{name} {res}^3')
            path = os.path.join(save_dir,
                                f'slices_{name}_{res}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            LOGGER.info(f"  saved: {path}")

            # 4. Filter effect (before/after Butterworth)
            from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq
            rx, ry, rz = dist.shape
            fx = fftfreq(rx)
            fy = fftfreq(ry)
            fz = rfftfreq(rz)
            FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
            freq_mag = np.sqrt(FX**2 + FY**2 + FZ**2)
            H = (1.0 / (1.0 + (freq_mag / lowpass_cutoff)**8)
                 ).astype(np.float32)
            del FX, FY, FZ, freq_mag
            D_fft = rfftn(dist)
            D_fft *= H
            del H
            dist_filt = irfftn(D_fft, s=(rx, ry, rz))
            del D_fft

            fig = plot_filter_effect(
                dist, dist_filt, cutoff=lowpass_cutoff,
                n_bins=n_bins,
                title=f'{name} {res}^3 — Filter Effect')
            path = os.path.join(save_dir,
                                f'filter_{name}_{res}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            LOGGER.info(f"  saved: {path}")
            del dist_filt

        return result

    def plot(self, *args, mode="volume", **kwargs):
        """Plot the model.

        Args:
            mode: "volume" for voxel volume mesh, "surf" for EDT surface mesh.
            target_reduction: (surf mode only) Fraction of triangles to remove.
            *args, **kwargs: Passed to PyVista plot().
        """
        t0 = time.time()
        LOGGER.info(f"{self.__class__.__name__} -> plot(mode={mode!r})")
        kwargs['color'] = kwargs.get('color', 'white')
        if mode == "surf":
            target_reduction = kwargs.pop('target_reduction', 0.0)
            surf = self.render_surface_mesh_edt(
                target_reduction=target_reduction)
            LOGGER.info(f"\tplotting surface ({surf.n_cells} tris)...")
            _t0 = time.time()
            surf.plot(*args, **kwargs)
            LOGGER.info(f"\t...plot completed in {time.time()-_t0:.1f} s")
        else:
            vol_mesh = self.render_volume_mesh()
            LOGGER.info(f"\tplotting volume mesh...")
            _t0 = time.time()
            vol_mesh.plot(*args, **kwargs)
            LOGGER.info(f"\t...plot completed in {time.time()-_t0:.1f} s")
        LOGGER.info(f"END plot, total time: {time.time()-t0:.1f} s")

    def export(self, filename, **kwargs):
        t0 = time.time()
        LOGGER.info(f"{self.__class__.__name__} -> export({filename!r})")
        basepath, ext = os.path.splitext(filename)
        if ext == ".stl":
            if ENV.use_edt_export:
                # EDT pipeline: extract isovalue and only_largest_component,
                # pass remaining kwargs for future extensibility
                isovalue = kwargs.pop('isovalue', 0.0)
                lowpass_cutoff = kwargs.pop('lowpass_cutoff', 0.25)
                only_largest = kwargs.pop('only_largest_component', False)
                cache = kwargs.pop('cache', True)
                target_reduction = kwargs.pop('target_reduction', 0.0)
                surf_mesh = self.render_surface_mesh_edt(
                    isovalue=isovalue,
                    lowpass_cutoff=lowpass_cutoff,
                    cache=cache,
                    only_largest_component=only_largest,
                    target_reduction=target_reduction,
                )
            else:
                kwargs.setdefault('cache', False)
                surf_mesh = self.render_surface_mesh(**kwargs)
            _t0 = time.time()
            LOGGER.info(f"\tsaving STL ({surf_mesh.n_cells} tris)...")
            surf_mesh.save(filename)
            LOGGER.info(f"\t...save completed in {time.time()-_t0:.2f} s")
        else:
            raise ValueError(f"The filetype of extension '{ext}' is not recognized!")
        LOGGER.info(f"END export, total time: {time.time()-t0:.1f} s")

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

    def __or__(self, other):
        if (self.voxel_data is not None and other.voxel_data is not None
                and self.grid.same_grid(other.grid)):
            return self._same_grid_op(other, np.bitwise_or)
        return CSGModel(self, 'or', other, self.grid | other.grid)

    def __and__(self, other):
        if (self.voxel_data is not None and other.voxel_data is not None
                and self.grid.same_grid(other.grid)):
            return self._same_grid_op(other, np.bitwise_and)
        return CSGModel(self, 'and', other, self.grid & other.grid)

    def __xor__(self, other):
        if (self.voxel_data is not None and other.voxel_data is not None
                and self.grid.same_grid(other.grid)):
            return self._same_grid_op(other, np.bitwise_xor)
        return CSGModel(self, 'xor', other, self.grid | other.grid)

    def __sub__(self, other):
        _sub_op = lambda a, b: np.bitwise_and(a, np.bitwise_not(b))
        if (self.voxel_data is not None and other.voxel_data is not None
                and self.grid.same_grid(other.grid)):
            packed = _sub_op(self.voxel_data, other.voxel_data)
            return VoxelModel(grid=self.grid, voxel_data=packed,
                              _voxel_shape=self._voxel_shape)
        return CSGModel(self, 'sub', other, self.grid)

    def __invert__(self):
        self._ensure_rendered()
        packed = np.bitwise_not(self.voxel_data)
        return VoxelModel(grid=self.grid, voxel_data=packed,
                          _voxel_shape=self._voxel_shape)


class CSGModel(VoxelModel):
    """Lazy boolean combination — defers materialization until consumption.

    At render time, all leaves render on a common grid via render_on_grid(),
    then results are combined with byte-level bitwise ops.
    """

    _BYTEWISE_OP_MAP = {
        'or':  np.bitwise_or,
        'and': np.bitwise_and,
        'xor': np.bitwise_xor,
        'sub': lambda a, b: np.bitwise_and(a, np.bitwise_not(b)),
    }

    def __init__(self, left, op, right, grid):
        super().__init__(grid=grid)
        self.left = left
        self.op = op
        self.right = right

    def _collect_leaves(self, leaves, ops):
        """Recursively collect leaf models and operations (postfix order)."""
        if isinstance(self.left, CSGModel):
            left_idx = self.left._collect_leaves(leaves, ops)
        else:
            left_idx = len(leaves)
            leaves.append(LeafNode(self.left, self.left.grid))

        if isinstance(self.right, CSGModel):
            right_idx = self.right._collect_leaves(leaves, ops)
        else:
            right_idx = len(leaves)
            leaves.append(LeafNode(self.right, self.right.grid))

        result_idx = len(leaves) + len(ops)
        ops.append((self.op, left_idx, right_idx))
        return result_idx

    def _plan_execution(self):
        """Build execution plan: collect leaves, compute common grid."""
        plan = ExecutionPlan()
        self._collect_leaves(plan.leaves, plan.operations)
        if not plan.leaves:
            return plan
        # Common grid = union of all leaf grids
        union_grid = plan.leaves[0].grid
        for leaf in plan.leaves[1:]:
            union_grid = union_grid | leaf.grid
        plan.common_grid = union_grid
        return plan

    def render_volume(self):
        """Render CSG tree: all leaves on common grid, byte-level combination."""
        if self.voxel_data is not None:
            return self.voxel_data

        plan = self._plan_execution()
        common_grid = plan.common_grid or self.grid

        TIMING_START("render_volume_csg")

        # Render all leaves onto common grid
        rendered = []
        for leaf in plan.leaves:
            if isinstance(leaf.model, TransformedModel):
                packed = leaf.model.source.render_on_grid(common_grid, leaf.model.M4inv)
            else:
                packed = leaf.model.render_on_grid(common_grid)
            rendered.append(packed)

        # Postfix combination with byte-level ops
        stack = list(rendered)
        for op, left_idx, right_idx in plan.operations:
            result = self._BYTEWISE_OP_MAP[op](stack[left_idx], stack[right_idx])
            stack.append(result)

        self.voxel_data = stack[-1]
        self._voxel_shape = tuple(int(r) for r in common_grid.res_vector)
        self.grid = common_grid
        TIMING_END("render_volume_csg")
        return self.voxel_data


class TransformedModel(VoxelModel):
    """Lazy affine transform — thin wrapper passing M4inv to source.

    Uses 4x4 homogeneous matrices for full affine transforms.
    Chained transforms compose into a single matrix pair.
    """

    def __init__(self, source, M4, M4inv, grid):
        super().__init__(grid=grid)
        self.source = source
        self.M4 = M4
        self.M4inv = M4inv

    def render_volume(self):
        """Delegate to source with inverse transform."""
        if self.voxel_data is not None:
            return self.voxel_data
        self.voxel_data = self.source.render_on_grid(self.grid, self.M4inv)
        self._voxel_shape = tuple(int(r) for r in self.grid.res_vector)
        return self.voxel_data

    def _transform_corners(self, source, M4):
        """Compute new bounding grid from transformed source corners."""
        C = source.grid.compute_box_corner_vectors()
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


def union_all(models):
    u = models[0]
    print(f"union_all #{0}: u.grid.sv: {u.grid.compute_size_vector()}")
    for i,m in enumerate(models[1:]):
        u |= m
        print(f"union_all #{i+1}: u.grid.sv: {u.grid.compute_size_vector()}")
    return u
