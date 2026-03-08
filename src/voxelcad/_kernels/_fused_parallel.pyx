# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Fused evaluate-and-pack Cython kernels for VoxelCAD primitives.

Each kernel evaluates geometry, thresholds, and bit-packs in a single pass.
Eliminates the 1 GB bool intermediate that exists between evaluate_slice()
and np.packbits(). Uses OpenMP prange for Z-slice parallelism.

Output is bit-compatible with np.packbits(V.ravel(order='F')).
(working with Craig Wm. Versek <cversek@gmail.com>)
"""

import os
import numpy as np
cimport numpy as np
from libc.math cimport cos, sin, fabs, sqrt, pow as cpow
from libc.stdio cimport fopen, fwrite, fseek, fclose, FILE, SEEK_SET, fprintf, stderr
from libc.string cimport memset
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, threadid

# pthread declarations for pipeline overlap (OPT 10)
cdef extern from "pthread.h" nogil:
    ctypedef struct pthread_mutex_t:
        pass
    ctypedef struct pthread_cond_t:
        pass
    int pthread_mutex_init(pthread_mutex_t *, void *)
    int pthread_mutex_lock(pthread_mutex_t *)
    int pthread_mutex_unlock(pthread_mutex_t *)
    int pthread_cond_init(pthread_cond_t *, void *)
    int pthread_cond_wait(pthread_cond_t *, pthread_mutex_t *)
    int pthread_cond_signal(pthread_cond_t *)
    int pthread_mutex_destroy(pthread_mutex_t *)
    int pthread_cond_destroy(pthread_cond_t *)

cdef extern from "pthread.h" nogil:
    ctypedef unsigned long pthread_t
    int pthread_create(pthread_t *, void *, void *(*)(void *), void *)
    int pthread_join(pthread_t, void **)

# MC layer-processing arguments struct (OPT 10: enables pthread extraction)
ctypedef struct mc_args_t:
    signed char *slice_a
    signed char *slice_b
    signed char *band_a
    signed char *band_b
    int px
    int py
    int k
    signed char *ft
    int ft_ncols
    float *layer_ptrs[5]
    int layer_jstride[5]
    float mc_ox
    float mc_oy
    float mc_oz
    float mc_vsx
    float mc_vsy
    float mc_vsz
    float isovalue
    unsigned char *stl_buf
    int buf_count
    int tri_count
    FILE *fp
    int BUF_MAX
    # Sync primitives (OPT 10 pipeline)
    pthread_mutex_t mutex
    pthread_cond_t slice_ready
    pthread_cond_t mc_done
    int mc_go
    int mc_finished
    int terminate
    # OPT 17: Normal elision flag (0 = zero normals, 1 = compute normals)
    int compute_normals
    # OPT 16: Two-pass parallel MC
    int mc_n_threads
    pthread_mutex_t write_mutex
    # Phase 4b: Mesh output (pthread mesh MC)
    float *mesh_verts
    int *mesh_faces
    int mesh_n_verts
    int mesh_n_faces
    int mesh_verts_cap
    int mesh_faces_cap
    int *id_layers[5]
    int id_jstride[5]

np.import_array()


# ---------------------------------------------------------------------------
# Thread detection (delegates to super_utils when available)
# ---------------------------------------------------------------------------

cdef int _cached_p_cores = 0

def _detect_p_cores():
    """Detect number of performance cores."""
    global _cached_p_cores
    if _cached_p_cores > 0:
        return _cached_p_cores
    try:
        from super_utils import detect_performance_cores
        info = detect_performance_cores()
        _cached_p_cores = info["performance_cores"]
    except (ImportError, KeyError):
        _cached_p_cores = os.cpu_count() or 1
    return _cached_p_cores


def _get_optimal_threads(int work_items):
    """Determine optimal thread count for uniform-work kernels."""
    cdef int p_cores = _detect_p_cores()
    return max(1, min(p_cores, work_items))


def _get_fused_export_threads(int work_items):
    """Determine optimal thread count for fused_stl_export pipeline.

    The fused pipeline has non-uniform work (OPT 13 surface band skip
    creates load imbalance) plus a dedicated MC pthread. Empirically,
    reserving 1-2 P-cores for OS/runtime/MC gave better throughput:
    10/12 P-cores (83%) was optimal on M3 Max 12P+4E.
    """
    cdef int p_cores = _detect_p_cores()
    cdef int effective = p_cores
    if p_cores >= 8:
        effective = p_cores - 2
    elif p_cores >= 4:
        effective = p_cores - 1
    return max(1, min(effective, work_items))


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------

cdef inline void set_bit(unsigned char *packed, long long lin_idx) noexcept nogil:
    """Set a bit in packed array (MSB-first, matching np.packbits)."""
    cdef long long byte_idx = lin_idx >> 3
    cdef int bit_pos = 7 - <int>(lin_idx & 7)
    packed[byte_idx] = packed[byte_idx] | <unsigned char>(1 << bit_pos)


# ---------------------------------------------------------------------------
# Cube kernel
# ---------------------------------------------------------------------------

def evaluate_and_pack_cube(
    double[::1] xcc, double[::1] ycc, double[::1] zcc,
    double cx, double cy, double cz,
    double half_sx, double half_sy, double half_sz,
    double[:, ::1] M4inv=None,
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for Cube.

    Args:
        xcc, ycc, zcc: 1D cell-center coordinate arrays
        cx, cy, cz: cube center coordinates
        half_sx, half_sy, half_sz: half-widths per axis
        M4inv: optional 4x4 inverse transform matrix
        n_threads: OpenMP threads (0 = auto-detect)

    Returns:
        np.ndarray[uint8]: packed boolean array (F-order, Z-slices contiguous)
    """
    cdef int rx = xcc.shape[0]
    cdef int ry = ycc.shape[0]
    cdef int rz = zcc.shape[0]
    cdef long long total_bits = <long long>rx * <long long>ry * <long long>rz
    cdef long long total_bytes = (total_bits + 7) >> 3
    cdef long long slice_bits = <long long>rx * <long long>ry

    packed = np.zeros(total_bytes, dtype=np.uint8)
    cdef unsigned char[::1] out_view = packed
    cdef unsigned char *out = &out_view[0]

    cdef int i, j, k
    cdef long long lin_idx
    cdef double x, y, z, xp, yp, zp
    cdef int actual_threads
    cdef int has_transform = M4inv is not None

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

    if has_transform:
        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            z = zcc[k]
            for j in range(ry):
                y = ycc[j]
                for i in range(rx):
                    x = xcc[i]
                    xp = M4inv[0,0]*x + M4inv[0,1]*y + M4inv[0,2]*z + M4inv[0,3]
                    yp = M4inv[1,0]*x + M4inv[1,1]*y + M4inv[1,2]*z + M4inv[1,3]
                    zp = M4inv[2,0]*x + M4inv[2,1]*y + M4inv[2,2]*z + M4inv[2,3]
                    if fabs(xp - cx) <= half_sx and fabs(yp - cy) <= half_sy and fabs(zp - cz) <= half_sz:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)
    else:
        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            z = zcc[k]
            if fabs(z - cz) > half_sz:
                continue
            for j in range(ry):
                y = ycc[j]
                if fabs(y - cy) > half_sy:
                    continue
                for i in range(rx):
                    x = xcc[i]
                    if fabs(x - cx) <= half_sx:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)

    return packed


# ---------------------------------------------------------------------------
# Sphere kernel
# ---------------------------------------------------------------------------

def evaluate_and_pack_sphere(
    double[::1] xcc, double[::1] ycc, double[::1] zcc,
    double cx, double cy, double cz,
    double r_sq,
    double[:, ::1] M4inv=None,
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for Sphere.

    Args:
        xcc, ycc, zcc: 1D cell-center coordinate arrays
        cx, cy, cz: sphere center coordinates
        r_sq: radius squared
        M4inv: optional 4x4 inverse transform matrix
        n_threads: OpenMP threads (0 = auto-detect)
    """
    cdef int rx = xcc.shape[0]
    cdef int ry = ycc.shape[0]
    cdef int rz = zcc.shape[0]
    cdef long long total_bits = <long long>rx * <long long>ry * <long long>rz
    cdef long long total_bytes = (total_bits + 7) >> 3
    cdef long long slice_bits = <long long>rx * <long long>ry

    packed = np.zeros(total_bytes, dtype=np.uint8)
    cdef unsigned char[::1] out_view = packed
    cdef unsigned char *out = &out_view[0]

    cdef int i, j, k
    cdef long long lin_idx
    cdef double x, y, z, xp, yp, zp
    cdef double dx, dy, dz, dz_sq, dy_sq
    cdef int actual_threads
    cdef int has_transform = M4inv is not None

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

    if has_transform:
        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            z = zcc[k]
            for j in range(ry):
                y = ycc[j]
                for i in range(rx):
                    x = xcc[i]
                    xp = M4inv[0,0]*x + M4inv[0,1]*y + M4inv[0,2]*z + M4inv[0,3]
                    yp = M4inv[1,0]*x + M4inv[1,1]*y + M4inv[1,2]*z + M4inv[1,3]
                    zp = M4inv[2,0]*x + M4inv[2,1]*y + M4inv[2,2]*z + M4inv[2,3]
                    dx = xp - cx
                    dy = yp - cy
                    dz = zp - cz
                    if dx * dx + dy * dy + dz * dz <= r_sq:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)
    else:
        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            dz = zcc[k] - cz
            dz_sq = dz * dz
            if dz_sq > r_sq:
                continue
            for j in range(ry):
                dy = ycc[j] - cy
                dy_sq = dy * dy
                if dz_sq + dy_sq > r_sq:
                    continue
                for i in range(rx):
                    dx = xcc[i] - cx
                    if dx * dx + dy_sq + dz_sq <= r_sq:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)

    return packed


# ---------------------------------------------------------------------------
# Cylinder kernel (supports truncated cone via r1, r2)
# ---------------------------------------------------------------------------

def evaluate_and_pack_cylinder(
    double[::1] xcc, double[::1] ycc, double[::1] zcc,
    double cx, double cy, double cz,
    double h, double r1, double r2,
    double[:, ::1] M4inv=None,
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for Cylinder/Cone.

    Args:
        cx, cy, cz: center of cylinder
        h: height
        r1: radius at bottom (z = cz - h/2)
        r2: radius at top (z = cz + h/2)
        M4inv: optional 4x4 inverse transform matrix
        n_threads: OpenMP threads (0 = auto-detect)
    """
    cdef int rx = xcc.shape[0]
    cdef int ry = ycc.shape[0]
    cdef int rz = zcc.shape[0]
    cdef long long total_bits = <long long>rx * <long long>ry * <long long>rz
    cdef long long total_bytes = (total_bits + 7) >> 3
    cdef long long slice_bits = <long long>rx * <long long>ry

    packed = np.zeros(total_bytes, dtype=np.uint8)
    cdef unsigned char[::1] out_view = packed
    cdef unsigned char *out = &out_view[0]

    cdef int i, j, k
    cdef long long lin_idx
    cdef double x, y, z, xp, yp, zp
    cdef double dx, dy, Zc, Pz, R, R_sq, dist_sq
    cdef int actual_threads
    cdef int has_transform = M4inv is not None

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

    if has_transform:
        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            z = zcc[k]
            for j in range(ry):
                y = ycc[j]
                for i in range(rx):
                    x = xcc[i]
                    xp = M4inv[0,0]*x + M4inv[0,1]*y + M4inv[0,2]*z + M4inv[0,3]
                    yp = M4inv[1,0]*x + M4inv[1,1]*y + M4inv[1,2]*z + M4inv[1,3]
                    zp = M4inv[2,0]*x + M4inv[2,1]*y + M4inv[2,2]*z + M4inv[2,3]
                    Zc = zp - cz
                    Pz = Zc / h + 0.5
                    if Pz < 0.0 or Pz > 1.0:
                        continue
                    R = r1 * (1.0 - Pz) + r2 * Pz
                    R_sq = R * R
                    dx = xp - cx
                    dy = yp - cy
                    dist_sq = dx * dx + dy * dy
                    if dist_sq <= R_sq:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)
    else:
        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            Zc = zcc[k] - cz
            Pz = Zc / h + 0.5
            if Pz < 0.0 or Pz > 1.0:
                continue
            R = r1 * (1.0 - Pz) + r2 * Pz
            R_sq = R * R
            for j in range(ry):
                dy = ycc[j] - cy
                if dy * dy > R_sq:
                    continue
                for i in range(rx):
                    dx = xcc[i] - cx
                    dist_sq = dx * dx + dy * dy
                    if dist_sq <= R_sq:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)

    return packed


# ---------------------------------------------------------------------------
# GyroidCube kernel (precomputed per-axis trig — 0 trig in inner loop)
# ---------------------------------------------------------------------------

def evaluate_and_pack_gyroid(
    double[::1] xcc, double[::1] ycc, double[::1] zcc,
    double cx, double cy, double cz,
    double half_sx, double half_sy, double half_sz,
    double ax, double ay, double az,
    double phi_x, double phi_y, double phi_z,
    double structure_param,
    double thresh1, double thresh2,
    int use_thresh2,
    double[:, ::1] M4inv=None,
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for GyroidCube.

    Non-transform path precomputes cos/sin per axis (0 trig in inner loop).
    Transform path computes trig inline (6 trig calls per voxel).

    Args:
        ax, ay, az: pi * lattice_param (angular frequency per axis)
        phi_x, phi_y, phi_z: phase shifts
        structure_param: gyroid structure parameter
        thresh1, thresh2: threshold bounds for shell
        use_thresh2: if True, use (thresh1 < F < thresh2); else (0 < F < thresh1)
        M4inv: optional 4x4 inverse transform matrix
    """
    cdef int rx = xcc.shape[0]
    cdef int ry = ycc.shape[0]
    cdef int rz = zcc.shape[0]
    cdef long long total_bits = <long long>rx * <long long>ry * <long long>rz
    cdef long long total_bytes = (total_bits + 7) >> 3
    cdef long long slice_bits = <long long>rx * <long long>ry

    packed = np.zeros(total_bytes, dtype=np.uint8)
    cdef unsigned char[::1] out_view = packed
    cdef unsigned char *out = &out_view[0]

    cdef int i, j, k
    cdef long long lin_idx
    cdef double x, y, z, xp, yp, zp
    cdef double Xa, Ya, Za, cos_z, sin_z, F
    cdef double cosXa, sinXa, cosYa, sinYa, cosZa, sinZa
    cdef double lo, hi
    cdef int actual_threads
    cdef int has_transform = M4inv is not None
    cdef double[::1] cos_x_v, sin_x_v, cos_y_v, sin_y_v

    if use_thresh2:
        lo = thresh1
        hi = thresh2
    else:
        lo = 0.0
        hi = thresh1

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

    if has_transform:
        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            z = zcc[k]
            for j in range(ry):
                y = ycc[j]
                for i in range(rx):
                    x = xcc[i]
                    xp = M4inv[0,0]*x + M4inv[0,1]*y + M4inv[0,2]*z + M4inv[0,3]
                    yp = M4inv[1,0]*x + M4inv[1,1]*y + M4inv[1,2]*z + M4inv[1,3]
                    zp = M4inv[2,0]*x + M4inv[2,1]*y + M4inv[2,2]*z + M4inv[2,3]
                    if fabs(xp - cx) > half_sx or fabs(yp - cy) > half_sy or fabs(zp - cz) > half_sz:
                        continue
                    cosXa = cos(xp * ax + phi_x)
                    sinXa = sin(xp * ax + phi_x)
                    cosYa = cos(yp * ay + phi_y)
                    sinYa = sin(yp * ay + phi_y)
                    cosZa = cos(zp * az + phi_z)
                    sinZa = sin(zp * az + phi_z)
                    F = cosXa * sinYa + cosYa * sinZa + cosZa * sinXa - structure_param
                    if F > lo and F < hi:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)
    else:
        # Precompute per-axis trig arrays (axis-separable optimization)
        cos_x_arr = np.empty(rx, dtype=np.float64)
        sin_x_arr = np.empty(rx, dtype=np.float64)
        cos_y_arr = np.empty(ry, dtype=np.float64)
        sin_y_arr = np.empty(ry, dtype=np.float64)
        cos_x_v = cos_x_arr
        sin_x_v = sin_x_arr
        cos_y_v = cos_y_arr
        sin_y_v = sin_y_arr

        for i in range(rx):
            Xa = xcc[i] * ax + phi_x
            cos_x_arr[i] = cos(Xa)
            sin_x_arr[i] = sin(Xa)
        for j in range(ry):
            Ya = ycc[j] * ay + phi_y
            cos_y_arr[j] = cos(Ya)
            sin_y_arr[j] = sin(Ya)

        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            if fabs(zcc[k] - cz) > half_sz:
                continue
            cos_z = cos(zcc[k] * az + phi_z)
            sin_z = sin(zcc[k] * az + phi_z)
            for j in range(ry):
                if fabs(ycc[j] - cy) > half_sy:
                    continue
                for i in range(rx):
                    if fabs(xcc[i] - cx) > half_sx:
                        continue
                    F = cos_x_v[i] * sin_y_v[j] + \
                        cos_y_v[j] * sin_z + \
                        cos_z * sin_x_v[i] - structure_param
                    if F > lo and F < hi:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)

    return packed


# ---------------------------------------------------------------------------
# WigglyGyroidCube kernel
# ---------------------------------------------------------------------------

def evaluate_and_pack_wiggly_gyroid(
    double[::1] xcc, double[::1] ycc, double[::1] zcc,
    double cx, double cy, double cz,
    double half_sx, double half_sy, double half_sz,
    double ax, double ay, double az,
    double phi_x, double phi_y, double phi_z,
    double structure_param,
    double thresh1, double thresh2,
    double w_freq, double w_amp, int w_expon,
    double[:, ::1] M4inv=None,
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for WigglyGyroidCube.

    Non-transform path precomputes per-axis base trig and wiggle trig.
    Transform path computes all trig inline from transformed coordinates.
    """
    cdef int rx = xcc.shape[0]
    cdef int ry = ycc.shape[0]
    cdef int rz = zcc.shape[0]
    cdef long long total_bits = <long long>rx * <long long>ry * <long long>rz
    cdef long long total_bytes = (total_bits + 7) >> 3
    cdef long long slice_bits = <long long>rx * <long long>ry

    packed = np.zeros(total_bytes, dtype=np.uint8)
    cdef unsigned char[::1] out_view = packed
    cdef unsigned char *out = &out_view[0]

    cdef int i, j, k
    cdef long long lin_idx
    cdef double x, y, z, xp, yp, zp
    cdef double Xa, Ya, Za, bX, bY, bZ
    cdef double cosX, sinX, cosY, sinY, cosZ, sinZ
    cdef double cos_bX, sin_bX, cos_bY, sin_bY, cos_bZ, sin_bZ
    cdef double gradX, gradY, gradZ
    cdef double wx, wy, wz
    cdef double x1, y1, z1, x2, y2, z2
    cdef double Fw1, Fw2
    cdef int inside
    cdef int actual_threads
    cdef int has_transform = M4inv is not None
    cdef double[::1] cos_x, sin_x, cos_bx, sin_bx, xa_v
    cdef double[::1] cos_y, sin_y, cos_by, sin_by, ya_v

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

    if has_transform:
        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            z = zcc[k]
            for j in range(ry):
                y = ycc[j]
                for i in range(rx):
                    x = xcc[i]
                    xp = M4inv[0,0]*x + M4inv[0,1]*y + M4inv[0,2]*z + M4inv[0,3]
                    yp = M4inv[1,0]*x + M4inv[1,1]*y + M4inv[1,2]*z + M4inv[1,3]
                    zp = M4inv[2,0]*x + M4inv[2,1]*y + M4inv[2,2]*z + M4inv[2,3]
                    if fabs(xp - cx) > half_sx or fabs(yp - cy) > half_sy or fabs(zp - cz) > half_sz:
                        continue
                    Xa = xp * ax
                    Ya = yp * ay
                    Za = zp * az
                    cosX = cos(Xa); sinX = sin(Xa)
                    cosY = cos(Ya); sinY = sin(Ya)
                    cosZ = cos(Za); sinZ = sin(Za)
                    gradX = cosZ * cosX - sinX * sinY
                    gradY = cosX * cosY - sinY * sinZ
                    gradZ = cosY * cosZ - sinZ * sinX
                    bX = w_freq * Xa; bY = w_freq * Ya; bZ = w_freq * Za
                    cos_bX = cos(bX); sin_bX = sin(bX)
                    cos_bY = cos(bY); sin_bY = sin(bY)
                    cos_bZ = cos(bZ); sin_bZ = sin(bZ)
                    wx = w_amp * cpow(cos_bY * sin_bZ, w_expon)
                    wy = w_amp * cpow(sin_bX * cos_bZ, w_expon)
                    wz = w_amp * cpow(cos_bX * sin_bY, w_expon)
                    x1 = Xa - wx * gradX
                    y1 = Ya - wy * gradY
                    z1 = Za - wz * gradZ
                    Fw1 = cos(x1) * sin(y1) + cos(y1) * sin(z1) + cos(z1) * sin(x1) - structure_param
                    x2 = Xa + wx * gradX
                    y2 = Ya + wy * gradY
                    z2 = Za + wz * gradZ
                    Fw2 = cos(x2) * sin(y2) + cos(y2) * sin(z2) + cos(z2) * sin(x2) - structure_param
                    inside = ((Fw1 > thresh1) and (Fw1 < thresh2)) or \
                             ((Fw2 > thresh1) and (Fw2 < thresh2))
                    if inside:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)
    else:
        # Precompute per-axis trig arrays (axis-separable optimization)
        cos_x = np.empty(rx, dtype=np.float64)
        sin_x = np.empty(rx, dtype=np.float64)
        cos_bx = np.empty(rx, dtype=np.float64)
        sin_bx = np.empty(rx, dtype=np.float64)
        xa_v = np.empty(rx, dtype=np.float64)
        cos_y = np.empty(ry, dtype=np.float64)
        sin_y = np.empty(ry, dtype=np.float64)
        cos_by = np.empty(ry, dtype=np.float64)
        sin_by = np.empty(ry, dtype=np.float64)
        ya_v = np.empty(ry, dtype=np.float64)

        for i in range(rx):
            Xa = xcc[i] * ax
            xa_v[i] = Xa
            cos_x[i] = cos(Xa)
            sin_x[i] = sin(Xa)
            bX = w_freq * Xa
            cos_bx[i] = cos(bX)
            sin_bx[i] = sin(bX)
        for j in range(ry):
            Ya = ycc[j] * ay
            ya_v[j] = Ya
            cos_y[j] = cos(Ya)
            sin_y[j] = sin(Ya)
            bY = w_freq * Ya
            cos_by[j] = cos(bY)
            sin_by[j] = sin(bY)

        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            if fabs(zcc[k] - cz) > half_sz:
                continue
            Za = zcc[k] * az
            cosZ = cos(Za)
            sinZ = sin(Za)
            bZ = w_freq * Za
            cos_bZ = cos(bZ)
            sin_bZ = sin(bZ)
            for j in range(ry):
                if fabs(ycc[j] - cy) > half_sy:
                    continue
                cosY = cos_y[j]
                sinY = sin_y[j]
                for i in range(rx):
                    if fabs(xcc[i] - cx) > half_sx:
                        continue
                    cosX = cos_x[i]
                    sinX = sin_x[i]
                    gradX = cosZ * cosX - sinX * sinY
                    gradY = cosX * cosY - sinY * sinZ
                    gradZ = cosY * cosZ - sinZ * sinX
                    wx = w_amp * cpow(cos_by[j] * sin_bZ, w_expon)
                    wy = w_amp * cpow(sin_bx[i] * cos_bZ, w_expon)
                    wz = w_amp * cpow(cos_bx[i] * sin_by[j], w_expon)
                    x1 = xa_v[i] - wx * gradX
                    y1 = ya_v[j] - wy * gradY
                    z1 = Za - wz * gradZ
                    Fw1 = cos(x1) * sin(y1) + cos(y1) * sin(z1) + cos(z1) * sin(x1) - structure_param
                    x2 = xa_v[i] + wx * gradX
                    y2 = ya_v[j] + wy * gradY
                    z2 = Za + wz * gradZ
                    Fw2 = cos(x2) * sin(y2) + cos(y2) * sin(z2) + cos(z2) * sin(x2) - structure_param
                    inside = ((Fw1 > thresh1) and (Fw1 < thresh2)) or \
                             ((Fw2 > thresh1) and (Fw2 < thresh2))
                    if inside:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)

    return packed


# ---------------------------------------------------------------------------
# HyperWigglyGyroidCube kernel
# ---------------------------------------------------------------------------

def evaluate_and_pack_hyperwiggly_gyroid(
    double[::1] xcc, double[::1] ycc, double[::1] zcc,
    double cx, double cy, double cz,
    double half_sx, double half_sy, double half_sz,
    double ax, double ay, double az,
    double phi_x, double phi_y, double phi_z,
    double structure_param,
    double thresh1, double thresh2,
    double w_freq, double w_amp, int w_expon,
    double[:, ::1] M4inv=None,
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for HyperWigglyGyroidCube.

    Same as WigglyGyroidCube but with double-frequency modulation terms.
    Non-transform path precomputes per-axis trig. Transform path inline.
    """
    cdef int rx = xcc.shape[0]
    cdef int ry = ycc.shape[0]
    cdef int rz = zcc.shape[0]
    cdef long long total_bits = <long long>rx * <long long>ry * <long long>rz
    cdef long long total_bytes = (total_bits + 7) >> 3
    cdef long long slice_bits = <long long>rx * <long long>ry

    packed = np.zeros(total_bytes, dtype=np.uint8)
    cdef unsigned char[::1] out_view = packed
    cdef unsigned char *out = &out_view[0]

    cdef int i, j, k
    cdef long long lin_idx
    cdef double x, y, z, xp, yp, zp
    cdef double Xa, Ya, Za, bX, bY, bZ
    cdef double cosX, sinX, cosY, sinY, cosZ, sinZ
    cdef double cos_bX, sin_bX, cos_bY, sin_bY, cos_bZ, sin_bZ
    cdef double cos_3bX, sin_3bX, cos_3bY, sin_3bY, cos_3bZ, sin_3bZ
    cdef double gradX, gradY, gradZ
    cdef double wx, wy, wz
    cdef double x1, y1, z1, x2, y2, z2
    cdef double Fw1, Fw2
    cdef int inside
    cdef int actual_threads
    cdef int has_transform = M4inv is not None
    cdef double half_amp = 0.5 * w_amp
    cdef int p1 = w_expon + 1
    cdef double[::1] hw_cos_x, hw_sin_x, hw_cos_bx, hw_sin_bx, hw_cos_3bx, hw_sin_3bx, hw_xa_v
    cdef double[::1] hw_cos_y, hw_sin_y, hw_cos_by, hw_sin_by, hw_cos_3by, hw_sin_3by, hw_ya_v

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

    if has_transform:
        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            z = zcc[k]
            for j in range(ry):
                y = ycc[j]
                for i in range(rx):
                    x = xcc[i]
                    xp = M4inv[0,0]*x + M4inv[0,1]*y + M4inv[0,2]*z + M4inv[0,3]
                    yp = M4inv[1,0]*x + M4inv[1,1]*y + M4inv[1,2]*z + M4inv[1,3]
                    zp = M4inv[2,0]*x + M4inv[2,1]*y + M4inv[2,2]*z + M4inv[2,3]
                    if fabs(xp - cx) > half_sx or fabs(yp - cy) > half_sy or fabs(zp - cz) > half_sz:
                        continue
                    Xa = xp * ax
                    Ya = yp * ay
                    Za = zp * az
                    cosX = cos(Xa); sinX = sin(Xa)
                    cosY = cos(Ya); sinY = sin(Ya)
                    cosZ = cos(Za); sinZ = sin(Za)
                    gradX = cosZ * cosX - sinX * sinY
                    gradY = cosX * cosY - sinY * sinZ
                    gradZ = cosY * cosZ - sinZ * sinX
                    bX = w_freq * Xa; bY = w_freq * Ya; bZ = w_freq * Za
                    cos_bX = cos(bX); sin_bX = sin(bX)
                    cos_bY = cos(bY); sin_bY = sin(bY)
                    cos_bZ = cos(bZ); sin_bZ = sin(bZ)
                    cos_3bX = cos(3.0 * bX); sin_3bX = sin(3.0 * bX)
                    cos_3bY = cos(3.0 * bY); sin_3bY = sin(3.0 * bY)
                    cos_3bZ = cos(3.0 * bZ); sin_3bZ = sin(3.0 * bZ)
                    wx = w_amp * cpow(cos_bY * sin_bZ, w_expon) + \
                         half_amp * cpow(cos_3bY * sin_3bZ, p1)
                    wy = w_amp * cpow(sin_bX * cos_bZ, w_expon) + \
                         half_amp * cpow(sin_3bX * cos_3bZ, p1)
                    wz = w_amp * cpow(cos_bX * sin_bY, w_expon) + \
                         half_amp * cpow(cos_3bX * sin_3bY, p1)
                    x1 = Xa - wx * gradX
                    y1 = Ya - wy * gradY
                    z1 = Za - wz * gradZ
                    Fw1 = cos(x1) * sin(y1) + cos(y1) * sin(z1) + cos(z1) * sin(x1) - structure_param
                    x2 = Xa + wx * gradX
                    y2 = Ya + wy * gradY
                    z2 = Za + wz * gradZ
                    Fw2 = cos(x2) * sin(y2) + cos(y2) * sin(z2) + cos(z2) * sin(x2) - structure_param
                    inside = ((Fw1 > thresh1) and (Fw1 < thresh2)) or \
                             ((Fw2 > thresh1) and (Fw2 < thresh2))
                    if inside:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)
    else:
        # Precompute per-axis trig arrays (axis-separable optimization)
        hw_cos_x = np.empty(rx, dtype=np.float64)
        hw_sin_x = np.empty(rx, dtype=np.float64)
        hw_cos_bx = np.empty(rx, dtype=np.float64)
        hw_sin_bx = np.empty(rx, dtype=np.float64)
        hw_cos_3bx = np.empty(rx, dtype=np.float64)
        hw_sin_3bx = np.empty(rx, dtype=np.float64)
        hw_xa_v = np.empty(rx, dtype=np.float64)
        hw_cos_y = np.empty(ry, dtype=np.float64)
        hw_sin_y = np.empty(ry, dtype=np.float64)
        hw_cos_by = np.empty(ry, dtype=np.float64)
        hw_sin_by = np.empty(ry, dtype=np.float64)
        hw_cos_3by = np.empty(ry, dtype=np.float64)
        hw_sin_3by = np.empty(ry, dtype=np.float64)
        hw_ya_v = np.empty(ry, dtype=np.float64)

        for i in range(rx):
            Xa = xcc[i] * ax
            hw_xa_v[i] = Xa
            hw_cos_x[i] = cos(Xa)
            hw_sin_x[i] = sin(Xa)
            bX = w_freq * Xa
            hw_cos_bx[i] = cos(bX)
            hw_sin_bx[i] = sin(bX)
            hw_cos_3bx[i] = cos(3.0 * bX)
            hw_sin_3bx[i] = sin(3.0 * bX)
        for j in range(ry):
            Ya = ycc[j] * ay
            hw_ya_v[j] = Ya
            hw_cos_y[j] = cos(Ya)
            hw_sin_y[j] = sin(Ya)
            bY = w_freq * Ya
            hw_cos_by[j] = cos(bY)
            hw_sin_by[j] = sin(bY)
            hw_cos_3by[j] = cos(3.0 * bY)
            hw_sin_3by[j] = sin(3.0 * bY)

        for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
            if fabs(zcc[k] - cz) > half_sz:
                continue
            Za = zcc[k] * az
            cosZ = cos(Za)
            sinZ = sin(Za)
            bZ = w_freq * Za
            cos_bZ = cos(bZ)
            sin_bZ = sin(bZ)
            cos_3bZ = cos(3.0 * bZ)
            sin_3bZ = sin(3.0 * bZ)
            for j in range(ry):
                if fabs(ycc[j] - cy) > half_sy:
                    continue
                cosY = hw_cos_y[j]
                sinY = hw_sin_y[j]
                for i in range(rx):
                    if fabs(xcc[i] - cx) > half_sx:
                        continue
                    cosX = hw_cos_x[i]
                    sinX = hw_sin_x[i]
                    gradX = cosZ * cosX - sinX * sinY
                    gradY = cosX * cosY - sinY * sinZ
                    gradZ = cosY * cosZ - sinZ * sinX
                    wx = w_amp * cpow(hw_cos_by[j] * sin_bZ, w_expon) + \
                         half_amp * cpow(hw_cos_3by[j] * sin_3bZ, p1)
                    wy = w_amp * cpow(hw_sin_bx[i] * cos_bZ, w_expon) + \
                         half_amp * cpow(hw_sin_3bx[i] * cos_3bZ, p1)
                    wz = w_amp * cpow(hw_cos_bx[i] * hw_sin_by[j], w_expon) + \
                         half_amp * cpow(hw_cos_3bx[i] * hw_sin_3by[j], p1)
                    x1 = hw_xa_v[i] - wx * gradX
                    y1 = hw_ya_v[j] - wy * gradY
                    z1 = Za - wz * gradZ
                    Fw1 = cos(x1) * sin(y1) + cos(y1) * sin(z1) + cos(z1) * sin(x1) - structure_param
                    x2 = hw_xa_v[i] + wx * gradX
                    y2 = hw_ya_v[j] + wy * gradY
                    z2 = Za + wz * gradZ
                    Fw2 = cos(x2) * sin(y2) + cos(y2) * sin(z2) + cos(z2) * sin(x2) - structure_param
                    inside = ((Fw1 > thresh1) and (Fw1 < thresh2)) or \
                             ((Fw2 > thresh1) and (Fw2 < thresh2))
                    if inside:
                        lin_idx = i + j * rx + k * slice_bits
                        set_bit(out, lin_idx)

    return packed


# ---------------------------------------------------------------------------
# Resample-and-pack kernel (data-only models: nearest-neighbor lookup)
# ---------------------------------------------------------------------------

def resample_and_pack(
    double[::1] src_x, double[::1] src_y, double[::1] src_z,
    unsigned char[::1] src_packed,
    long long src_rx, long long src_ry, long long src_rz,
    double[::1] dst_x, double[::1] dst_y, double[::1] dst_z,
    double[:, ::1] M4inv=None,
    int n_threads=0,
):
    """Fused nearest-neighbor resample + bit-pack for data-only VoxelModels.

    Looks up each destination voxel in the source packed array via
    nearest-neighbor indexing. Handles optional M4inv transform.

    Args:
        src_x, src_y, src_z: 1D source grid coordinate arrays
        src_packed: source packed uint8 array (F-order)
        src_rx, src_ry, src_rz: source grid resolution
        dst_x, dst_y, dst_z: 1D destination grid coordinate arrays
        M4inv: optional 4x4 inverse transform matrix
        n_threads: OpenMP threads (0 = auto-detect)

    Returns:
        np.ndarray[uint8]: packed boolean array (F-order)
    """
    cdef int drx = dst_x.shape[0]
    cdef int dry = dst_y.shape[0]
    cdef int drz = dst_z.shape[0]
    cdef long long total_bits = <long long>drx * <long long>dry * <long long>drz
    cdef long long total_bytes = (total_bits + 7) >> 3
    cdef long long slice_bits = <long long>drx * <long long>dry
    cdef long long src_slice_bits = src_rx * src_ry

    packed = np.zeros(total_bytes, dtype=np.uint8)
    cdef unsigned char[::1] out_view = packed
    cdef unsigned char *out = &out_view[0]
    cdef unsigned char *src = &src_packed[0]

    cdef double src_x0 = src_x[0]
    cdef double src_y0 = src_y[0]
    cdef double src_z0 = src_z[0]
    cdef double src_dx = src_x[1] - src_x[0] if src_rx > 1 else 1.0
    cdef double src_dy = src_y[1] - src_y[0] if src_ry > 1 else 1.0
    cdef double src_dz = src_z[1] - src_z[0] if src_rz > 1 else 1.0
    cdef int has_transform = M4inv is not None

    cdef int i, j, k
    cdef long long lin_idx, src_lin
    cdef long long si, sj, sk
    cdef double x, y, z, xp, yp, zp
    cdef int actual_threads

    if n_threads <= 0:
        n_threads = _get_optimal_threads(drz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

    for k in prange(drz, nogil=True, num_threads=actual_threads, schedule='static'):
        z = dst_z[k]
        for j in range(dry):
            y = dst_y[j]
            for i in range(drx):
                x = dst_x[i]
                if has_transform:
                    xp = M4inv[0,0]*x + M4inv[0,1]*y + M4inv[0,2]*z + M4inv[0,3]
                    yp = M4inv[1,0]*x + M4inv[1,1]*y + M4inv[1,2]*z + M4inv[1,3]
                    zp = M4inv[2,0]*x + M4inv[2,1]*y + M4inv[2,2]*z + M4inv[2,3]
                else:
                    xp = x
                    yp = y
                    zp = z
                # Nearest-neighbor index in source grid
                si = <long long>((xp - src_x0) / src_dx + 0.5)
                sj = <long long>((yp - src_y0) / src_dy + 0.5)
                sk = <long long>((zp - src_z0) / src_dz + 0.5)
                # Bounds check + bit lookup in F-order packed array
                if 0 <= si < src_rx and 0 <= sj < src_ry and 0 <= sk < src_rz:
                    src_lin = si + sj * src_rx + sk * src_slice_bits
                    if (src[src_lin >> 3] >> (7 - <int>(src_lin & 7))) & 1:
                        lin_idx = i + j * drx + k * slice_bits
                        set_bit(out, lin_idx)

    return packed


# ---------------------------------------------------------------------------
# Chessboard CDT -> int8 SDF (reads packed bits, eliminates memory balloon)
# ---------------------------------------------------------------------------

cdef inline int _read_packed_bit(
    const unsigned char *packed, long long lin_idx,
) noexcept nogil:
    """Read a single bit from MSB-first packed array."""
    return (packed[lin_idx >> 3] >> (7 - <int>(lin_idx & 7))) & 1


cdef void _cdt_init(
    signed char *dist,
    const unsigned char *packed,
    long long rx, long long ry, long long rz,
    int stride, int px, int py, int pz,
    int exterior,
    int n_threads,
) noexcept nogil:
    """Initialize CDT distance array from packed bits.

    Interior (exterior=0): empty->127 (background), occupied->0 (seed)
    Exterior (exterior=1): empty->0 (seed), occupied->127 (background)
    """
    cdef long long total = <long long>px * <long long>py * <long long>pz
    cdef signed char fg_val
    cdef int si, sj, sk, pi, pj, pk
    cdef long long full_lin
    cdef int sx = <int>((rx + stride - 1) // stride)
    cdef int sy = <int>((ry + stride - 1) // stride)
    cdef int sz = <int>((rz + stride - 1) // stride)
    cdef long long full_slice = rx * ry
    cdef int pad_slice = px * py

    # Interior (exterior=0): CDT(V) = distance to nearest empty voxel.
    #   Occupied voxels -> 127 (to be computed), empty voxels -> 0 (seeds).
    #   Default fill = 0 (empty/padding). Set occupied bits to 127.
    # Exterior (exterior=1): CDT(~V) = distance to nearest occupied voxel.
    #   Empty voxels -> 127 (to be computed), occupied voxels -> 0 (seeds).
    #   Default fill = 127 (empty/padding are far). Set occupied bits to 0.
    if exterior:
        memset(dist, 127, <size_t>total)
        fg_val = 0
    else:
        memset(dist, 0, <size_t>total)
        fg_val = 127

    for sk in prange(sz, num_threads=n_threads, schedule='static'):
        pk = sk + 2
        for sj in range(sy):
            pj = sj + 2
            for si in range(sx):
                pi = si + 2
                full_lin = (<long long>(si * stride) +
                            <long long>(sj * stride) * rx +
                            <long long>(sk * stride) * full_slice)
                if _read_packed_bit(packed, full_lin):
                    dist[pi + pj * px + pk * pad_slice] = fg_val


cdef void _cdt_forward(signed char *d, int px, int py, int pz) noexcept nogil:
    """Forward raster scan: 13 forward-cone neighbors, chessboard cost=1."""
    cdef int i, j, k, idx, v, nv
    cdef int sxy = px * py
    cdef int base_k, base_km1
    cdef int base_jk, base_jm1_k, base_jm1_km1, base_j_km1, base_jp1_km1

    for k in range(1, pz - 1):
        base_k = k * sxy
        base_km1 = (k - 1) * sxy
        for j in range(1, py - 1):
            base_jk = j * px + base_k
            base_jm1_k = (j - 1) * px + base_k
            base_jm1_km1 = (j - 1) * px + base_km1
            base_j_km1 = j * px + base_km1
            base_jp1_km1 = (j + 1) * px + base_km1
            for i in range(1, px - 1):
                idx = i + base_jk
                v = d[idx]
                if v == 0:
                    continue
                # k-1 plane (9 neighbors)
                nv = d[i - 1 + base_jm1_km1] + 1
                if nv < v: v = nv
                nv = d[i + base_jm1_km1] + 1
                if nv < v: v = nv
                nv = d[i + 1 + base_jm1_km1] + 1
                if nv < v: v = nv
                nv = d[i - 1 + base_j_km1] + 1
                if nv < v: v = nv
                nv = d[i + base_j_km1] + 1
                if nv < v: v = nv
                nv = d[i + 1 + base_j_km1] + 1
                if nv < v: v = nv
                nv = d[i - 1 + base_jp1_km1] + 1
                if nv < v: v = nv
                nv = d[i + base_jp1_km1] + 1
                if nv < v: v = nv
                nv = d[i + 1 + base_jp1_km1] + 1
                if nv < v: v = nv
                # same k, j-1 row (3 neighbors)
                nv = d[i - 1 + base_jm1_k] + 1
                if nv < v: v = nv
                nv = d[i + base_jm1_k] + 1
                if nv < v: v = nv
                nv = d[i + 1 + base_jm1_k] + 1
                if nv < v: v = nv
                # same k, same j, i-1
                nv = d[idx - 1] + 1
                if nv < v: v = nv
                d[idx] = <signed char>v


cdef void _cdt_backward(signed char *d, int px, int py, int pz) noexcept nogil:
    """Backward raster scan: 13 backward-cone neighbors, chessboard cost=1."""
    cdef int i, j, k, idx, v, nv
    cdef int sxy = px * py
    cdef int base_k, base_kp1
    cdef int base_jk, base_jp1_k, base_jm1_kp1, base_j_kp1, base_jp1_kp1

    for k in range(pz - 2, 0, -1):
        base_k = k * sxy
        base_kp1 = (k + 1) * sxy
        for j in range(py - 2, 0, -1):
            base_jk = j * px + base_k
            base_jp1_k = (j + 1) * px + base_k
            base_jm1_kp1 = (j - 1) * px + base_kp1
            base_j_kp1 = j * px + base_kp1
            base_jp1_kp1 = (j + 1) * px + base_kp1
            for i in range(px - 2, 0, -1):
                idx = i + base_jk
                v = d[idx]
                if v == 0:
                    continue
                # k+1 plane (9 neighbors)
                nv = d[i - 1 + base_jm1_kp1] + 1
                if nv < v: v = nv
                nv = d[i + base_jm1_kp1] + 1
                if nv < v: v = nv
                nv = d[i + 1 + base_jm1_kp1] + 1
                if nv < v: v = nv
                nv = d[i - 1 + base_j_kp1] + 1
                if nv < v: v = nv
                nv = d[i + base_j_kp1] + 1
                if nv < v: v = nv
                nv = d[i + 1 + base_j_kp1] + 1
                if nv < v: v = nv
                nv = d[i - 1 + base_jp1_kp1] + 1
                if nv < v: v = nv
                nv = d[i + base_jp1_kp1] + 1
                if nv < v: v = nv
                nv = d[i + 1 + base_jp1_kp1] + 1
                if nv < v: v = nv
                # same k, j+1 row (3 neighbors)
                nv = d[i - 1 + base_jp1_k] + 1
                if nv < v: v = nv
                nv = d[i + base_jp1_k] + 1
                if nv < v: v = nv
                nv = d[i + 1 + base_jp1_k] + 1
                if nv < v: v = nv
                # same k, same j, i+1
                nv = d[idx + 1] + 1
                if nv < v: v = nv
                d[idx] = <signed char>v


def compute_sdf_cdt(
    const unsigned char[::1] packed,
    long long rx, long long ry, long long rz,
    int stride,
    int n_threads=0,
):
    """Compute int8 SDF from packed binary voxels using chessboard CDT.

    Reads packed F-order bits directly, applies stride, pads by 1 voxel,
    computes interior and exterior chessboard distance transforms (2-pass
    raster scan each), returns SDF = interior - exterior as int8.

    Distances clamped to 127 (int8 max). Only the zero-crossing matters
    for marching cubes; Butterworth impulse response is 99% within radius 3.

    Args:
        packed: F-order packed binary volume
        rx, ry, rz: full-resolution grid dimensions
        stride: subsample factor (1 = no stride)
        n_threads: OpenMP threads (0 = auto-detect)

    Returns:
        np.ndarray[int8]: SDF array, F-order shape (px, py, pz) where
        px = ceil(rx/stride)+2, etc. +2 is 1-voxel padding per side.
        Positive inside, negative outside.
    """
    cdef int sx = <int>((rx + stride - 1) // stride)
    cdef int sy = <int>((ry + stride - 1) // stride)
    cdef int sz = <int>((rz + stride - 1) // stride)
    # Pad by 2: outer layer is sacrificial (never scanned), inner layer
    # ensures CDT propagation reaches all data voxels. Data starts at
    # index 2, CDT scans [1, dim-2]. Zero-crossing guaranteed inside.
    cdef int px = sx + 4
    cdef int py = sy + 4
    cdef int pz = sz + 4
    cdef long long total = <long long>px * <long long>py * <long long>pz
    cdef int pad_slice = px * py

    if n_threads <= 0:
        n_threads = _get_optimal_threads(pz)

    interior_arr = np.empty(total, dtype=np.int8)
    exterior_arr = np.empty(total, dtype=np.int8)
    cdef signed char[::1] interior_view = interior_arr
    cdef signed char[::1] exterior_view = exterior_arr
    cdef signed char *interior = &interior_view[0]
    cdef signed char *exterior = &exterior_view[0]
    cdef const unsigned char *src = &packed[0]
    cdef long long idx
    cdef int k

    with nogil:
        # Interior CDT: occupied->0 (seed), empty->127
        _cdt_init(interior, src, rx, ry, rz, stride, px, py, pz, 0, n_threads)
        _cdt_forward(interior, px, py, pz)
        _cdt_backward(interior, px, py, pz)

        # Exterior CDT: empty->0 (seed), occupied->127
        _cdt_init(exterior, src, rx, ry, rz, stride, px, py, pz, 1, n_threads)
        _cdt_forward(exterior, px, py, pz)
        _cdt_backward(exterior, px, py, pz)

        # SDF = interior - exterior (in-place into interior)
        for k in prange(pz, num_threads=n_threads, schedule='static'):
            for idx in range(<long long>(k * pad_slice),
                             <long long>((k + 1) * pad_slice)):
                interior[idx] = <signed char>(
                    <int>interior[idx] - <int>exterior[idx])

    del exterior_arr
    # Strip sacrificial outer padding, keep inner 1-voxel pad for MC
    sdf_3d = interior_arr.reshape((px, py, pz), order='F')
    return sdf_3d[1:-1, 1:-1, 1:-1].copy()


# ---------------------------------------------------------------------------
# Fused scale + convolve (replaces CDT + convolve for scaled smoothing path)
# ---------------------------------------------------------------------------

def fused_scale_convolve(
    const unsigned char[::1] packed,
    long long rx, long long ry, long long rz,
    const signed char[:, :, ::1] kernel,
    int stride=1,
    int n_threads=0,
):
    """Fused unpack-scale-convolve: packed bits -> smoothed int8 field.

    Replaces compute_sdf_cdt() + convolve_sdf_spatial() for the scaled
    smoothing fast path. Each packed bit is scaled to {-1, +1}
    (empty/solid) and convolved with the int8 Butterworth kernel
    (sum ~256) using int16 accumulation. Accumulator range [-256, +256]
    maps smoothly to int8 output, producing a wide transition band
    at the surface for precise MC vertex interpolation.

    No intermediate volumes: bits are unpacked on-the-fly during
    convolution. Memory: output array only (padded output bytes).

    The output is padded by 1 voxel on each side (matching
    compute_sdf_cdt convention) so that marching cubes has boundary
    context.

    Args:
        packed: F-order packed binary volume (big-endian bitorder)
        rx, ry, rz: full-resolution grid dimensions
        kernel: int8 quantized Butterworth kernel, shape (2r+1, 2r+1, 2r+1)
        stride: subsample factor (1 = full resolution, 2 = half, etc.)
        n_threads: OpenMP threads (0 = auto-detect)

    Returns:
        np.ndarray[int8]: smoothed field, C-contiguous.
            Shape (sx+2, sy+2, sz+2) where s* = ceil(r*/stride).
            1-voxel padding on each side (padding = -1, empty).
            Positive inside, negative outside. Zero-crossing at surface.
    """
    # Strided output dimensions (same as compute_sdf_cdt)
    cdef int sx = <int>((rx + stride - 1) // stride)
    cdef int sy = <int>((ry + stride - 1) // stride)
    cdef int sz = <int>((rz + stride - 1) // stride)
    # Add 1-voxel padding per side for MC boundary context
    cdef int px = sx + 2
    cdef int py = sy + 2
    cdef int pz = sz + 2

    cdef int kx = kernel.shape[0]
    cdef int ky = kernel.shape[1]
    cdef int kz = kernel.shape[2]
    cdef int krx = kx // 2
    cdef int kry = ky // 2
    cdef int krz = kz // 2

    # Initialize to -1 (empty) so padding border is correct without fill loop
    out_arr = np.full((px, py, pz), -1, dtype=np.int8)
    cdef signed char[:, :, ::1] out = out_arr

    cdef int i, j, k, di, dj, dk
    cdef int si, sj, sk
    cdef int fi, fj, fk
    cdef short acc
    cdef int val
    cdef long long bit_idx, byte_idx
    cdef int bit_pos
    cdef signed char src_val
    cdef const unsigned char *src = &packed[0]
    cdef long long rx_ll = rx
    cdef long long rxy = rx * ry
    cdef int str = stride

    if n_threads <= 0:
        n_threads = _get_optimal_threads(px)

    with nogil:
        # Iterate over padded output grid. Data voxels at [1..sx, 1..sy, 1..sz].
        # Padding voxels (i=0, i=sx+1, etc.) stay as 0 → filled below.
        for i in prange(sx, num_threads=n_threads, schedule='static'):
            for j in range(sy):
                for k in range(sz):
                    acc = 0
                    for di in range(kx):
                        # Source voxel in strided coords
                        si = i + di - krx
                        for dj in range(ky):
                            sj = j + dj - kry
                            for dk in range(kz):
                                sk = k + dk - krz
                                # Map strided coords to full-res
                                fi = si * str
                                fj = sj * str
                                fk = sk * str
                                # Out-of-bounds in full-res = empty
                                if (fi < 0 or fi >= <int>rx or
                                        fj < 0 or fj >= <int>ry or
                                        fk < 0 or fk >= <int>rz):
                                    src_val = -1
                                else:
                                    # F-order bit index in packed array
                                    bit_idx = (
                                        <long long>fi +
                                        <long long>fj * rx_ll +
                                        <long long>fk * rxy)
                                    byte_idx = bit_idx >> 3
                                    bit_pos = <int>(bit_idx & 7)
                                    # big-endian packbits: MSB first
                                    if (src[byte_idx] >> (7 - bit_pos)) & 1:
                                        src_val = 1
                                    else:
                                        src_val = -1
                                acc = acc + <short>(
                                    <short>src_val *
                                    <short>kernel[di, dj, dk])
                    # Clip to int8 range
                    val = <int>acc
                    if val > 127:
                        val = 127
                    elif val < -128:
                        val = -128
                    # Write to padded output (+1 offset for padding)
                    out[i + 1, j + 1, k + 1] = <signed char>val

    return out_arr


# ---------------------------------------------------------------------------
# Int8 spatial convolution (replaces FFT Butterworth for narrow kernels)
# ---------------------------------------------------------------------------

def convolve_sdf_spatial(
    const signed char[:, :, ::1] sdf,
    const signed char[:, :, ::1] kernel,
    int n_threads=0,
):
    """Apply int8 spatial convolution to int8 SDF volume.

    Replaces the FFT Butterworth pipeline for narrow kernels (radius <= 3).
    int8 x int8 -> int16 accumulation, clipped back to int8. The zero-crossing
    is scale-invariant: multiplying the kernel by 256 doesn't move where the
    output crosses zero.

    Memory: only input + output arrays. No FFT, no complex64, no float32
    intermediates. At 504^3: 32 MB total vs 514 MB for FFT.

    Args:
        sdf: int8 SDF array, C-contiguous, shape (nx, ny, nz)
        kernel: int8 quantized convolution kernel, shape (2r+1, 2r+1, 2r+1)
            Typically from compute_butterworth_kernel()['int8']
        n_threads: OpenMP threads (0 = auto-detect)

    Returns:
        np.ndarray[int8]: filtered SDF, same shape as input, C-contiguous
    """
    cdef int nx = sdf.shape[0]
    cdef int ny = sdf.shape[1]
    cdef int nz = sdf.shape[2]
    cdef int kx = kernel.shape[0]
    cdef int ky = kernel.shape[1]
    cdef int kz = kernel.shape[2]
    cdef int rx = kx // 2
    cdef int ry = ky // 2
    cdef int rz = kz // 2

    out_arr = np.zeros((nx, ny, nz), dtype=np.int8)
    cdef signed char[:, :, ::1] out = out_arr

    cdef int i, j, k, di, dj, dk
    cdef int si, sj, sk
    cdef short acc
    cdef int val

    if n_threads <= 0:
        n_threads = _get_optimal_threads(nx)

    with nogil:
        for i in prange(nx, num_threads=n_threads, schedule='static'):
            for j in range(ny):
                for k in range(nz):
                    acc = 0
                    for di in range(kx):
                        si = i + di - rx
                        if si < 0 or si >= nx:
                            continue
                        for dj in range(ky):
                            sj = j + dj - ry
                            if sj < 0 or sj >= ny:
                                continue
                            for dk in range(kz):
                                sk = k + dk - rz
                                if sk < 0 or sk >= nz:
                                    continue
                                acc = acc + <short>(
                                    <short>sdf[si, sj, sk] *
                                    <short>kernel[di, dj, dk])
                    # Clip to int8 range
                    val = <int>acc
                    if val > 127:
                        val = 127
                    elif val < -128:
                        val = -128
                    out[i, j, k] = <signed char>val

    return out_arr


# ---------------------------------------------------------------------------
# Fused streaming: convolution + marching cubes in one Z-sweep
# ---------------------------------------------------------------------------

# Edge endpoint vertex indices (v0, v1) for each of 12 MC edges
cdef int _EDGE_V0[12]
cdef int _EDGE_V1[12]
# Edge 0: v0-v1, 1: v1-v2, 2: v2-v3, 3: v3-v0
# Edge 4: v4-v5, 5: v5-v6, 6: v6-v7, 7: v7-v4
# Edge 8: v0-v4, 9: v1-v5, 10: v2-v6, 11: v3-v7
_EDGE_V0[0]=0; _EDGE_V1[0]=1;  _EDGE_V0[1]=1; _EDGE_V1[1]=2
_EDGE_V0[2]=2; _EDGE_V1[2]=3;  _EDGE_V0[3]=3; _EDGE_V1[3]=0
_EDGE_V0[4]=4; _EDGE_V1[4]=5;  _EDGE_V0[5]=5; _EDGE_V1[5]=6
_EDGE_V0[6]=6; _EDGE_V1[6]=7;  _EDGE_V0[7]=7; _EDGE_V1[7]=4
_EDGE_V0[8]=0; _EDGE_V1[8]=4;  _EDGE_V0[9]=1; _EDGE_V1[9]=5
_EDGE_V0[10]=2; _EDGE_V1[10]=6; _EDGE_V0[11]=3; _EDGE_V1[11]=7

# Vertex offsets (di, dj, dk) for each of the 8 cube vertices
cdef int _VTX_DI[8]
cdef int _VTX_DJ[8]
cdef int _VTX_DK[8]
# v0=(0,0,0) v1=(1,0,0) v2=(1,1,0) v3=(0,1,0)
# v4=(0,0,1) v5=(1,0,1) v6=(1,1,1) v7=(0,1,1)
_VTX_DI[0]=0; _VTX_DJ[0]=0; _VTX_DK[0]=0
_VTX_DI[1]=1; _VTX_DJ[1]=0; _VTX_DK[1]=0
_VTX_DI[2]=1; _VTX_DJ[2]=1; _VTX_DK[2]=0
_VTX_DI[3]=0; _VTX_DJ[3]=1; _VTX_DK[3]=0
_VTX_DI[4]=0; _VTX_DJ[4]=0; _VTX_DK[4]=1
_VTX_DI[5]=1; _VTX_DJ[5]=0; _VTX_DK[5]=1
_VTX_DI[6]=1; _VTX_DJ[6]=1; _VTX_DK[6]=1
_VTX_DI[7]=0; _VTX_DJ[7]=1; _VTX_DK[7]=1

# Face-layer edge dispatch table: maps MC edge index (0-11) to
# layer id (0=lax,1=lay,2=lbx,3=lby,4=zed) and (di,dj) offsets.
# Replaces 36-branch if/elif cascade with indexed pointer access.
cdef int _EDGE_LAYER[12]
cdef int _EDGE_DI_OFF[12]
cdef int _EDGE_DJ_OFF[12]
# e0: lax[i,j]    e1: lay[i+1,j]  e2: lax[i,j+1]  e3: lay[i,j]
_EDGE_LAYER[0]=0; _EDGE_DI_OFF[0]=0; _EDGE_DJ_OFF[0]=0
_EDGE_LAYER[1]=1; _EDGE_DI_OFF[1]=1; _EDGE_DJ_OFF[1]=0
_EDGE_LAYER[2]=0; _EDGE_DI_OFF[2]=0; _EDGE_DJ_OFF[2]=1
_EDGE_LAYER[3]=1; _EDGE_DI_OFF[3]=0; _EDGE_DJ_OFF[3]=0
# e4: lbx[i,j]    e5: lby[i+1,j]  e6: lbx[i,j+1]  e7: lby[i,j]
_EDGE_LAYER[4]=2; _EDGE_DI_OFF[4]=0; _EDGE_DJ_OFF[4]=0
_EDGE_LAYER[5]=3; _EDGE_DI_OFF[5]=1; _EDGE_DJ_OFF[5]=0
_EDGE_LAYER[6]=2; _EDGE_DI_OFF[6]=0; _EDGE_DJ_OFF[6]=1
_EDGE_LAYER[7]=3; _EDGE_DI_OFF[7]=0; _EDGE_DJ_OFF[7]=0
# e8: zed[i,j]    e9: zed[i+1,j]  e10: zed[i+1,j+1]  e11: zed[i,j+1]
_EDGE_LAYER[8]=4;  _EDGE_DI_OFF[8]=0;  _EDGE_DJ_OFF[8]=0
_EDGE_LAYER[9]=4;  _EDGE_DI_OFF[9]=1;  _EDGE_DJ_OFF[9]=0
_EDGE_LAYER[10]=4; _EDGE_DI_OFF[10]=1; _EDGE_DJ_OFF[10]=1
_EDGE_LAYER[11]=4; _EDGE_DI_OFF[11]=0; _EDGE_DJ_OFF[11]=1


def streaming_mc_mesh(
    const signed char[:, :, ::1] sdf,
    float vsx, float vsy, float vsz,
    float ox=0.0, float oy=0.0, float oz=0.0,
    float isovalue=0.0,
):
    """Streaming marching cubes on int8 SDF — returns vertices and faces.

    Processes the SDF volume one Z-slice pair at a time. No VTK or PyVista
    dependency. Returns numpy arrays suitable for constructing a PolyData mesh.

    Args:
        sdf: int8 SDF array, C-contiguous, shape (nx, ny, nz)
        vsx, vsy, vsz: voxel spacing per axis
        ox, oy, oz: origin offset
        isovalue: isosurface level (typically 0.0)

    Returns:
        tuple (vertices, faces):
            vertices: float32 array (N, 3)
            faces: int32 array (M, 3) — triangle vertex indices
    """
    from voxelcad._kernels._mc_tables import EDGE_TABLE, TRI_TABLE

    cdef int nx = sdf.shape[0]
    cdef int ny = sdf.shape[1]
    cdef int nz = sdf.shape[2]

    # Edge and triangle tables
    cdef unsigned short[::1] edge_tbl = EDGE_TABLE
    cdef signed char[:, ::1] tri_tbl = TRI_TABLE

    # Pre-allocate output (conservative: ~4 tris per surface cell)
    cdef int est_tris = 4 * (nx * ny + ny * nz + nx * nz)
    verts_list = np.empty((est_tris * 3, 3), dtype=np.float32)
    faces_list = np.empty((est_tris, 3), dtype=np.int32)
    cdef float[:, ::1] verts = verts_list
    cdef int[:, ::1] faces = faces_list
    cdef int n_verts = 0
    cdef int n_faces = 0

    cdef int i, j, k, e, t_idx
    cdef int cube_idx
    cdef unsigned short edges
    cdef float vals[8]
    cdef float vert_coords[12][3]
    cdef float v0_val, v1_val, t_interp
    cdef int v0_idx, v1_idx
    cdef int tri_edge[3]

    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                # Read 8 corner values
                vals[0] = <float>sdf[i, j, k]
                vals[1] = <float>sdf[i+1, j, k]
                vals[2] = <float>sdf[i+1, j+1, k]
                vals[3] = <float>sdf[i, j+1, k]
                vals[4] = <float>sdf[i, j, k+1]
                vals[5] = <float>sdf[i+1, j, k+1]
                vals[6] = <float>sdf[i+1, j+1, k+1]
                vals[7] = <float>sdf[i, j+1, k+1]

                # Compute cube index
                cube_idx = 0
                if vals[0] < isovalue: cube_idx |= 1
                if vals[1] < isovalue: cube_idx |= 2
                if vals[2] < isovalue: cube_idx |= 4
                if vals[3] < isovalue: cube_idx |= 8
                if vals[4] < isovalue: cube_idx |= 16
                if vals[5] < isovalue: cube_idx |= 32
                if vals[6] < isovalue: cube_idx |= 64
                if vals[7] < isovalue: cube_idx |= 128

                edges = edge_tbl[cube_idx]
                if edges == 0:
                    continue

                # Interpolate vertices on intersected edges
                for e in range(12):
                    if not (edges & (1 << e)):
                        continue
                    v0_idx = _EDGE_V0[e]
                    v1_idx = _EDGE_V1[e]
                    v0_val = vals[v0_idx]
                    v1_val = vals[v1_idx]
                    if v0_val == v1_val:
                        t_interp = 0.5
                    else:
                        t_interp = (isovalue - v0_val) / (v1_val - v0_val)
                    vert_coords[e][0] = ox + vsx * (
                        <float>(i + _VTX_DI[v0_idx]) +
                        t_interp * <float>(_VTX_DI[v1_idx] - _VTX_DI[v0_idx]))
                    vert_coords[e][1] = oy + vsy * (
                        <float>(j + _VTX_DJ[v0_idx]) +
                        t_interp * <float>(_VTX_DJ[v1_idx] - _VTX_DJ[v0_idx]))
                    vert_coords[e][2] = oz + vsz * (
                        <float>(k + _VTX_DK[v0_idx]) +
                        t_interp * <float>(_VTX_DK[v1_idx] - _VTX_DK[v0_idx]))

                # Emit triangles
                t_idx = 0
                while tri_tbl[cube_idx, t_idx] != -1:
                    # Grow arrays if needed
                    if n_verts + 3 > verts_list.shape[0]:
                        new_size = verts_list.shape[0] * 2
                        verts_list.resize((new_size, 3), refcheck=False)
                        verts = verts_list
                    if n_faces + 1 > faces_list.shape[0]:
                        new_size = faces_list.shape[0] * 2
                        faces_list.resize((new_size, 3), refcheck=False)
                        faces = faces_list

                    for e in range(3):
                        tri_edge[e] = tri_tbl[cube_idx, t_idx + e]
                        verts[n_verts + e, 0] = vert_coords[tri_edge[e]][0]
                        verts[n_verts + e, 1] = vert_coords[tri_edge[e]][1]
                        verts[n_verts + e, 2] = vert_coords[tri_edge[e]][2]

                    faces[n_faces, 0] = n_verts
                    faces[n_faces, 1] = n_verts + 1
                    faces[n_faces, 2] = n_verts + 2
                    n_verts += 3
                    n_faces += 1
                    t_idx += 3

    return verts_list[:n_verts].copy(), faces_list[:n_faces].copy()


def streaming_mc_stl(
    const signed char[:, :, ::1] sdf,
    float vsx, float vsy, float vsz,
    str filename,
    float ox=0.0, float oy=0.0, float oz=0.0,
    float isovalue=0.0,
):
    """Streaming marching cubes — writes binary STL directly to disk.

    Processes the SDF volume one cell at a time. Triangles are written
    immediately to the output file. The full mesh never exists in memory.

    Binary STL format: 80-byte header, 4-byte triangle count,
    then 50 bytes per triangle (normal + 3 vertices + attribute).

    Args:
        sdf: int8 SDF array, C-contiguous, shape (nx, ny, nz)
        vsx, vsy, vsz: voxel spacing per axis
        filename: output STL file path
        ox, oy, oz: origin offset
        isovalue: isosurface level (typically 0.0)

    Returns:
        int: number of triangles written
    """
    import struct
    from voxelcad._kernels._mc_tables import EDGE_TABLE, TRI_TABLE

    cdef int nx = sdf.shape[0]
    cdef int ny = sdf.shape[1]
    cdef int nz = sdf.shape[2]

    cdef unsigned short[::1] edge_tbl = EDGE_TABLE
    cdef signed char[:, ::1] tri_tbl = TRI_TABLE

    cdef int i, j, k, e, t_idx
    cdef int cube_idx
    cdef unsigned short edges
    cdef float vals[8]
    cdef float vert_coords[12][3]
    cdef float v0_val, v1_val, t_interp
    cdef int v0_idx, v1_idx
    cdef int tri_count = 0

    f = open(filename, 'wb')
    # Header (80 bytes) + placeholder triangle count (4 bytes)
    f.write(b'\x00' * 80)
    f.write(struct.pack('<I', 0))

    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                vals[0] = <float>sdf[i, j, k]
                vals[1] = <float>sdf[i+1, j, k]
                vals[2] = <float>sdf[i+1, j+1, k]
                vals[3] = <float>sdf[i, j+1, k]
                vals[4] = <float>sdf[i, j, k+1]
                vals[5] = <float>sdf[i+1, j, k+1]
                vals[6] = <float>sdf[i+1, j+1, k+1]
                vals[7] = <float>sdf[i, j+1, k+1]

                cube_idx = 0
                if vals[0] < isovalue: cube_idx |= 1
                if vals[1] < isovalue: cube_idx |= 2
                if vals[2] < isovalue: cube_idx |= 4
                if vals[3] < isovalue: cube_idx |= 8
                if vals[4] < isovalue: cube_idx |= 16
                if vals[5] < isovalue: cube_idx |= 32
                if vals[6] < isovalue: cube_idx |= 64
                if vals[7] < isovalue: cube_idx |= 128

                edges = edge_tbl[cube_idx]
                if edges == 0:
                    continue

                for e in range(12):
                    if not (edges & (1 << e)):
                        continue
                    v0_idx = _EDGE_V0[e]
                    v1_idx = _EDGE_V1[e]
                    v0_val = vals[v0_idx]
                    v1_val = vals[v1_idx]
                    if v0_val == v1_val:
                        t_interp = 0.5
                    else:
                        t_interp = (isovalue - v0_val) / (v1_val - v0_val)
                    vert_coords[e][0] = ox + vsx * (
                        <float>(i + _VTX_DI[v0_idx]) +
                        t_interp * <float>(_VTX_DI[v1_idx] - _VTX_DI[v0_idx]))
                    vert_coords[e][1] = oy + vsy * (
                        <float>(j + _VTX_DJ[v0_idx]) +
                        t_interp * <float>(_VTX_DJ[v1_idx] - _VTX_DJ[v0_idx]))
                    vert_coords[e][2] = oz + vsz * (
                        <float>(k + _VTX_DK[v0_idx]) +
                        t_interp * <float>(_VTX_DK[v1_idx] - _VTX_DK[v0_idx]))

                t_idx = 0
                while tri_tbl[cube_idx, t_idx] != -1:
                    # Normal = (0,0,0) placeholder — slicer computes from verts
                    f.write(struct.pack('<fff', 0.0, 0.0, 0.0))
                    for e in range(3):
                        edge_idx = tri_tbl[cube_idx, t_idx + e]
                        f.write(struct.pack('<fff',
                            vert_coords[edge_idx][0],
                            vert_coords[edge_idx][1],
                            vert_coords[edge_idx][2]))
                    f.write(struct.pack('<H', 0))  # attribute byte count
                    tri_count += 1
                    t_idx += 3

    # Seek back and write triangle count
    f.seek(80)
    f.write(struct.pack('<I', tri_count))
    f.close()

    return tri_count


def sweep_mc_mesh(
    const signed char[:, :, ::1] sdf,
    float vsx, float vsy, float vsz,
    float ox=0.0, float oy=0.0, float oz=0.0,
    float isovalue=0.0,
):
    """Sweep-plane marching cubes with face-layer vertex dedup.

    Based on Lorensen & Cline's Marching Cubes algorithm [1] with
    sweep-plane vertex sharing inspired by scikit-image's Lewiner MC
    implementation [2,3]. Each edge vertex is computed once and shared
    by all adjacent cells via face-layer index arrays, producing
    manifold output by construction without post-MC dedup.

    Two layers of edge-to-vertex-index arrays (bottom/top z-face) are
    swapped per Z-slice advance. Memory: ~5 * nx * ny * 4 bytes.

    References:
        [1] Lorensen & Cline, "Marching Cubes: A High Resolution 3D
            Surface Construction Algorithm", SIGGRAPH 1987.
        [2] Lewiner et al., "Efficient Implementation of Marching Cubes'
            Cases with Topological Guarantees", J. Graphics Tools 2003.
        [3] scikit-image _marching_cubes_lewiner_cy.pyx — face-layer
            edge caching pattern for O(1) vertex sharing.

    Args:
        sdf: int8 SDF array, C-contiguous, shape (nx, ny, nz)
        vsx, vsy, vsz: voxel spacing per axis
        ox, oy, oz: origin offset
        isovalue: isosurface level (typically 0.0)

    Returns:
        tuple (vertices, faces):
            vertices: float32 array (N, 3) — deduplicated
            faces: int32 array (M, 3) — triangle vertex indices
    """
    from voxelcad._kernels import _mc_tables_lewiner as _lew_mod

    cdef int nx = sdf.shape[0]
    cdef int ny = sdf.shape[1]
    cdef int nz = sdf.shape[2]

    # Lewiner MC lookup tables
    cdef const signed char[:, ::1] lew_cases = _lew_mod.cases
    cdef const signed char[:, ::1] lew_t1 = _lew_mod.tiling1
    cdef const signed char[:, ::1] lew_t2 = _lew_mod.tiling2
    cdef const signed char[:, ::1] lew_t3_1 = _lew_mod.tiling3_1
    cdef const signed char[:, ::1] lew_t3_2 = _lew_mod.tiling3_2
    cdef const signed char[:, ::1] lew_t4_1 = _lew_mod.tiling4_1
    cdef const signed char[:, ::1] lew_t4_2 = _lew_mod.tiling4_2
    cdef const signed char[:, ::1] lew_t5 = _lew_mod.tiling5
    cdef const signed char[:, ::1] lew_t6_11 = _lew_mod.tiling6_1_1
    cdef const signed char[:, ::1] lew_t6_12 = _lew_mod.tiling6_1_2
    cdef const signed char[:, ::1] lew_t6_2 = _lew_mod.tiling6_2
    cdef const signed char[:, ::1] lew_t7_1 = _lew_mod.tiling7_1
    cdef const signed char[:, :, ::1] lew_t7_2 = _lew_mod.tiling7_2
    cdef const signed char[:, :, ::1] lew_t7_3 = _lew_mod.tiling7_3
    cdef const signed char[:, ::1] lew_t7_41 = _lew_mod.tiling7_4_1
    cdef const signed char[:, ::1] lew_t7_42 = _lew_mod.tiling7_4_2
    cdef const signed char[:, ::1] lew_t8 = _lew_mod.tiling8
    cdef const signed char[:, ::1] lew_t9 = _lew_mod.tiling9
    cdef const signed char[:, ::1] lew_t10_11 = _lew_mod.tiling10_1_1
    cdef const signed char[:, ::1] lew_t10_11p = _lew_mod.tiling10_1_1_
    cdef const signed char[:, ::1] lew_t10_12 = _lew_mod.tiling10_1_2
    cdef const signed char[:, ::1] lew_t10_2 = _lew_mod.tiling10_2
    cdef const signed char[:, ::1] lew_t10_2p = _lew_mod.tiling10_2_
    cdef const signed char[:, ::1] lew_t11 = _lew_mod.tiling11
    cdef const signed char[:, ::1] lew_t12_11 = _lew_mod.tiling12_1_1
    cdef const signed char[:, ::1] lew_t12_11p = _lew_mod.tiling12_1_1_
    cdef const signed char[:, ::1] lew_t12_12 = _lew_mod.tiling12_1_2
    cdef const signed char[:, ::1] lew_t12_2 = _lew_mod.tiling12_2
    cdef const signed char[:, ::1] lew_t12_2p = _lew_mod.tiling12_2_
    cdef const signed char[:, ::1] lew_t13_1 = _lew_mod.tiling13_1
    cdef const signed char[:, ::1] lew_t13_1p = _lew_mod.tiling13_1_
    cdef const signed char[:, :, ::1] lew_t13_2 = _lew_mod.tiling13_2
    cdef const signed char[:, :, ::1] lew_t13_2p = _lew_mod.tiling13_2_
    cdef const signed char[:, :, ::1] lew_t13_3 = _lew_mod.tiling13_3
    cdef const signed char[:, :, ::1] lew_t13_3p = _lew_mod.tiling13_3_
    cdef const signed char[:, :, ::1] lew_t13_4 = _lew_mod.tiling13_4
    cdef const signed char[:, :, ::1] lew_t13_51 = _lew_mod.tiling13_5_1
    cdef const signed char[:, :, ::1] lew_t13_52 = _lew_mod.tiling13_5_2
    cdef const signed char[:, ::1] lew_t14 = _lew_mod.tiling14
    cdef const signed char[::1] lew_test3 = _lew_mod.test3
    cdef const signed char[::1] lew_test4 = _lew_mod.test4
    cdef const signed char[:, ::1] lew_test6 = _lew_mod.test6
    cdef const signed char[:, ::1] lew_test7 = _lew_mod.test7
    cdef const signed char[:, ::1] lew_test10 = _lew_mod.test10
    cdef const signed char[:, ::1] lew_test12 = _lew_mod.test12
    cdef const signed char[:, ::1] lew_test13 = _lew_mod.test13
    cdef const signed char[::1] lew_subconfig13 = _lew_mod.subconfig13

    # Face-layer arrays for vertex dedup.
    # x-edges at z=k: between (i,j,k) and (i+1,j,k), shape (nx-1, ny)
    # y-edges at z=k: between (i,j,k) and (i,j+1,k), shape (nx, ny-1)
    # z-edges:        between (i,j,k) and (i,j,k+1), shape (nx, ny)
    layer_a_x_np = np.full((nx - 1, ny), -1, dtype=np.int32)
    layer_a_y_np = np.full((nx, ny - 1), -1, dtype=np.int32)
    layer_b_x_np = np.full((nx - 1, ny), -1, dtype=np.int32)
    layer_b_y_np = np.full((nx, ny - 1), -1, dtype=np.int32)
    z_edges_np = np.full((nx, ny), -1, dtype=np.int32)

    cdef int[:, ::1] lax = layer_a_x_np
    cdef int[:, ::1] lay = layer_a_y_np
    cdef int[:, ::1] lbx = layer_b_x_np
    cdef int[:, ::1] lby = layer_b_y_np
    cdef int[:, ::1] zed = z_edges_np

    # Pre-allocate output (conservative estimate)
    cdef int est_tris = max(4 * (nx * ny + ny * nz + nx * nz), 64)
    verts_np = np.empty((est_tris, 3), dtype=np.float32)
    faces_np = np.empty((est_tris, 3), dtype=np.int32)
    cdef float[:, ::1] verts = verts_np
    cdef int[:, ::1] faces = faces_np
    cdef int n_verts = 0
    cdef int n_faces = 0

    cdef int i, j, k, e, t_idx
    cdef int cube_idx, vid
    cdef unsigned short edges
    cdef float vals[8]
    cdef int edge_vids[13]  # 0-11 = edges, 12 = center vertex
    cdef float v0_val, v1_val, t_interp
    cdef int v0_idx, v1_idx
    # Lewiner dispatch variables
    cdef int case_id, lew_config, lew_subconfig
    cdef signed char edge_buf[36]
    cdef int n_edges, need_edge12
    cdef float w_e, wsum

    for k in range(nz - 1):
        # Reset top-face layers and z-edges for this slice
        layer_b_x_np[:] = -1
        layer_b_y_np[:] = -1
        z_edges_np[:] = -1
        lbx = layer_b_x_np
        lby = layer_b_y_np
        zed = z_edges_np

        for j in range(ny - 1):
            for i in range(nx - 1):
                # Read 8 corner values
                vals[0] = <float>sdf[i, j, k]
                vals[1] = <float>sdf[i+1, j, k]
                vals[2] = <float>sdf[i+1, j+1, k]
                vals[3] = <float>sdf[i, j+1, k]
                vals[4] = <float>sdf[i, j, k+1]
                vals[5] = <float>sdf[i+1, j, k+1]
                vals[6] = <float>sdf[i+1, j+1, k+1]
                vals[7] = <float>sdf[i, j+1, k+1]

                # Lewiner convention: val > isovalue → bit set (inside).
                # Tables produce inward normals; emission swaps e1/e2
                # to get outward normals.
                cube_idx = 0
                if vals[0] > isovalue: cube_idx |= 1
                if vals[1] > isovalue: cube_idx |= 2
                if vals[2] > isovalue: cube_idx |= 4
                if vals[3] > isovalue: cube_idx |= 8
                if vals[4] > isovalue: cube_idx |= 16
                if vals[5] > isovalue: cube_idx |= 32
                if vals[6] > isovalue: cube_idx |= 64
                if vals[7] > isovalue: cube_idx |= 128

                # Lewiner case dispatch
                case_id = lew_cases[cube_idx, 0]
                if case_id == 0:
                    continue
                lew_config = lew_cases[cube_idx, 1]
                lew_subconfig = 0
                n_edges = 0

                if case_id == 1:
                    n_edges = _lew_copy_2d(lew_t1, lew_config, 1, edge_buf)
                elif case_id == 2:
                    n_edges = _lew_copy_2d(lew_t2, lew_config, 2, edge_buf)
                elif case_id == 3:
                    if _lew_test_face(vals, lew_test3[lew_config]):
                        n_edges = _lew_copy_2d(lew_t3_2, lew_config, 4, edge_buf)
                    else:
                        n_edges = _lew_copy_2d(lew_t3_1, lew_config, 2, edge_buf)
                elif case_id == 4:
                    if _lew_test_internal(vals, 4, lew_config, 0,
                                          lew_test4[lew_config],
                                          lew_test6, lew_test7, lew_test12, lew_t13_51):
                        n_edges = _lew_copy_2d(lew_t4_1, lew_config, 2, edge_buf)
                    else:
                        n_edges = _lew_copy_2d(lew_t4_2, lew_config, 6, edge_buf)
                elif case_id == 5:
                    n_edges = _lew_copy_2d(lew_t5, lew_config, 3, edge_buf)
                elif case_id == 6:
                    if _lew_test_face(vals, lew_test6[lew_config, 0]):
                        n_edges = _lew_copy_2d(lew_t6_2, lew_config, 5, edge_buf)
                    else:
                        if _lew_test_internal(vals, 6, lew_config, 0,
                                              lew_test6[lew_config, 1],
                                              lew_test6, lew_test7, lew_test12, lew_t13_51):
                            n_edges = _lew_copy_2d(lew_t6_11, lew_config, 3, edge_buf)
                        else:
                            n_edges = _lew_copy_2d(lew_t6_12, lew_config, 9, edge_buf)
                elif case_id == 7:
                    lew_subconfig = 0
                    if _lew_test_face(vals, lew_test7[lew_config, 0]):
                        lew_subconfig += 1
                    if _lew_test_face(vals, lew_test7[lew_config, 1]):
                        lew_subconfig += 2
                    if _lew_test_face(vals, lew_test7[lew_config, 2]):
                        lew_subconfig += 4
                    if lew_subconfig == 0:
                        n_edges = _lew_copy_2d(lew_t7_1, lew_config, 3, edge_buf)
                    elif lew_subconfig == 1:
                        n_edges = _lew_copy_3d(lew_t7_2, lew_config, 0, 5, edge_buf)
                    elif lew_subconfig == 2:
                        n_edges = _lew_copy_3d(lew_t7_2, lew_config, 1, 5, edge_buf)
                    elif lew_subconfig == 3:
                        n_edges = _lew_copy_3d(lew_t7_3, lew_config, 0, 9, edge_buf)
                    elif lew_subconfig == 4:
                        n_edges = _lew_copy_3d(lew_t7_2, lew_config, 2, 5, edge_buf)
                    elif lew_subconfig == 5:
                        n_edges = _lew_copy_3d(lew_t7_3, lew_config, 1, 9, edge_buf)
                    elif lew_subconfig == 6:
                        n_edges = _lew_copy_3d(lew_t7_3, lew_config, 2, 9, edge_buf)
                    elif lew_subconfig == 7:
                        if _lew_test_internal(vals, 7, lew_config, lew_subconfig,
                                              lew_test7[lew_config, 3],
                                              lew_test6, lew_test7, lew_test12, lew_t13_51):
                            n_edges = _lew_copy_2d(lew_t7_42, lew_config, 9, edge_buf)
                        else:
                            n_edges = _lew_copy_2d(lew_t7_41, lew_config, 5, edge_buf)
                elif case_id == 8:
                    n_edges = _lew_copy_2d(lew_t8, lew_config, 2, edge_buf)
                elif case_id == 9:
                    n_edges = _lew_copy_2d(lew_t9, lew_config, 4, edge_buf)
                elif case_id == 10:
                    if _lew_test_face(vals, lew_test10[lew_config, 0]):
                        if _lew_test_face(vals, lew_test10[lew_config, 1]):
                            n_edges = _lew_copy_2d(lew_t10_11p, lew_config, 4, edge_buf)
                        else:
                            n_edges = _lew_copy_2d(lew_t10_2, lew_config, 8, edge_buf)
                    else:
                        if _lew_test_face(vals, lew_test10[lew_config, 1]):
                            n_edges = _lew_copy_2d(lew_t10_2p, lew_config, 8, edge_buf)
                        else:
                            if _lew_test_internal(vals, 10, lew_config, 0,
                                                  lew_test10[lew_config, 2],
                                                  lew_test6, lew_test7, lew_test12, lew_t13_51):
                                n_edges = _lew_copy_2d(lew_t10_11, lew_config, 4, edge_buf)
                            else:
                                n_edges = _lew_copy_2d(lew_t10_12, lew_config, 8, edge_buf)
                elif case_id == 11:
                    n_edges = _lew_copy_2d(lew_t11, lew_config, 4, edge_buf)
                elif case_id == 12:
                    if _lew_test_face(vals, lew_test12[lew_config, 0]):
                        if _lew_test_face(vals, lew_test12[lew_config, 1]):
                            n_edges = _lew_copy_2d(lew_t12_11p, lew_config, 4, edge_buf)
                        else:
                            n_edges = _lew_copy_2d(lew_t12_2, lew_config, 8, edge_buf)
                    else:
                        if _lew_test_face(vals, lew_test12[lew_config, 1]):
                            n_edges = _lew_copy_2d(lew_t12_2p, lew_config, 8, edge_buf)
                        else:
                            if _lew_test_internal(vals, 12, lew_config, 0,
                                                  lew_test12[lew_config, 2],
                                                  lew_test6, lew_test7, lew_test12, lew_t13_51):
                                n_edges = _lew_copy_2d(lew_t12_11, lew_config, 4, edge_buf)
                            else:
                                n_edges = _lew_copy_2d(lew_t12_12, lew_config, 8, edge_buf)
                elif case_id == 13:
                    lew_subconfig = 0
                    if _lew_test_face(vals, lew_test13[lew_config, 0]):
                        lew_subconfig |= 1
                    if _lew_test_face(vals, lew_test13[lew_config, 1]):
                        lew_subconfig |= 2
                    if _lew_test_face(vals, lew_test13[lew_config, 2]):
                        lew_subconfig |= 4
                    if _lew_test_face(vals, lew_test13[lew_config, 3]):
                        lew_subconfig |= 8
                    if _lew_test_face(vals, lew_test13[lew_config, 4]):
                        lew_subconfig |= 16
                    if _lew_test_face(vals, lew_test13[lew_config, 5]):
                        lew_subconfig |= 32
                    lew_subconfig = lew_subconfig13[lew_subconfig]
                    if lew_subconfig == 0:
                        n_edges = _lew_copy_2d(lew_t13_1, lew_config, 4, edge_buf)
                    elif 1 <= lew_subconfig <= 6:
                        n_edges = _lew_copy_3d(lew_t13_2, lew_config, lew_subconfig - 1, 6, edge_buf)
                    elif 7 <= lew_subconfig <= 18:
                        n_edges = _lew_copy_3d(lew_t13_3, lew_config, lew_subconfig - 7, 10, edge_buf)
                    elif 19 <= lew_subconfig <= 22:
                        n_edges = _lew_copy_3d(lew_t13_4, lew_config, lew_subconfig - 19, 12, edge_buf)
                    elif 23 <= lew_subconfig <= 26:
                        if _lew_test_internal(vals, 13, lew_config,
                                              lew_subconfig - 23,
                                              lew_test13[lew_config, 6],
                                              lew_test6, lew_test7, lew_test12, lew_t13_51):
                            n_edges = _lew_copy_3d(lew_t13_51, lew_config, lew_subconfig - 23, 6, edge_buf)
                        else:
                            n_edges = _lew_copy_3d(lew_t13_52, lew_config, lew_subconfig - 23, 10, edge_buf)
                    elif 27 <= lew_subconfig <= 38:
                        n_edges = _lew_copy_3d(lew_t13_3p, lew_config, lew_subconfig - 27, 10, edge_buf)
                    elif 39 <= lew_subconfig <= 44:
                        n_edges = _lew_copy_3d(lew_t13_2p, lew_config, lew_subconfig - 39, 6, edge_buf)
                    elif lew_subconfig == 45:
                        n_edges = _lew_copy_2d(lew_t13_1p, lew_config, 4, edge_buf)
                elif case_id == 14:
                    n_edges = _lew_copy_2d(lew_t14, lew_config, 4, edge_buf)

                if n_edges == 0:
                    continue

                # Build edges_mask from edge_buf; detect edge 12
                edges = 0
                need_edge12 = 0
                for t_idx in range(n_edges):
                    if edge_buf[t_idx] == 12:
                        need_edge12 = 1
                    elif 0 <= edge_buf[t_idx] < 12:
                        edges |= <unsigned short>(1 << edge_buf[t_idx])

                # Look up or create vertex for each active edge
                for e in range(12):
                    if not (edges & (1 << e)):
                        edge_vids[e] = -1
                        continue

                    # Face-layer lookup: which array holds this edge?
                    # e0:  x-edge at (i, j, k)      -> lax[i, j]
                    # e1:  y-edge at (i+1, j, k)    -> lay[i+1, j]
                    # e2:  x-edge at (i, j+1, k)    -> lax[i, j+1]
                    # e3:  y-edge at (i, j, k)      -> lay[i, j]
                    # e4:  x-edge at (i, j, k+1)    -> lbx[i, j]
                    # e5:  y-edge at (i+1, j, k+1)  -> lby[i+1, j]
                    # e6:  x-edge at (i, j+1, k+1)  -> lbx[i, j+1]
                    # e7:  y-edge at (i, j, k+1)    -> lby[i, j]
                    # e8:  z-edge at (i, j)          -> zed[i, j]
                    # e9:  z-edge at (i+1, j)        -> zed[i+1, j]
                    # e10: z-edge at (i+1, j+1)      -> zed[i+1, j+1]
                    # e11: z-edge at (i, j+1)        -> zed[i, j+1]
                    if e == 0: vid = lax[i, j]
                    elif e == 1: vid = lay[i + 1, j]
                    elif e == 2: vid = lax[i, j + 1]
                    elif e == 3: vid = lay[i, j]
                    elif e == 4: vid = lbx[i, j]
                    elif e == 5: vid = lby[i + 1, j]
                    elif e == 6: vid = lbx[i, j + 1]
                    elif e == 7: vid = lby[i, j]
                    elif e == 8: vid = zed[i, j]
                    elif e == 9: vid = zed[i + 1, j]
                    elif e == 10: vid = zed[i + 1, j + 1]
                    else: vid = zed[i, j + 1]

                    if vid >= 0:
                        edge_vids[e] = vid
                        continue

                    # Vertex not yet created — interpolate
                    if n_verts >= verts_np.shape[0]:
                        new_size = verts_np.shape[0] * 2
                        verts_np.resize((new_size, 3), refcheck=False)
                        verts = verts_np

                    v0_idx = _EDGE_V0[e]
                    v1_idx = _EDGE_V1[e]
                    v0_val = vals[v0_idx]
                    v1_val = vals[v1_idx]
                    if v0_val == v1_val:
                        t_interp = 0.5
                    else:
                        t_interp = (isovalue - v0_val) / (v1_val - v0_val)
                    verts[n_verts, 0] = ox + vsx * (
                        <float>(i + _VTX_DI[v0_idx]) +
                        t_interp * <float>(_VTX_DI[v1_idx] - _VTX_DI[v0_idx]))
                    verts[n_verts, 1] = oy + vsy * (
                        <float>(j + _VTX_DJ[v0_idx]) +
                        t_interp * <float>(_VTX_DJ[v1_idx] - _VTX_DJ[v0_idx]))
                    verts[n_verts, 2] = oz + vsz * (
                        <float>(k + _VTX_DK[v0_idx]) +
                        t_interp * <float>(_VTX_DK[v1_idx] - _VTX_DK[v0_idx]))

                    vid = n_verts
                    n_verts += 1
                    edge_vids[e] = vid

                    # Store in face-layer
                    if e == 0: lax[i, j] = vid
                    elif e == 1: lay[i + 1, j] = vid
                    elif e == 2: lax[i, j + 1] = vid
                    elif e == 3: lay[i, j] = vid
                    elif e == 4: lbx[i, j] = vid
                    elif e == 5: lby[i + 1, j] = vid
                    elif e == 6: lbx[i, j + 1] = vid
                    elif e == 7: lby[i, j] = vid
                    elif e == 8: zed[i, j] = vid
                    elif e == 9: zed[i + 1, j] = vid
                    elif e == 10: zed[i + 1, j + 1] = vid
                    else: zed[i, j + 1] = vid

                # Edge 12: center vertex (inverse-distance weighted avg)
                if need_edge12:
                    if n_verts >= verts_np.shape[0]:
                        new_size = verts_np.shape[0] * 2
                        verts_np.resize((new_size, 3), refcheck=False)
                        verts = verts_np
                    wsum = 0.0
                    verts[n_verts, 0] = 0.0
                    verts[n_verts, 1] = 0.0
                    verts[n_verts, 2] = 0.0
                    for e in range(8):
                        w_e = vals[e] - isovalue
                        if w_e < 0:
                            w_e = -w_e
                        w_e = 1.0 / (1e-12 + w_e)
                        verts[n_verts, 0] += w_e * (ox + vsx * <float>(i + _VTX_DI[e]))
                        verts[n_verts, 1] += w_e * (oy + vsy * <float>(j + _VTX_DJ[e]))
                        verts[n_verts, 2] += w_e * (oz + vsz * <float>(k + _VTX_DK[e]))
                        wsum += w_e
                    if wsum > 0.0:
                        verts[n_verts, 0] /= wsum
                        verts[n_verts, 1] /= wsum
                        verts[n_verts, 2] /= wsum
                    edge_vids[12] = n_verts
                    n_verts += 1

                # Emit triangles: swap indices 1/2 to reverse Lewiner's inward
                # winding to outward normals.
                t_idx = 0
                while t_idx < n_edges:
                    if n_faces >= faces_np.shape[0]:
                        new_size = faces_np.shape[0] * 2
                        faces_np.resize((new_size, 3), refcheck=False)
                        faces = faces_np

                    faces[n_faces, 0] = edge_vids[edge_buf[t_idx]]
                    faces[n_faces, 1] = edge_vids[edge_buf[t_idx + 2]]
                    faces[n_faces, 2] = edge_vids[edge_buf[t_idx + 1]]
                    n_faces += 1
                    t_idx += 3

        # Swap layers: top becomes bottom for next Z-slice
        layer_a_x_np, layer_b_x_np = layer_b_x_np, layer_a_x_np
        layer_a_y_np, layer_b_y_np = layer_b_y_np, layer_a_y_np
        lax = layer_a_x_np
        lay = layer_a_y_np

    return verts_np[:n_verts].copy(), faces_np[:n_faces].copy()


# ---------------------------------------------------------------------------
# Lewiner MC helpers: topologically correct marching cubes dispatch
# Reference: Lewiner et al. 2003; ported from scikit-image implementation.
# ---------------------------------------------------------------------------

DEF LEW_EPS = 1e-12

cdef inline int _lew_test_face(float *v, int face) noexcept nogil:
    """Asymptotic Decider face test for Lewiner MC.

    Tests whether the isosurface topology on a cube face requires
    the surface to cross the face diagonal. face is signed: abs(face)
    selects face 1-6, sign affects degenerate tiebreaker.

    Face-to-corner mapping (matching scikit-image convention):
      face 1: v0 v4 v5 v1    face 2: v1 v5 v6 v2
      face 3: v2 v6 v7 v3    face 4: v3 v7 v4 v0
      face 5: v0 v3 v2 v1    face 6: v4 v7 v6 v5
    """
    cdef int af = face if face > 0 else -face
    cdef float A = 0.0, B = 0.0, C = 0.0, D = 0.0
    cdef float AC_BD

    if af == 1:
        A = v[0]; B = v[4]; C = v[5]; D = v[1]
    elif af == 2:
        A = v[1]; B = v[5]; C = v[6]; D = v[2]
    elif af == 3:
        A = v[2]; B = v[6]; C = v[7]; D = v[3]
    elif af == 4:
        A = v[3]; B = v[7]; C = v[4]; D = v[0]
    elif af == 5:
        A = v[0]; B = v[3]; C = v[2]; D = v[1]
    elif af == 6:
        A = v[4]; B = v[7]; C = v[6]; D = v[5]

    AC_BD = A * C - B * D
    if AC_BD > -LEW_EPS and AC_BD < LEW_EPS:
        return 1 if face >= 0 else 0
    return 1 if face * A * AC_BD >= 0 else 0


cdef inline int _lew_test_internal(
    float *v, int case_id, int config, int subconfig, int s,
    const signed char[:, ::1] test6,
    const signed char[:, ::1] test7,
    const signed char[:, ::1] test12,
    const signed char[:, :, ::1] tiling13_5_1,
) noexcept nogil:
    """Interior test for Lewiner MC ambiguous cases 4,6,7,10,12,13.

    Parametric evaluation along a reference edge, then Asymptotic Decider
    on interpolated corner values. Returns 1 or 0 based on topology.
    """
    cdef float t, At, Bt, Ct, Dt, a, b, AC_BD
    cdef int test_val = 0
    cdef int edge = -1

    At = 0.0; Bt = 0.0; Ct = 0.0; Dt = 0.0

    if case_id == 4 or case_id == 10:
        a = (v[4]-v[0])*(v[6]-v[2]) - (v[7]-v[3])*(v[5]-v[1])
        b = (v[2]*(v[4]-v[0]) + v[0]*(v[6]-v[2])
             - v[1]*(v[7]-v[3]) - v[3]*(v[5]-v[1]))
        t = -b / (2.0 * a + LEW_EPS)
        if t < 0.0 or t > 1.0:
            return 1 if s > 0 else 0
        At = v[0] + (v[4] - v[0]) * t
        Bt = v[3] + (v[7] - v[3]) * t
        Ct = v[2] + (v[6] - v[2]) * t
        Dt = v[1] + (v[5] - v[1]) * t

    elif case_id == 6 or case_id == 7 or case_id == 12 or case_id == 13:
        if case_id == 6:
            edge = test6[config, 2]
        elif case_id == 7:
            edge = test7[config, 4]
        elif case_id == 12:
            edge = test12[config, 3]
        elif case_id == 13:
            edge = tiling13_5_1[config, subconfig, 0]

        # Parametric interpolation along reference edge
        if edge == 0:
            t = v[0]/(v[0]-v[1]+LEW_EPS)
            At=0; Bt=v[3]+(v[2]-v[3])*t; Ct=v[7]+(v[6]-v[7])*t; Dt=v[4]+(v[5]-v[4])*t
        elif edge == 1:
            t = v[1]/(v[1]-v[2]+LEW_EPS)
            At=0; Bt=v[0]+(v[3]-v[0])*t; Ct=v[4]+(v[7]-v[4])*t; Dt=v[5]+(v[6]-v[5])*t
        elif edge == 2:
            t = v[2]/(v[2]-v[3]+LEW_EPS)
            At=0; Bt=v[1]+(v[0]-v[1])*t; Ct=v[5]+(v[4]-v[5])*t; Dt=v[6]+(v[7]-v[6])*t
        elif edge == 3:
            t = v[3]/(v[3]-v[0]+LEW_EPS)
            At=0; Bt=v[2]+(v[1]-v[2])*t; Ct=v[6]+(v[5]-v[6])*t; Dt=v[7]+(v[4]-v[7])*t
        elif edge == 4:
            t = v[4]/(v[4]-v[5]+LEW_EPS)
            At=0; Bt=v[7]+(v[6]-v[7])*t; Ct=v[3]+(v[2]-v[3])*t; Dt=v[0]+(v[1]-v[0])*t
        elif edge == 5:
            t = v[5]/(v[5]-v[6]+LEW_EPS)
            At=0; Bt=v[4]+(v[7]-v[4])*t; Ct=v[0]+(v[3]-v[0])*t; Dt=v[1]+(v[2]-v[1])*t
        elif edge == 6:
            t = v[6]/(v[6]-v[7]+LEW_EPS)
            At=0; Bt=v[5]+(v[4]-v[5])*t; Ct=v[1]+(v[0]-v[1])*t; Dt=v[2]+(v[3]-v[2])*t
        elif edge == 7:
            t = v[7]/(v[7]-v[4]+LEW_EPS)
            At=0; Bt=v[6]+(v[5]-v[6])*t; Ct=v[2]+(v[1]-v[2])*t; Dt=v[3]+(v[0]-v[3])*t
        elif edge == 8:
            t = v[0]/(v[0]-v[4]+LEW_EPS)
            At=0; Bt=v[3]+(v[7]-v[3])*t; Ct=v[2]+(v[6]-v[2])*t; Dt=v[1]+(v[5]-v[1])*t
        elif edge == 9:
            t = v[1]/(v[1]-v[5]+LEW_EPS)
            At=0; Bt=v[0]+(v[4]-v[0])*t; Ct=v[3]+(v[7]-v[3])*t; Dt=v[2]+(v[6]-v[2])*t
        elif edge == 10:
            t = v[2]/(v[2]-v[6]+LEW_EPS)
            At=0; Bt=v[1]+(v[5]-v[1])*t; Ct=v[0]+(v[4]-v[0])*t; Dt=v[3]+(v[7]-v[3])*t
        elif edge == 11:
            t = v[3]/(v[3]-v[7]+LEW_EPS)
            At=0; Bt=v[2]+(v[6]-v[2])*t; Ct=v[1]+(v[5]-v[1])*t; Dt=v[0]+(v[4]-v[0])*t

    # Classify interpolated corners
    if At >= 0: test_val += 1
    if Bt >= 0: test_val += 2
    if Ct >= 0: test_val += 4
    if Dt >= 0: test_val += 8

    # Interpret result: most cases return s>0, some return s<0
    if test_val == 5:
        AC_BD = At * Ct - Bt * Dt
        if AC_BD < LEW_EPS:
            return 1 if s > 0 else 0
        return 1 if s < 0 else 0
    elif test_val == 10:
        AC_BD = At * Ct - Bt * Dt
        if AC_BD >= LEW_EPS:
            return 1 if s > 0 else 0
        return 1 if s < 0 else 0
    elif test_val == 7 or test_val == 11 or test_val == 13 or test_val == 14 or test_val == 15:
        return 1 if s < 0 else 0
    else:
        return 1 if s > 0 else 0


cdef inline int _lew_copy_2d(
    const signed char[:, ::1] tbl, int config, int n_tri,
    signed char *buf,
) noexcept nogil:
    """Copy n_tri*3 edge indices from 2D tiling table to buf. Returns count."""
    cdef int i, n = n_tri * 3
    for i in range(n):
        buf[i] = tbl[config, i]
    return n


cdef inline int _lew_copy_3d(
    const signed char[:, :, ::1] tbl, int config, int sub, int n_tri,
    signed char *buf,
) noexcept nogil:
    """Copy n_tri*3 edge indices from 3D tiling table to buf. Returns count."""
    cdef int i, n = n_tri * 3
    for i in range(n):
        buf[i] = tbl[config, sub, i]
    return n


cdef void* _mc_thread_func(void *arg) noexcept nogil:
    cdef mc_args_t *a = <mc_args_t*>arg
    while True:
        pthread_mutex_lock(&a.mutex)
        while a.mc_go == 0 and a.terminate == 0:
            pthread_cond_wait(&a.slice_ready, &a.mutex)
        if a.terminate != 0:
            pthread_mutex_unlock(&a.mutex)
            return NULL
        a.mc_go = 0
        pthread_mutex_unlock(&a.mutex)
        _mc_process_layer(a)
        pthread_mutex_lock(&a.mutex)
        a.mc_finished = 1
        pthread_cond_signal(&a.mc_done)
        pthread_mutex_unlock(&a.mutex)
    return NULL


cdef void _mc_process_layer(mc_args_t *a) noexcept nogil:
    """Process one Z-layer of marching cubes, emitting triangles to STL buffer.
    Pure C — no Python objects, no GIL. All state via mc_args_t struct.
    """
    cdef int i, j, e, t_idx
    cdef int cube_idx
    cdef unsigned short edges_mask
    cdef float corner_vals[8]
    cdef float vert_coords[13][3]
    cdef float v0_val, v1_val, t_interp
    cdef int v0_idx, v1_idx
    cdef float vx
    # OPT 17: Normal computation variables (only used when compute_normals=1)
    cdef float ux, uy, uz, wx, wy, wz, nn
    cdef signed char edge_buf[36]
    cdef int n_edges, need_edge12
    cdef float w_e, wsum
    cdef int fl_li, fl_off
    cdef float *fl_p
    cdef float *fptr
    cdef unsigned short *up
    cdef int buf_off
    cdef int ei0, ei1, ei2
    cdef int px = a.px
    cdef int py = a.py
    cdef int k = a.k
    cdef signed char *sa = a.slice_a
    cdef signed char *sb = a.slice_b
    cdef signed char *ft = a.ft
    cdef int ft_ncols = a.ft_ncols
    cdef float isovalue = a.isovalue
    cdef float mc_ox = a.mc_ox
    cdef float mc_oy = a.mc_oy
    cdef float mc_oz = a.mc_oz
    cdef float mc_vsx = a.mc_vsx
    cdef float mc_vsy = a.mc_vsy
    cdef float mc_vsz = a.mc_vsz

    cdef signed char *ba = a.band_a
    cdef signed char *bb = a.band_b

    # OPT 18: i-outer, j-inner for cache-friendly face-layer access
    # (j varies fast → stride-3 floats = 12 bytes, vs old stride py*12 bytes)
    for i in range(px - 1):
        for j in range(py - 1):
            # OPT 15: Skip cells where all 8 corners are non-surface
            if (ba[i * py + j] == 0 and ba[(i + 1) * py + j] == 0
                    and ba[i * py + (j + 1)] == 0
                    and ba[(i + 1) * py + (j + 1)] == 0
                    and bb[i * py + j] == 0
                    and bb[(i + 1) * py + j] == 0
                    and bb[i * py + (j + 1)] == 0
                    and bb[(i + 1) * py + (j + 1)] == 0):
                continue

            corner_vals[0] = <float>sa[i * py + j]
            corner_vals[1] = <float>sa[(i + 1) * py + j]
            corner_vals[2] = <float>sa[(i + 1) * py + (j + 1)]
            corner_vals[3] = <float>sa[i * py + (j + 1)]
            corner_vals[4] = <float>sb[i * py + j]
            corner_vals[5] = <float>sb[(i + 1) * py + j]
            corner_vals[6] = <float>sb[(i + 1) * py + (j + 1)]
            corner_vals[7] = <float>sb[i * py + (j + 1)]

            cube_idx = 0
            if corner_vals[0] > isovalue: cube_idx |= 1
            if corner_vals[1] > isovalue: cube_idx |= 2
            if corner_vals[2] > isovalue: cube_idx |= 4
            if corner_vals[3] > isovalue: cube_idx |= 8
            if corner_vals[4] > isovalue: cube_idx |= 16
            if corner_vals[5] > isovalue: cube_idx |= 32
            if corner_vals[6] > isovalue: cube_idx |= 64
            if corner_vals[7] > isovalue: cube_idx |= 128

            n_edges = ft[cube_idx * ft_ncols]
            if n_edges == 0:
                continue

            edges_mask = 0
            need_edge12 = 0
            for t_idx in range(n_edges):
                edge_buf[t_idx] = ft[cube_idx * ft_ncols + 1 + t_idx]
                if edge_buf[t_idx] == 12:
                    need_edge12 = 1
                elif 0 <= edge_buf[t_idx] < 12:
                    edges_mask |= <unsigned short>(1 << edge_buf[t_idx])

            for e in range(12):
                if not (edges_mask & (1 << e)):
                    continue
                fl_li = _EDGE_LAYER[e]
                fl_p = a.layer_ptrs[fl_li]
                fl_off = ((i + _EDGE_DI_OFF[e]) * a.layer_jstride[fl_li]
                          + (j + _EDGE_DJ_OFF[e]) * 3)
                vx = fl_p[fl_off]
                if vx == vx:
                    vert_coords[e][0] = fl_p[fl_off]
                    vert_coords[e][1] = fl_p[fl_off + 1]
                    vert_coords[e][2] = fl_p[fl_off + 2]
                    continue
                v0_idx = _EDGE_V0[e]
                v1_idx = _EDGE_V1[e]
                v0_val = corner_vals[v0_idx]
                v1_val = corner_vals[v1_idx]
                if v0_val == v1_val:
                    t_interp = 0.5
                else:
                    t_interp = (isovalue - v0_val) / (v1_val - v0_val)
                vert_coords[e][0] = mc_ox + mc_vsx * (
                    <float>(i + _VTX_DI[v0_idx]) +
                    t_interp * <float>(_VTX_DI[v1_idx] - _VTX_DI[v0_idx]))
                vert_coords[e][1] = mc_oy + mc_vsy * (
                    <float>(j + _VTX_DJ[v0_idx]) +
                    t_interp * <float>(_VTX_DJ[v1_idx] - _VTX_DJ[v0_idx]))
                vert_coords[e][2] = mc_oz + mc_vsz * (
                    <float>(k + _VTX_DK[v0_idx]) +
                    t_interp * <float>(_VTX_DK[v1_idx] - _VTX_DK[v0_idx]))
                fl_p[fl_off] = vert_coords[e][0]
                fl_p[fl_off + 1] = vert_coords[e][1]
                fl_p[fl_off + 2] = vert_coords[e][2]

            if need_edge12:
                wsum = 0.0
                vert_coords[12][0] = 0.0
                vert_coords[12][1] = 0.0
                vert_coords[12][2] = 0.0
                for e in range(8):
                    w_e = corner_vals[e] - isovalue
                    if w_e < 0:
                        w_e = -w_e
                    w_e = 1.0 / (LEW_EPS + w_e)
                    vert_coords[12][0] += w_e * (mc_ox + mc_vsx * <float>(i + _VTX_DI[e]))
                    vert_coords[12][1] += w_e * (mc_oy + mc_vsy * <float>(j + _VTX_DJ[e]))
                    vert_coords[12][2] += w_e * (mc_oz + mc_vsz * <float>(k + _VTX_DK[e]))
                    wsum += w_e
                if wsum > 0.0:
                    vert_coords[12][0] /= wsum
                    vert_coords[12][1] /= wsum
                    vert_coords[12][2] /= wsum

            t_idx = 0
            while t_idx < n_edges:
                ei0 = edge_buf[t_idx]
                ei1 = edge_buf[t_idx + 2]
                ei2 = edge_buf[t_idx + 1]
                buf_off = a.buf_count * 50
                fptr = <float*>(&a.stl_buf[buf_off])
                # OPT 17: Skip normal computation unless explicitly requested
                if a.compute_normals:
                    ux = vert_coords[ei1][0] - vert_coords[ei0][0]
                    uy = vert_coords[ei1][1] - vert_coords[ei0][1]
                    uz = vert_coords[ei1][2] - vert_coords[ei0][2]
                    wx = vert_coords[ei2][0] - vert_coords[ei0][0]
                    wy = vert_coords[ei2][1] - vert_coords[ei0][1]
                    wz = vert_coords[ei2][2] - vert_coords[ei0][2]
                    fptr[0] = uy * wz - uz * wy
                    fptr[1] = uz * wx - ux * wz
                    fptr[2] = ux * wy - uy * wx
                    nn = sqrt(fptr[0]*fptr[0] + fptr[1]*fptr[1] + fptr[2]*fptr[2])
                    if nn > 0.0:
                        fptr[0] = fptr[0] / nn
                        fptr[1] = fptr[1] / nn
                        fptr[2] = fptr[2] / nn
                else:
                    fptr[0] = 0.0
                    fptr[1] = 0.0
                    fptr[2] = 0.0
                fptr[3] = vert_coords[ei0][0]
                fptr[4] = vert_coords[ei0][1]
                fptr[5] = vert_coords[ei0][2]
                fptr[6] = vert_coords[ei1][0]
                fptr[7] = vert_coords[ei1][1]
                fptr[8] = vert_coords[ei1][2]
                fptr[9] = vert_coords[ei2][0]
                fptr[10] = vert_coords[ei2][1]
                fptr[11] = vert_coords[ei2][2]
                up = <unsigned short*>(&a.stl_buf[buf_off + 48])
                up[0] = 0
                a.buf_count += 1
                a.tri_count += 1
                t_idx += 3
                if a.buf_count == a.BUF_MAX:
                    fwrite(a.stl_buf, 1, a.BUF_MAX * 50, a.fp)
                    a.buf_count = 0


# ---------------------------------------------------------------------------
# OPT 16: Two-pass parallel MC — face-layer dedup preserved
# ---------------------------------------------------------------------------

cdef void _precompute_layer_edges(mc_args_t *a, int compute_bottom) noexcept nogil:
    """Pass 1: Pre-compute all edge vertices for this Z-layer into face-layer arrays.
    Each face-layer position is written by exactly one thread — no races.
    For k>0, bottom-face edges (lax/lay) come from previous layer's swap.
    """
    cdef int i, j, off
    cdef float v0, v1, t, iso
    cdef float ox, oy, oz, vsx, vsy, vsz
    cdef int py_val = a.py
    cdef int px_val = a.px
    cdef int k = a.k
    cdef signed char *sa = a.slice_a
    cdef signed char *sb = a.slice_b
    cdef float *lax_p = a.layer_ptrs[0]
    cdef float *lay_p = a.layer_ptrs[1]
    cdef float *lbx_p = a.layer_ptrs[2]
    cdef float *lby_p = a.layer_ptrs[3]
    cdef float *zed_p = a.layer_ptrs[4]
    cdef int lax_js = a.layer_jstride[0]
    cdef int lay_js = a.layer_jstride[1]
    cdef int lbx_js = a.layer_jstride[2]
    cdef int lby_js = a.layer_jstride[3]
    cdef int zed_js = a.layer_jstride[4]
    cdef int n_t = a.mc_n_threads

    iso = a.isovalue
    ox = a.mc_ox; oy = a.mc_oy; oz = a.mc_oz
    vsx = a.mc_vsx; vsy = a.mc_vsy; vsz = a.mc_vsz

    if compute_bottom:
        # Bottom X-edges: lax[i,j] = edge (i,j,k)-(i+1,j,k) on slice_a
        for i in prange(px_val - 1, num_threads=n_t, nogil=True):
            for j in range(py_val):
                v0 = <float>sa[i * py_val + j]
                v1 = <float>sa[(i + 1) * py_val + j]
                if (v0 > iso) != (v1 > iso):
                    if v0 == v1:
                        t = 0.5
                    else:
                        t = (iso - v0) / (v1 - v0)
                    off = i * lax_js + j * 3
                    lax_p[off] = ox + vsx * (<float>i + t)
                    lax_p[off + 1] = oy + vsy * <float>j
                    lax_p[off + 2] = oz + vsz * <float>k

        # Bottom Y-edges: lay[i,j] = edge (i,j,k)-(i,j+1,k) on slice_a
        for i in prange(px_val, num_threads=n_t, nogil=True):
            for j in range(py_val - 1):
                v0 = <float>sa[i * py_val + j]
                v1 = <float>sa[i * py_val + (j + 1)]
                if (v0 > iso) != (v1 > iso):
                    if v0 == v1:
                        t = 0.5
                    else:
                        t = (iso - v0) / (v1 - v0)
                    off = i * lay_js + j * 3
                    lay_p[off] = ox + vsx * <float>i
                    lay_p[off + 1] = oy + vsy * (<float>j + t)
                    lay_p[off + 2] = oz + vsz * <float>k

    # Top X-edges: lbx[i,j] = edge (i,j,k+1)-(i+1,j,k+1) on slice_b
    for i in prange(px_val - 1, num_threads=n_t, nogil=True):
        for j in range(py_val):
            v0 = <float>sb[i * py_val + j]
            v1 = <float>sb[(i + 1) * py_val + j]
            if (v0 > iso) != (v1 > iso):
                if v0 == v1:
                    t = 0.5
                else:
                    t = (iso - v0) / (v1 - v0)
                off = i * lbx_js + j * 3
                lbx_p[off] = ox + vsx * (<float>i + t)
                lbx_p[off + 1] = oy + vsy * <float>j
                lbx_p[off + 2] = oz + vsz * <float>(k + 1)

    # Top Y-edges: lby[i,j] = edge (i,j,k+1)-(i,j+1,k+1) on slice_b
    for i in prange(px_val, num_threads=n_t, nogil=True):
        for j in range(py_val - 1):
            v0 = <float>sb[i * py_val + j]
            v1 = <float>sb[i * py_val + (j + 1)]
            if (v0 > iso) != (v1 > iso):
                if v0 == v1:
                    t = 0.5
                else:
                    t = (iso - v0) / (v1 - v0)
                off = i * lby_js + j * 3
                lby_p[off] = ox + vsx * <float>i
                lby_p[off + 1] = oy + vsy * (<float>j + t)
                lby_p[off + 2] = oz + vsz * <float>(k + 1)

    # Z-edges: zed[i,j] = edge (i,j,k)-(i,j,k+1) between slices
    for i in prange(px_val, num_threads=n_t, nogil=True):
        for j in range(py_val):
            v0 = <float>sa[i * py_val + j]
            v1 = <float>sb[i * py_val + j]
            if (v0 > iso) != (v1 > iso):
                if v0 == v1:
                    t = 0.5
                else:
                    t = (iso - v0) / (v1 - v0)
                off = i * zed_js + j * 3
                zed_p[off] = ox + vsx * <float>i
                zed_p[off + 1] = oy + vsy * <float>j
                zed_p[off + 2] = oz + vsz * (<float>k + t)


cdef int _mc_emit_cell(mc_args_t *a, int i, int j,
                        unsigned char *buf, int buf_start) noexcept nogil:
    """Process one MC cell: read pre-computed edges from face-layers, emit triangles.
    Returns number of triangles written to buf starting at buf_start * 50.
    Stack-local arrays ensure thread safety when called from prange.
    """
    cdef float corner_vals[8]
    cdef float vert_coords[13][3]
    cdef signed char edge_buf[36]
    cdef int cube_idx, n_edges, need_edge12, t_idx, e
    cdef unsigned short edges_mask
    cdef int fl_li, fl_off
    cdef float *fl_p
    cdef float w_e, wsum
    cdef int ei0, ei1, ei2, buf_off
    cdef float *fptr
    cdef unsigned short *up
    cdef float ux, uy, uz, wx, wy, wz, nn
    cdef int py = a.py
    cdef int k = a.k
    cdef signed char *sa = a.slice_a
    cdef signed char *sb = a.slice_b
    cdef signed char *ft = a.ft
    cdef int ft_ncols = a.ft_ncols
    cdef float isovalue = a.isovalue
    cdef float mc_ox = a.mc_ox, mc_oy = a.mc_oy, mc_oz = a.mc_oz
    cdef float mc_vsx = a.mc_vsx, mc_vsy = a.mc_vsy, mc_vsz = a.mc_vsz
    cdef int n_tris = 0

    corner_vals[0] = <float>sa[i * py + j]
    corner_vals[1] = <float>sa[(i + 1) * py + j]
    corner_vals[2] = <float>sa[(i + 1) * py + (j + 1)]
    corner_vals[3] = <float>sa[i * py + (j + 1)]
    corner_vals[4] = <float>sb[i * py + j]
    corner_vals[5] = <float>sb[(i + 1) * py + j]
    corner_vals[6] = <float>sb[(i + 1) * py + (j + 1)]
    corner_vals[7] = <float>sb[i * py + (j + 1)]

    cube_idx = 0
    if corner_vals[0] > isovalue: cube_idx |= 1
    if corner_vals[1] > isovalue: cube_idx |= 2
    if corner_vals[2] > isovalue: cube_idx |= 4
    if corner_vals[3] > isovalue: cube_idx |= 8
    if corner_vals[4] > isovalue: cube_idx |= 16
    if corner_vals[5] > isovalue: cube_idx |= 32
    if corner_vals[6] > isovalue: cube_idx |= 64
    if corner_vals[7] > isovalue: cube_idx |= 128

    n_edges = ft[cube_idx * ft_ncols]
    if n_edges == 0:
        return 0

    edges_mask = 0
    need_edge12 = 0
    for t_idx in range(n_edges):
        edge_buf[t_idx] = ft[cube_idx * ft_ncols + 1 + t_idx]
        if edge_buf[t_idx] == 12:
            need_edge12 = 1
        elif 0 <= edge_buf[t_idx] < 12:
            edges_mask |= <unsigned short>(1 << edge_buf[t_idx])

    # Read pre-computed edge vertices from face-layers (no NaN check)
    for e in range(12):
        if not (edges_mask & (1 << e)):
            continue
        fl_li = _EDGE_LAYER[e]
        fl_p = a.layer_ptrs[fl_li]
        fl_off = ((i + _EDGE_DI_OFF[e]) * a.layer_jstride[fl_li]
                  + (j + _EDGE_DJ_OFF[e]) * 3)
        vert_coords[e][0] = fl_p[fl_off]
        vert_coords[e][1] = fl_p[fl_off + 1]
        vert_coords[e][2] = fl_p[fl_off + 2]

    # Edge 12: center vertex (cell-local, unchanged from sequential)
    if need_edge12:
        wsum = 0.0
        vert_coords[12][0] = 0.0
        vert_coords[12][1] = 0.0
        vert_coords[12][2] = 0.0
        for e in range(8):
            w_e = corner_vals[e] - isovalue
            if w_e < 0:
                w_e = -w_e
            w_e = 1.0 / (LEW_EPS + w_e)
            vert_coords[12][0] += w_e * (mc_ox + mc_vsx * <float>(i + _VTX_DI[e]))
            vert_coords[12][1] += w_e * (mc_oy + mc_vsy * <float>(j + _VTX_DJ[e]))
            vert_coords[12][2] += w_e * (mc_oz + mc_vsz * <float>(k + _VTX_DK[e]))
            wsum += w_e
        if wsum > 0.0:
            vert_coords[12][0] /= wsum
            vert_coords[12][1] /= wsum
            vert_coords[12][2] /= wsum

    # Assemble triangles — ei1/ei2 swap for outward normals (identical to sequential)
    t_idx = 0
    while t_idx < n_edges:
        ei0 = edge_buf[t_idx]
        ei1 = edge_buf[t_idx + 2]
        ei2 = edge_buf[t_idx + 1]
        buf_off = (buf_start + n_tris) * 50
        fptr = <float*>(&buf[buf_off])
        ux = vert_coords[ei1][0] - vert_coords[ei0][0]
        uy = vert_coords[ei1][1] - vert_coords[ei0][1]
        uz = vert_coords[ei1][2] - vert_coords[ei0][2]
        wx = vert_coords[ei2][0] - vert_coords[ei0][0]
        wy = vert_coords[ei2][1] - vert_coords[ei0][1]
        wz = vert_coords[ei2][2] - vert_coords[ei0][2]
        fptr[0] = uy * wz - uz * wy
        fptr[1] = uz * wx - ux * wz
        fptr[2] = ux * wy - uy * wx
        nn = sqrt(fptr[0]*fptr[0] + fptr[1]*fptr[1] + fptr[2]*fptr[2])
        if nn > 0.0:
            fptr[0] = fptr[0] / nn
            fptr[1] = fptr[1] / nn
            fptr[2] = fptr[2] / nn
        fptr[3] = vert_coords[ei0][0]
        fptr[4] = vert_coords[ei0][1]
        fptr[5] = vert_coords[ei0][2]
        fptr[6] = vert_coords[ei1][0]
        fptr[7] = vert_coords[ei1][1]
        fptr[8] = vert_coords[ei1][2]
        fptr[9] = vert_coords[ei2][0]
        fptr[10] = vert_coords[ei2][1]
        fptr[11] = vert_coords[ei2][2]
        up = <unsigned short*>(&buf[buf_off + 48])
        up[0] = 0
        n_tris += 1
        t_idx += 3

    return n_tris


cdef void _mc_assemble_triangles_parallel(mc_args_t *a) noexcept nogil:
    """Pass 2: Parallel triangle assembly — face-layers are read-only.
    Per-thread STL buffers with mutex-protected fwrite.
    """
    cdef int j, i, tid, t, n_cell_tris
    cdef int px = a.px
    cdef int py = a.py
    cdef int n_threads = a.mc_n_threads
    cdef int BUF_MAX = a.BUF_MAX
    cdef signed char *ba = a.band_a
    cdef signed char *bb = a.band_b

    # Per-thread STL buffers
    cdef unsigned char **t_bufs = <unsigned char**>malloc(n_threads * sizeof(unsigned char*))
    cdef int *t_buf_counts = <int*>malloc(n_threads * sizeof(int))
    cdef int *t_tri_counts = <int*>malloc(n_threads * sizeof(int))

    for t in range(n_threads):
        t_bufs[t] = <unsigned char*>malloc(BUF_MAX * 50)
        t_buf_counts[t] = 0
        t_tri_counts[t] = 0

    for j in prange(py - 1, num_threads=n_threads, schedule='dynamic', chunksize=4):
        tid = threadid()
        for i in range(px - 1):
            # OPT 15: Band skip (identical to sequential)
            if (ba[i * py + j] == 0 and ba[(i + 1) * py + j] == 0
                    and ba[i * py + (j + 1)] == 0
                    and ba[(i + 1) * py + (j + 1)] == 0
                    and bb[i * py + j] == 0
                    and bb[(i + 1) * py + j] == 0
                    and bb[i * py + (j + 1)] == 0
                    and bb[(i + 1) * py + (j + 1)] == 0):
                continue

            # Flush if not enough room for worst case (12 tris/cell)
            if t_buf_counts[tid] > BUF_MAX - 12:
                pthread_mutex_lock(&a.write_mutex)
                fwrite(t_bufs[tid], 1, t_buf_counts[tid] * 50, a.fp)
                pthread_mutex_unlock(&a.write_mutex)
                t_buf_counts[tid] = 0

            n_cell_tris = _mc_emit_cell(a, i, j, t_bufs[tid], t_buf_counts[tid])
            t_buf_counts[tid] += n_cell_tris
            t_tri_counts[tid] += n_cell_tris

    # Flush remaining per-thread buffers and sum counts
    cdef int total_tris = 0
    for t in range(n_threads):
        if t_buf_counts[t] > 0:
            fwrite(t_bufs[t], 1, t_buf_counts[t] * 50, a.fp)
        total_tris += t_tri_counts[t]
        free(t_bufs[t])
    a.tri_count += total_tris

    free(t_bufs)
    free(t_buf_counts)
    free(t_tri_counts)


cdef void _mc_process_layer_two_pass(mc_args_t *a, int is_first_layer) noexcept nogil:
    """OPT 16: Two-pass parallel MC preserving face-layer vertex dedup.
    Pass 1: Pre-compute edge vertices in parallel (each position written once).
    Pass 2: Assemble triangles in parallel (face-layers read-only).
    """
    _precompute_layer_edges(a, is_first_layer)
    _mc_assemble_triangles_parallel(a)


def fused_stl_export(
    const unsigned char[::1] packed,
    long long rx, long long ry, long long rz,
    const signed char[:, :, ::1] kernel,
    float vsx, float vsy, float vsz,
    str filename,
    int stride=1,
    float isovalue=0.0,
    int n_threads=0,
    int compute_normals=0,
):
    """Fully fused binary STL export: packed bits -> STL file on disk.

    Fuses scale+convolve (Phase 1) + marching cubes + binary STL write
    into a single streaming kernel. Only 2 Z-slices of convolved output
    are held in memory at a time — no intermediate volumes, no mesh arrays.

    The convolution is parallelized with OpenMP over the X-axis for each
    Z-slice. MC cell processing and STL writes are sequential, streaming
    triangles to disk via a buffered write.

    Binary STL format: 80-byte header, 4-byte uint32 triangle count,
    then 50 bytes per triangle (normal + 3 vertices + attribute).

    References:
        [1] Lorensen & Cline, "Marching Cubes: A High Resolution 3D
            Surface Construction Algorithm", SIGGRAPH 1987.

    Args:
        packed: F-order packed binary volume (big-endian bitorder)
        rx, ry, rz: full-resolution grid dimensions
        kernel: int8 quantized Butterworth kernel, shape (2r+1, 2r+1, 2r+1)
        vsx, vsy, vsz: voxel spacing per axis
        filename: output STL file path
        stride: subsample factor (1 = full resolution, 2 = half, etc.)
        isovalue: isosurface level (typically 0.0)
        n_threads: OpenMP threads (0 = auto-detect)

    Returns:
        int: number of triangles written
    """
    from voxelcad._kernels import _mc_tables_lewiner as _lew_mod

    # Strided output dimensions (same as fused_scale_convolve)
    cdef int sx = <int>((rx + stride - 1) // stride)
    cdef int sy = <int>((ry + stride - 1) // stride)
    cdef int sz = <int>((rz + stride - 1) // stride)
    cdef int px = sx + 2  # padded
    cdef int py = sy + 2
    cdef int pz = sz + 2

    cdef int kx = kernel.shape[0]
    cdef int ky = kernel.shape[1]
    cdef int kz_dim = kernel.shape[2]
    cdef int krx = kx // 2
    cdef int kry = ky // 2
    cdef int krz = kz_dim // 2

    # MC voxel spacing and origin offset:
    # stride=1: Phase 1+2 pipeline strips padding → offset by -1 voxel
    # stride>1: Phase 1+2 pipeline keeps padding → origin at (0,0,0)
    cdef float mc_vsx, mc_vsy, mc_vsz
    cdef float mc_ox, mc_oy, mc_oz
    if stride == 1:
        mc_vsx = vsx
        mc_vsy = vsy
        mc_vsz = vsz
        mc_ox = -vsx
        mc_oy = -vsy
        mc_oz = -vsz
    else:
        mc_vsx = vsx * <float>stride
        mc_vsy = vsy * <float>stride
        mc_vsz = vsz * <float>stride
        mc_ox = 0.0
        mc_oy = 0.0
        mc_oz = 0.0

    # Pre-merged flat tiling table: O(1) cube_idx → edge list lookup.
    # Replaces full Lewiner 15-case dispatch. Uses simpler tiling variant
    # for rare ambiguous cases (lossless for Butterworth-smoothed int8 SDF).
    # Full Lewiner dispatch with face/interior tests preserved in:
    #   - sweep_mc_mesh() in THIS FILE (precision path, always available)
    #   - Git commit 2ac923d (last commit with full dispatch in fused_stl_export)
    # Recovery: copy the big_switch from sweep_mc_mesh when building Task #95
    # (CDT Precision Path) which needs state-of-the-art ambiguity resolution.
    cdef const signed char[:, ::1] ft = _lew_mod.fast_tiling

    # Two Z-slices of convolved output (int8)
    slice_a_np = np.full((px, py), -1, dtype=np.int8)
    slice_b_np = np.full((px, py), -1, dtype=np.int8)
    slice_c_np = np.full((px, py), -1, dtype=np.int8)
    cdef signed char[:, ::1] slice_a = slice_a_np
    cdef signed char[:, ::1] slice_b = slice_b_np
    cdef signed char[:, ::1] slice_c = slice_c_np

    # OPT 15: Surface band mask (conv→MC skip)
    band_a_np = np.zeros((px, py), dtype=np.int8)
    band_b_np = np.zeros((px, py), dtype=np.int8)
    band_c_np = np.zeros((px, py), dtype=np.int8)
    cdef signed char[:, ::1] band_a = band_a_np
    cdef signed char[:, ::1] band_b = band_b_np
    cdef signed char[:, ::1] band_c = band_c_np

    # Source packed bits
    cdef const unsigned char *src_ptr = &packed[0]
    cdef long long rx_ll = rx
    cdef long long rxy = rx * ry
    cdef int str_val = stride

    # Convolution variables
    cdef int pi, pj, di, dj, dk
    cdef int s_i, s_j, s_k, f_i, f_j, f_k
    cdef short acc
    cdef int conv_val
    cdef long long bit_idx
    cdef signed char src_val
    cdef signed char center_val
    cdef int center_sx, center_sy
    cdef int conv_k

    # Face-layer coordinate arrays for vertex dedup (NaN = not computed)
    # Same topology as sweep_mc_mesh but stores float32[3] coords, not int32 ids
    lax_np = np.full((px - 1, py, 3), np.nan, dtype=np.float32)
    lay_np = np.full((px, py - 1, 3), np.nan, dtype=np.float32)
    lbx_np = np.full((px - 1, py, 3), np.nan, dtype=np.float32)
    lby_np = np.full((px, py - 1, 3), np.nan, dtype=np.float32)
    zed_np = np.full((px, py, 3), np.nan, dtype=np.float32)

    cdef float[:, :, ::1] lax = lax_np
    cdef float[:, :, ::1] lay = lay_np
    cdef float[:, :, ::1] lbx = lbx_np
    cdef float[:, :, ::1] lby = lby_np
    cdef float[:, :, ::1] zed = zed_np

    # MC variables
    cdef int i, j, e, t_idx, k
    cdef int cube_idx
    cdef unsigned short edges_mask
    cdef float corner_vals[8]
    cdef float vert_coords[13][3]  # 0-11 = edges, 12 = center vertex
    cdef float v0_val, v1_val, t_interp
    cdef int v0_idx, v1_idx
    cdef int tri_count = 0
    cdef float vx, vy, vz
    # Cross-product normal computation (no gradient — Lewiner guarantees winding)
    cdef float ux, uy, uz, wx, wy, wz, nn
    # Lewiner dispatch variables
    cdef int case_id, lew_config, lew_subconfig
    cdef signed char edge_buf[36]  # max 12 tri * 3 edges
    cdef int n_edges, need_edge12
    cdef float w_e, wsum

    # Face-layer jump table: pointers + strides for indexed edge dispatch
    cdef float *layer_ptrs[5]
    cdef int layer_jstride[5]
    cdef int fl_li, fl_off
    cdef float *fl_p
    cdef float *tmp_ptr

    # STL write buffer (4096 triangles * 50 bytes = 200 KB)
    cdef int BUF_MAX = 4096
    stl_buf_np = np.zeros(BUF_MAX * 50, dtype=np.uint8)
    cdef unsigned char[::1] stl_buf_view = stl_buf_np
    cdef unsigned char *stl_buf = &stl_buf_view[0]
    cdef float *fptr
    cdef unsigned short *up
    cdef int buf_count = 0
    cdef int buf_off
    cdef int ei0, ei1, ei2

    # OPT 9: Slab buffer variables (amortize bit extraction)
    cdef int pad_x = krx * str_val
    cdef int pad_y = kry * str_val
    cdef int slab_rx_dim = <int>rx + 2 * pad_x
    cdef int slab_ry_dim = <int>ry + 2 * pad_y
    cdef int slab_slice_size = slab_rx_dim * slab_ry_dim
    cdef int slab_slot, slab_fi, slab_fj
    cdef int new_src_z, evict_slot
    cdef signed char *slab_ptr
    cdef int xi, yi
    cdef int src_z
    cdef long long slab_bit_idx
    cdef int dk_off[64]  # precomputed slot_offset = slot * slab_slice_size
    cdef mc_args_t mc_args
    cdef int _l
    cdef pthread_t mc_thread

    if n_threads <= 0:
        n_threads = _get_fused_export_threads(px)

    # Open file with C-level I/O (no GIL needed for writes)
    cdef bytes fn_bytes = filename.encode('utf-8')
    cdef FILE *fp = fopen(fn_bytes, "wb")
    if fp == NULL:
        raise IOError(f"Cannot open file: {filename}")
    # Write STL header + placeholder triangle count (4 bytes)
    # Use VTK-style header — some viewers (macOS Preview) may use it
    cdef unsigned char c_header[84]
    memset(c_header, 0, 84)
    cdef const char *hdr_str = b"VoxelCAD fused STL export"
    cdef int hdr_len = 25
    memset(c_header, 0x20, 80)  # space-fill
    cdef int hi
    for hi in range(hdr_len):
        c_header[hi] = <unsigned char>hdr_str[hi]
    fwrite(c_header, 1, 84, fp)

    # --- OPT 9: Allocate slab buffer and extract initial Z-slices ---
    # Circular buffer of kz_dim int8 Z-slices, padded in X/Y to eliminate
    # bounds checks. Pre-filled with -1 (outside = -1 convention).
    slab_np = np.full((kz_dim, slab_rx_dim, slab_ry_dim), -1, dtype=np.int8)
    cdef signed char[:, :, ::1] slab_view = slab_np
    slab_ptr = &slab_view[0, 0, 0]

    # Extract initial kz_dim source Z-slices into slab
    conv_k = 0
    for dk in range(kz_dim):
        src_z = (conv_k + dk - krz) * str_val
        if src_z < 0 or src_z >= <int>rz:
            continue  # slot stays -1 (padding)
        slab_slot = (conv_k + dk) % kz_dim
        with nogil:
            for xi in prange(<int>rx, num_threads=n_threads, schedule='static'):
                for yi in range(<int>ry):
                    slab_bit_idx = (
                        <long long>xi +
                        <long long>yi * rx_ll +
                        <long long>src_z * rxy)
                    if (src_ptr[slab_bit_idx >> 3] >>
                            (7 - <int>(slab_bit_idx & 7))) & 1:
                        slab_ptr[slab_slot * slab_slice_size +
                                 (xi + pad_x) * slab_ry_dim + (yi + pad_y)] = 1

    # --- Compute initial slice_b (z=1) using slab ---
    # Precompute slot offsets to eliminate modulo from inner loop
    for dk in range(kz_dim):
        dk_off[dk] = ((conv_k + dk) % kz_dim) * slab_slice_size
    with nogil:
        for pi in prange(px, num_threads=n_threads, schedule='dynamic', chunksize=4):
            for pj in range(py):
                if pi == 0 or pi == px - 1 or pj == 0 or pj == py - 1:
                    slice_b[pi, pj] = -1
                else:
                    # OPT 13: 6-point axis probe — detect surface band
                    center_sx = ((pi - 1) + krx) * str_val
                    center_sy = ((pj - 1) + kry) * str_val
                    center_val = slab_ptr[
                        dk_off[krz] +
                        center_sx * slab_ry_dim + center_sy]
                    if (slab_ptr[dk_off[0] +
                                 center_sx * slab_ry_dim + center_sy]
                            != center_val
                        or slab_ptr[dk_off[kz_dim - 1] +
                                    center_sx * slab_ry_dim + center_sy]
                            != center_val
                        or slab_ptr[dk_off[krz] +
                                    (pi - 1) * str_val * slab_ry_dim +
                                    center_sy]
                            != center_val
                        or slab_ptr[dk_off[krz] +
                                    ((pi - 1) + kx - 1) * str_val *
                                    slab_ry_dim + center_sy]
                            != center_val
                        or slab_ptr[dk_off[krz] +
                                    center_sx * slab_ry_dim +
                                    (pj - 1) * str_val]
                            != center_val
                        or slab_ptr[dk_off[krz] +
                                    center_sx * slab_ry_dim +
                                    ((pj - 1) + ky - 1) * str_val]
                            != center_val):
                        # Surface band: full convolution
                        acc = 0
                        for di in range(kx):
                            for dj in range(ky):
                                for dk in range(kz_dim):
                                    if kernel[di, dj, dk] == 0:
                                        continue
                                    slab_fi = ((pi - 1) + di) * str_val
                                    slab_fj = ((pj - 1) + dj) * str_val
                                    src_val = slab_ptr[
                                        dk_off[dk] +
                                        slab_fi * slab_ry_dim + slab_fj]
                                    acc = acc + <short>(
                                        <short>src_val *
                                        <short>kernel[di, dj, dk])
                        conv_val = <int>acc
                        if conv_val > 127:
                            conv_val = 127
                        elif conv_val < -128:
                            conv_val = -128
                        slice_b[pi, pj] = <signed char>conv_val
                        band_b[pi, pj] = 1  # OPT 15: surface
                    else:
                        # Interior/exterior: branchless constant fill
                        slice_b[pi, pj] = <signed char>(
                            <int>center_val * 127 +
                            (<int>center_val >> 7))
                        band_b[pi, pj] = 0  # OPT 15: non-surface

    # --- Init face-layer jump table ---
    # layer_jstride[L] = number of floats per i-row in layer L
    layer_jstride[0] = py * 3       # lax: (px-1, py, 3)
    layer_jstride[1] = (py-1) * 3   # lay: (px, py-1, 3)
    layer_jstride[2] = py * 3       # lbx: (px-1, py, 3)
    layer_jstride[3] = (py-1) * 3   # lby: (px, py-1, 3)
    layer_jstride[4] = py * 3       # zed: (px, py, 3)
    layer_ptrs[0] = &lax[0, 0, 0]
    layer_ptrs[1] = &lay[0, 0, 0]
    layer_ptrs[2] = &lbx[0, 0, 0]
    layer_ptrs[3] = &lby[0, 0, 0]
    layer_ptrs[4] = &zed[0, 0, 0]

    # OPT 10: Initialize MC args struct
    mc_args.px = px
    mc_args.py = py
    mc_args.ft = <signed char*>&ft[0, 0]
    mc_args.ft_ncols = ft.shape[1]
    mc_args.mc_ox = mc_ox
    mc_args.mc_oy = mc_oy
    mc_args.mc_oz = mc_oz
    mc_args.mc_vsx = mc_vsx
    mc_args.mc_vsy = mc_vsy
    mc_args.mc_vsz = mc_vsz
    mc_args.isovalue = isovalue
    mc_args.stl_buf = stl_buf
    mc_args.buf_count = 0
    mc_args.tri_count = 0
    mc_args.fp = fp
    mc_args.BUF_MAX = BUF_MAX
    for _l in range(5):
        mc_args.layer_ptrs[_l] = layer_ptrs[_l]
        mc_args.layer_jstride[_l] = layer_jstride[_l]

    # OPT 17: Normal elision flag
    mc_args.compute_normals = compute_normals
    # --- OPT 16: Set MC thread count and init write mutex ---
    mc_args.mc_n_threads = _get_fused_export_threads(py - 1)
    pthread_mutex_init(&mc_args.write_mutex, NULL)

    # --- OPT 10: Init pthread MC pipeline ---
    mc_args.mc_go = 0
    mc_args.mc_finished = 0
    mc_args.terminate = 0
    pthread_mutex_init(&mc_args.mutex, NULL)
    pthread_cond_init(&mc_args.slice_ready, NULL)
    pthread_cond_init(&mc_args.mc_done, NULL)
    pthread_create(&mc_thread, NULL, _mc_thread_func, &mc_args)

    # --- Main Z-sweep: MC on slice pairs + STL write ---
    for k in range(pz - 1):
        mc_args.k = k
        mc_args.slice_a = &slice_a[0, 0]
        mc_args.slice_b = &slice_b[0, 0]
        mc_args.band_a = &band_a[0, 0]
        mc_args.band_b = &band_b[0, 0]

        # Signal MC thread to start processing current pair
        with nogil:
            pthread_mutex_lock(&mc_args.mutex)
            mc_args.mc_go = 1
            mc_args.mc_finished = 0
            pthread_cond_signal(&mc_args.slice_ready)
            pthread_mutex_unlock(&mc_args.mutex)

        # Conv next slice into slice_c while MC runs (overlap)
        if k + 2 < pz:
            if k + 2 >= pz - 1:
                slice_c_np[:] = -1
                slice_c = slice_c_np
                band_c_np[:] = 0  # OPT 15: boundary slice has no surface
                band_c = band_c_np
            else:
                conv_k = k + 2 - 1
                evict_slot = (conv_k - 1) % kz_dim
                memset(slab_ptr + evict_slot * slab_slice_size,
                       0xFF, slab_slice_size)
                new_src_z = (conv_k + krz) * str_val
                if new_src_z >= 0 and new_src_z < <int>rz:
                    with nogil:
                        for xi in prange(<int>rx, num_threads=n_threads,
                                         schedule='static'):
                            for yi in range(<int>ry):
                                slab_bit_idx = (
                                    <long long>xi +
                                    <long long>yi * rx_ll +
                                    <long long>new_src_z * rxy)
                                if (src_ptr[slab_bit_idx >> 3] >>
                                        (7 - <int>(slab_bit_idx & 7))) & 1:
                                    slab_ptr[
                                        evict_slot * slab_slice_size +
                                        (xi + pad_x) * slab_ry_dim +
                                        (yi + pad_y)] = 1
                for dk in range(kz_dim):
                    dk_off[dk] = ((conv_k + dk) % kz_dim) * slab_slice_size
                with nogil:
                    for pi in prange(px, num_threads=n_threads,
                                     schedule='dynamic', chunksize=4):
                        for pj in range(py):
                            if (pi == 0 or pi == px - 1 or
                                    pj == 0 or pj == py - 1):
                                slice_c[pi, pj] = -1
                            else:
                                # OPT 13: 6-point axis probe
                                center_sx = ((pi - 1) + krx) * str_val
                                center_sy = ((pj - 1) + kry) * str_val
                                center_val = slab_ptr[
                                    dk_off[krz] +
                                    center_sx * slab_ry_dim + center_sy]
                                if (slab_ptr[dk_off[0] +
                                             center_sx * slab_ry_dim +
                                             center_sy]
                                        != center_val
                                    or slab_ptr[dk_off[kz_dim - 1] +
                                                center_sx * slab_ry_dim +
                                                center_sy]
                                        != center_val
                                    or slab_ptr[dk_off[krz] +
                                                (pi - 1) * str_val *
                                                slab_ry_dim + center_sy]
                                        != center_val
                                    or slab_ptr[dk_off[krz] +
                                                ((pi - 1) + kx - 1) *
                                                str_val * slab_ry_dim +
                                                center_sy]
                                        != center_val
                                    or slab_ptr[dk_off[krz] +
                                                center_sx * slab_ry_dim +
                                                (pj - 1) * str_val]
                                        != center_val
                                    or slab_ptr[dk_off[krz] +
                                                center_sx * slab_ry_dim +
                                                ((pj - 1) + ky - 1) *
                                                str_val]
                                        != center_val):
                                    # Surface band: full convolution
                                    acc = 0
                                    for di in range(kx):
                                        for dj in range(ky):
                                            for dk in range(kz_dim):
                                                if kernel[di, dj, dk] == 0:
                                                    continue
                                                slab_fi = ((pi - 1) + di) * str_val
                                                slab_fj = ((pj - 1) + dj) * str_val
                                                src_val = slab_ptr[
                                                    dk_off[dk] +
                                                    slab_fi * slab_ry_dim +
                                                    slab_fj]
                                                acc = acc + <short>(
                                                    <short>src_val *
                                                    <short>kernel[di, dj, dk])
                                    conv_val = <int>acc
                                    if conv_val > 127:
                                        conv_val = 127
                                    elif conv_val < -128:
                                        conv_val = -128
                                    slice_c[pi, pj] = <signed char>conv_val
                                    band_c[pi, pj] = 1  # OPT 15: surface
                                else:
                                    # Interior/exterior: branchless fill
                                    slice_c[pi, pj] = <signed char>(
                                        <int>center_val * 127 +
                                        (<int>center_val >> 7))
                                    band_c[pi, pj] = 0  # OPT 15: non-surface

        # Wait for MC to finish
        with nogil:
            pthread_mutex_lock(&mc_args.mutex)
            while mc_args.mc_finished == 0:
                pthread_cond_wait(&mc_args.mc_done, &mc_args.mutex)
            pthread_mutex_unlock(&mc_args.mutex)

        # Rotate: b->a (next bottom), c->b (next top), a->c (free)
        temp_np = slice_a_np
        slice_a_np = slice_b_np
        slice_b_np = slice_c_np
        slice_c_np = temp_np
        slice_a = slice_a_np
        slice_b = slice_b_np
        slice_c = slice_c_np

        # OPT 15: Rotate band arrays in sync with slices
        temp_np = band_a_np
        band_a_np = band_b_np
        band_b_np = band_c_np
        band_c_np = temp_np
        band_a = band_a_np
        band_b = band_b_np
        band_c = band_c_np

        # Swap face layers (after MC done)
        tmp_ptr = mc_args.layer_ptrs[0]; mc_args.layer_ptrs[0] = mc_args.layer_ptrs[2]; mc_args.layer_ptrs[2] = tmp_ptr
        tmp_ptr = mc_args.layer_ptrs[1]; mc_args.layer_ptrs[1] = mc_args.layer_ptrs[3]; mc_args.layer_ptrs[3] = tmp_ptr
        memset(mc_args.layer_ptrs[2], 0xFF, (px - 1) * py * 3 * sizeof(float))
        memset(mc_args.layer_ptrs[3], 0xFF, px * (py - 1) * 3 * sizeof(float))
        memset(mc_args.layer_ptrs[4], 0xFF, px * py * 3 * sizeof(float))

    # --- Teardown pthread ---
    with nogil:
        pthread_mutex_lock(&mc_args.mutex)
        mc_args.terminate = 1
        pthread_cond_signal(&mc_args.slice_ready)
        pthread_mutex_unlock(&mc_args.mutex)
        pthread_join(mc_thread, NULL)
        pthread_mutex_destroy(&mc_args.mutex)
        pthread_mutex_destroy(&mc_args.write_mutex)
        pthread_cond_destroy(&mc_args.slice_ready)
        pthread_cond_destroy(&mc_args.mc_done)

    # Flush remaining buffer
    if mc_args.buf_count > 0:
        fwrite(stl_buf, 1, mc_args.buf_count * 50, fp)

    # Write triangle count at byte 80
    cdef unsigned int tri_count_u = <unsigned int>mc_args.tri_count
    fseek(fp, 80, SEEK_SET)
    fwrite(&tri_count_u, 4, 1, fp)
    fclose(fp)

    return mc_args.tri_count


# ---------------------------------------------------------------------------
# Fused mesh export: packed bits -> (vertices, faces) numpy arrays
# ---------------------------------------------------------------------------
# Shares convolution frontend design with fused_stl_export above.
# DRY extraction into shared helpers deferred to Task #96 (consolidation).
# MC backend uses int32 face-layer vertex IDs (from sweep_mc_mesh design)
# instead of float32 coordinate storage + STL output.
# ---------------------------------------------------------------------------

cdef void _mc_mesh_assign_ids(mc_args_t *a, int assign_bottom) noexcept nogil:
    """Assign vertex IDs to pre-computed edge vertices in float32 face-layers.
    Sequential — shared mesh_n_verts counter. Copies coords to mesh_verts.
    For k>0, bottom layers (lax/lay) have IDs from previous layer's swap.
    """
    cdef int i, j, off, nv, li
    cdef float *fl
    cdef int *idl
    cdef int imax, jmax
    cdef int start_layer = 0 if assign_bottom else 2

    nv = a.mesh_n_verts

    for li in range(start_layer, 5):
        fl = a.layer_ptrs[li]
        idl = a.id_layers[li]
        if li == 0 or li == 2:  # lax/lbx: (px-1, py)
            imax = a.px - 1; jmax = a.py
        elif li == 1 or li == 3:  # lay/lby: (px, py-1)
            imax = a.px; jmax = a.py - 1
        else:  # zed: (px, py)
            imax = a.px; jmax = a.py

        for i in range(imax):
            for j in range(jmax):
                off = i * a.layer_jstride[li] + j * 3
                if fl[off] == fl[off]:  # not NaN
                    if nv < a.mesh_verts_cap:
                        a.mesh_verts[nv * 3] = fl[off]
                        a.mesh_verts[nv * 3 + 1] = fl[off + 1]
                        a.mesh_verts[nv * 3 + 2] = fl[off + 2]
                        idl[i * a.id_jstride[li] + j] = nv
                        nv += 1

    a.mesh_n_verts = nv


cdef void _mc_mesh_emit_triangles(mc_args_t *a) noexcept nogil:
    """Emit triangles using pre-assigned vertex IDs from int32 face-layers."""
    cdef int i, j, e, t_idx, cube_idx, n_edges, need_edge12
    cdef float corner_vals[8]
    cdef signed char edge_buf[36]
    cdef unsigned short edges_mask
    cdef int edge_vids[13]
    cdef int vid, nv, nf, fl_li
    cdef int *id_p
    cdef int py_val = a.py
    cdef signed char *sa = a.slice_a
    cdef signed char *sb = a.slice_b
    cdef signed char *ft = a.ft
    cdef int ft_nc = a.ft_ncols
    cdef float iso = a.isovalue
    cdef float w_e, wsum

    nv = a.mesh_n_verts
    nf = a.mesh_n_faces

    for i in range(a.px - 1):
        for j in range(a.py - 1):
            # OPT 15: skip non-surface
            if (a.band_a[i * py_val + j] == 0
                    and a.band_a[(i + 1) * py_val + j] == 0
                    and a.band_a[i * py_val + j + 1] == 0
                    and a.band_a[(i + 1) * py_val + j + 1] == 0
                    and a.band_b[i * py_val + j] == 0
                    and a.band_b[(i + 1) * py_val + j] == 0
                    and a.band_b[i * py_val + j + 1] == 0
                    and a.band_b[(i + 1) * py_val + j + 1] == 0):
                continue

            corner_vals[0] = <float>sa[i * py_val + j]
            corner_vals[1] = <float>sa[(i + 1) * py_val + j]
            corner_vals[2] = <float>sa[(i + 1) * py_val + j + 1]
            corner_vals[3] = <float>sa[i * py_val + j + 1]
            corner_vals[4] = <float>sb[i * py_val + j]
            corner_vals[5] = <float>sb[(i + 1) * py_val + j]
            corner_vals[6] = <float>sb[(i + 1) * py_val + j + 1]
            corner_vals[7] = <float>sb[i * py_val + j + 1]

            cube_idx = 0
            if corner_vals[0] > iso: cube_idx |= 1
            if corner_vals[1] > iso: cube_idx |= 2
            if corner_vals[2] > iso: cube_idx |= 4
            if corner_vals[3] > iso: cube_idx |= 8
            if corner_vals[4] > iso: cube_idx |= 16
            if corner_vals[5] > iso: cube_idx |= 32
            if corner_vals[6] > iso: cube_idx |= 64
            if corner_vals[7] > iso: cube_idx |= 128

            n_edges = ft[cube_idx * ft_nc]
            if n_edges == 0:
                continue

            edges_mask = 0
            need_edge12 = 0
            for t_idx in range(n_edges):
                edge_buf[t_idx] = ft[cube_idx * ft_nc + 1 + t_idx]
                if edge_buf[t_idx] == 12:
                    need_edge12 = 1
                elif 0 <= edge_buf[t_idx] < 12:
                    edges_mask |= <unsigned short>(1 << edge_buf[t_idx])

            # Look up vertex IDs from int32 face-layers
            for e in range(12):
                if not (edges_mask & (1 << e)):
                    edge_vids[e] = -1
                    continue
                fl_li = _EDGE_LAYER[e]
                id_p = a.id_layers[fl_li]
                vid = id_p[
                    (i + _EDGE_DI_OFF[e]) * a.id_jstride[fl_li] +
                    (j + _EDGE_DJ_OFF[e])]
                edge_vids[e] = vid

            # Edge 12: center vertex (cell-local, not in face-layers)
            if need_edge12:
                if nv < a.mesh_verts_cap:
                    wsum = 0.0
                    a.mesh_verts[nv * 3] = 0.0
                    a.mesh_verts[nv * 3 + 1] = 0.0
                    a.mesh_verts[nv * 3 + 2] = 0.0
                    for e in range(8):
                        w_e = corner_vals[e] - iso
                        if w_e < 0: w_e = -w_e
                        w_e = 1.0 / (LEW_EPS + w_e)
                        a.mesh_verts[nv * 3] += w_e * (a.mc_ox + a.mc_vsx * <float>(i + _VTX_DI[e]))
                        a.mesh_verts[nv * 3 + 1] += w_e * (a.mc_oy + a.mc_vsy * <float>(j + _VTX_DJ[e]))
                        a.mesh_verts[nv * 3 + 2] += w_e * (a.mc_oz + a.mc_vsz * <float>(a.k + _VTX_DK[e]))
                        wsum += w_e
                    if wsum > 0.0:
                        a.mesh_verts[nv * 3] /= wsum
                        a.mesh_verts[nv * 3 + 1] /= wsum
                        a.mesh_verts[nv * 3 + 2] /= wsum
                    edge_vids[12] = nv
                    nv += 1

            # Emit triangles (ei1/ei2 swap for outward normals)
            t_idx = 0
            while t_idx < n_edges:
                if nf < a.mesh_faces_cap:
                    a.mesh_faces[nf * 3] = edge_vids[edge_buf[t_idx]]
                    a.mesh_faces[nf * 3 + 1] = edge_vids[edge_buf[t_idx + 2]]
                    a.mesh_faces[nf * 3 + 2] = edge_vids[edge_buf[t_idx + 1]]
                    nf += 1
                t_idx += 3

    a.mesh_n_verts = nv
    a.mesh_n_faces = nf


cdef void _mc_mesh_process_layer(mc_args_t *a, int is_first_layer) noexcept nogil:
    """Three-step mesh MC: precompute edges -> assign IDs -> emit triangles."""
    _precompute_layer_edges(a, is_first_layer)
    _mc_mesh_assign_ids(a, is_first_layer)
    _mc_mesh_emit_triangles(a)


cdef void* _mc_mesh_thread_func(void *arg) noexcept nogil:
    """Pthread entry for mesh MC processing."""
    cdef mc_args_t *a = <mc_args_t*>arg
    while True:
        pthread_mutex_lock(&a.mutex)
        while not a.mc_go and not a.terminate:
            pthread_cond_wait(&a.slice_ready, &a.mutex)
        if a.terminate:
            pthread_mutex_unlock(&a.mutex)
            return NULL
        a.mc_go = 0
        pthread_mutex_unlock(&a.mutex)
        _mc_mesh_process_layer(a, a.k == 0)
        pthread_mutex_lock(&a.mutex)
        a.mc_finished = 1
        pthread_cond_signal(&a.mc_done)
        pthread_mutex_unlock(&a.mutex)
    return NULL


def fused_mesh_export(
    const unsigned char[::1] packed,
    long long rx, long long ry, long long rz,
    const signed char[:, :, ::1] kernel,
    float vsx, float vsy, float vsz,
    int stride=1,
    float isovalue=0.0,
    int n_threads=0,
    int mc_threads=0,
):
    """Fully fused mesh export: packed bits -> (vertices, faces) arrays.

    Phase 4b: Uses pthread conv/MC overlap with two-pass nogil MC.
    Pass 1: _precompute_layer_edges fills float32 face-layers (prange).
    Pass 2: Sequential ID assignment + triangle emission via int32 ID layers.

    Args:
        packed: F-order packed binary volume (big-endian bitorder)
        rx, ry, rz: full-resolution grid dimensions
        kernel: int8 quantized Butterworth kernel, shape (2r+1, 2r+1, 2r+1)
        vsx, vsy, vsz: voxel spacing per axis
        stride: subsample factor (1 = full resolution, 2 = half)
        isovalue: isosurface level (typically 0.0)
        n_threads: OpenMP threads for convolution (0 = auto-detect)

    Returns:
        tuple: (vertices float32[N,3], faces int32[M,3])
    """
    from voxelcad._kernels import _mc_tables_lewiner as _lew_mod

    # --- Dimension calculations (same as fused_stl_export) ---
    cdef int sx = <int>((rx + stride - 1) // stride)
    cdef int sy = <int>((ry + stride - 1) // stride)
    cdef int sz = <int>((rz + stride - 1) // stride)
    cdef int px = sx + 2
    cdef int py = sy + 2
    cdef int pz = sz + 2

    cdef int kx = kernel.shape[0]
    cdef int ky = kernel.shape[1]
    cdef int kz_dim = kernel.shape[2]
    cdef int krx = kx // 2
    cdef int kry = ky // 2
    cdef int krz = kz_dim // 2

    # MC spacing and origin (stride-dependent, same logic as fused_stl_export)
    cdef float mc_vsx, mc_vsy, mc_vsz
    cdef float mc_ox, mc_oy, mc_oz
    if stride == 1:
        mc_vsx = vsx; mc_vsy = vsy; mc_vsz = vsz
        mc_ox = -vsx; mc_oy = -vsy; mc_oz = -vsz
    else:
        mc_vsx = vsx * <float>stride
        mc_vsy = vsy * <float>stride
        mc_vsz = vsz * <float>stride
        mc_ox = 0.0; mc_oy = 0.0; mc_oz = 0.0

    # Fast tiling table
    cdef const signed char[:, ::1] ft = _lew_mod.fast_tiling

    # --- Conv slices + band masks ---
    slice_a_np = np.full((px, py), -1, dtype=np.int8)
    slice_b_np = np.full((px, py), -1, dtype=np.int8)
    slice_c_np = np.full((px, py), -1, dtype=np.int8)
    cdef signed char[:, ::1] slice_a = slice_a_np
    cdef signed char[:, ::1] slice_b = slice_b_np
    cdef signed char[:, ::1] slice_c = slice_c_np

    band_a_np = np.zeros((px, py), dtype=np.int8)
    band_b_np = np.zeros((px, py), dtype=np.int8)
    band_c_np = np.zeros((px, py), dtype=np.int8)
    cdef signed char[:, ::1] band_a = band_a_np
    cdef signed char[:, ::1] band_b = band_b_np
    cdef signed char[:, ::1] band_c = band_c_np

    # Source packed bits
    cdef const unsigned char *src_ptr = &packed[0]
    cdef long long rx_ll = rx
    cdef long long rxy = rx * ry
    cdef int str_val = stride

    # Conv variables
    cdef int pi, pj, di, dj, dk
    cdef short acc
    cdef int conv_val
    cdef signed char src_val, center_val
    cdef int center_sx, center_sy
    cdef int conv_k, slab_fi, slab_fj

    # --- Float32 face-layer arrays (Phase 4b: two-pass nogil MC) ---
    # Same topology as fused_stl_export: float32[3] coords with NaN sentinel
    lax_np = np.full((px - 1, py, 3), np.nan, dtype=np.float32)
    lay_np = np.full((px, py - 1, 3), np.nan, dtype=np.float32)
    lbx_np = np.full((px - 1, py, 3), np.nan, dtype=np.float32)
    lby_np = np.full((px, py - 1, 3), np.nan, dtype=np.float32)
    zed_np = np.full((px, py, 3), np.nan, dtype=np.float32)
    cdef float[:, :, ::1] lax = lax_np
    cdef float[:, :, ::1] lay = lay_np
    cdef float[:, :, ::1] lbx = lbx_np
    cdef float[:, :, ::1] lby = lby_np
    cdef float[:, :, ::1] zed = zed_np

    # Parallel int32 ID face-layers (-1 sentinel = no vertex assigned)
    id_lax_np = np.full((px - 1, py), -1, dtype=np.int32)
    id_lay_np = np.full((px, py - 1), -1, dtype=np.int32)
    id_lbx_np = np.full((px - 1, py), -1, dtype=np.int32)
    id_lby_np = np.full((px, py - 1), -1, dtype=np.int32)
    id_zed_np = np.full((px, py), -1, dtype=np.int32)
    cdef int[:, ::1] id_lax = id_lax_np
    cdef int[:, ::1] id_lay = id_lay_np
    cdef int[:, ::1] id_lbx = id_lbx_np
    cdef int[:, ::1] id_lby = id_lby_np
    cdef int[:, ::1] id_zed = id_zed_np

    # --- Output arrays (pre-allocated generously, resize at sync points) ---
    cdef int est_tris = max(8 * (px * py + py * pz + px * pz), 1024)
    verts_np = np.empty((est_tris, 3), dtype=np.float32)
    faces_np = np.empty((est_tris, 3), dtype=np.int32)
    cdef float[:, ::1] verts_mv = verts_np
    cdef int[:, ::1] faces_mv = faces_np
    cdef int new_size

    # MC/pthread variables
    cdef int k, _l
    cdef mc_args_t mc_args
    cdef pthread_t mc_thread
    cdef float *tmp_fl_ptr
    cdef int *tmp_id_ptr

    # Slab buffer variables (OPT 9)
    cdef int pad_x = krx * str_val
    cdef int pad_y = kry * str_val
    cdef int slab_rx_dim = <int>rx + 2 * pad_x
    cdef int slab_ry_dim = <int>ry + 2 * pad_y
    cdef int slab_slice_size = slab_rx_dim * slab_ry_dim
    cdef int new_src_z, evict_slot, slab_slot
    cdef signed char *slab_ptr
    cdef int xi, yi, src_z
    cdef long long slab_bit_idx
    cdef int dk_off[64]

    cdef int total_cores = _detect_p_cores()
    if n_threads <= 0:
        n_threads = _get_fused_export_threads(px)
    if mc_threads <= 0:
        # Conv dominates; give MC just enough threads for prange edge precompute.
        # Empirically: conv=14,mc=2 optimal on 16-core (Gyroid 0.72x of STL).
        mc_threads = max(1, min(total_cores // 4, py - 1))

    # --- OPT 9: Slab buffer allocation and initial extraction ---
    slab_np = np.full((kz_dim, slab_rx_dim, slab_ry_dim), -1, dtype=np.int8)
    cdef signed char[:, :, ::1] slab_view = slab_np
    slab_ptr = &slab_view[0, 0, 0]

    conv_k = 0
    for dk in range(kz_dim):
        src_z = (conv_k + dk - krz) * str_val
        if src_z < 0 or src_z >= <int>rz:
            continue
        slab_slot = (conv_k + dk) % kz_dim
        with nogil:
            for xi in prange(<int>rx, num_threads=n_threads, schedule='static'):
                for yi in range(<int>ry):
                    slab_bit_idx = (
                        <long long>xi +
                        <long long>yi * rx_ll +
                        <long long>src_z * rxy)
                    if (src_ptr[slab_bit_idx >> 3] >>
                            (7 - <int>(slab_bit_idx & 7))) & 1:
                        slab_ptr[slab_slot * slab_slice_size +
                                 (xi + pad_x) * slab_ry_dim + (yi + pad_y)] = 1

    # --- Compute initial slice_b (first convolution at z=0) ---
    for dk in range(kz_dim):
        dk_off[dk] = ((conv_k + dk) % kz_dim) * slab_slice_size
    with nogil:
        for pi in prange(px, num_threads=n_threads, schedule='dynamic', chunksize=4):
            for pj in range(py):
                if pi == 0 or pi == px - 1 or pj == 0 or pj == py - 1:
                    slice_b[pi, pj] = -1
                else:
                    # OPT 13: 6-point axis probe
                    center_sx = ((pi - 1) + krx) * str_val
                    center_sy = ((pj - 1) + kry) * str_val
                    center_val = slab_ptr[
                        dk_off[krz] + center_sx * slab_ry_dim + center_sy]
                    if (slab_ptr[dk_off[0] + center_sx * slab_ry_dim + center_sy] != center_val
                        or slab_ptr[dk_off[kz_dim - 1] + center_sx * slab_ry_dim + center_sy] != center_val
                        or slab_ptr[dk_off[krz] + (pi - 1) * str_val * slab_ry_dim + center_sy] != center_val
                        or slab_ptr[dk_off[krz] + ((pi - 1) + kx - 1) * str_val * slab_ry_dim + center_sy] != center_val
                        or slab_ptr[dk_off[krz] + center_sx * slab_ry_dim + (pj - 1) * str_val] != center_val
                        or slab_ptr[dk_off[krz] + center_sx * slab_ry_dim + ((pj - 1) + ky - 1) * str_val] != center_val):
                        acc = 0
                        for di in range(kx):
                            for dj in range(ky):
                                for dk in range(kz_dim):
                                    if kernel[di, dj, dk] == 0:
                                        continue
                                    slab_fi = ((pi - 1) + di) * str_val
                                    slab_fj = ((pj - 1) + dj) * str_val
                                    src_val = slab_ptr[dk_off[dk] + slab_fi * slab_ry_dim + slab_fj]
                                    acc = acc + <short>(<short>src_val * <short>kernel[di, dj, dk])
                        conv_val = <int>acc
                        if conv_val > 127: conv_val = 127
                        elif conv_val < -128: conv_val = -128
                        slice_b[pi, pj] = <signed char>conv_val
                        band_b[pi, pj] = 1
                    else:
                        slice_b[pi, pj] = <signed char>(<int>center_val * 127 + (<int>center_val >> 7))
                        band_b[pi, pj] = 0

    # --- Init face-layer jump table (float32 coords) ---
    cdef float *layer_ptrs[5]
    cdef int layer_jstride[5]
    layer_jstride[0] = py * 3       # lax: (px-1, py, 3)
    layer_jstride[1] = (py-1) * 3   # lay: (px, py-1, 3)
    layer_jstride[2] = py * 3       # lbx: (px-1, py, 3)
    layer_jstride[3] = (py-1) * 3   # lby: (px, py-1, 3)
    layer_jstride[4] = py * 3       # zed: (px, py, 3)
    layer_ptrs[0] = &lax[0, 0, 0]
    layer_ptrs[1] = &lay[0, 0, 0]
    layer_ptrs[2] = &lbx[0, 0, 0]
    layer_ptrs[3] = &lby[0, 0, 0]
    layer_ptrs[4] = &zed[0, 0, 0]

    # --- Init mc_args_t for pthread mesh MC ---
    mc_args.px = px
    mc_args.py = py
    mc_args.ft = <signed char*>&ft[0, 0]
    mc_args.ft_ncols = ft.shape[1]
    mc_args.mc_ox = mc_ox
    mc_args.mc_oy = mc_oy
    mc_args.mc_oz = mc_oz
    mc_args.mc_vsx = mc_vsx
    mc_args.mc_vsy = mc_vsy
    mc_args.mc_vsz = mc_vsz
    mc_args.isovalue = isovalue
    mc_args.mc_n_threads = mc_threads
    for _l in range(5):
        mc_args.layer_ptrs[_l] = layer_ptrs[_l]
        mc_args.layer_jstride[_l] = layer_jstride[_l]

    # ID layer pointers
    mc_args.id_layers[0] = &id_lax[0, 0]
    mc_args.id_layers[1] = &id_lay[0, 0]
    mc_args.id_layers[2] = &id_lbx[0, 0]
    mc_args.id_layers[3] = &id_lby[0, 0]
    mc_args.id_layers[4] = &id_zed[0, 0]
    mc_args.id_jstride[0] = py        # id_lax: (px-1, py)
    mc_args.id_jstride[1] = py - 1    # id_lay: (px, py-1)
    mc_args.id_jstride[2] = py        # id_lbx: (px-1, py)
    mc_args.id_jstride[3] = py - 1    # id_lby: (px, py-1)
    mc_args.id_jstride[4] = py        # id_zed: (px, py)

    # Mesh output buffers
    mc_args.mesh_verts = &verts_mv[0, 0]
    mc_args.mesh_faces = &faces_mv[0, 0]
    mc_args.mesh_n_verts = 0
    mc_args.mesh_n_faces = 0
    mc_args.mesh_verts_cap = est_tris
    mc_args.mesh_faces_cap = est_tris

    # --- Init pthread pipeline ---
    mc_args.mc_go = 0
    mc_args.mc_finished = 0
    mc_args.terminate = 0
    pthread_mutex_init(&mc_args.mutex, NULL)
    pthread_cond_init(&mc_args.slice_ready, NULL)
    pthread_cond_init(&mc_args.mc_done, NULL)
    pthread_create(&mc_thread, NULL, _mc_mesh_thread_func, &mc_args)

    # ======= Main Z-sweep: pthread MC on slice pairs + streaming conv =======
    for k in range(pz - 1):
        mc_args.k = k
        mc_args.slice_a = &slice_a[0, 0]
        mc_args.slice_b = &slice_b[0, 0]
        mc_args.band_a = &band_a[0, 0]
        mc_args.band_b = &band_b[0, 0]

        # Signal MC thread to process current pair
        with nogil:
            pthread_mutex_lock(&mc_args.mutex)
            mc_args.mc_go = 1
            mc_args.mc_finished = 0
            pthread_cond_signal(&mc_args.slice_ready)
            pthread_mutex_unlock(&mc_args.mutex)

        # --- Advance: compute next conv slice into slice_c ---
        if k + 2 < pz:
            if k + 2 >= pz - 1:
                slice_c_np[:] = -1
                slice_c = slice_c_np
                band_c_np[:] = 0
                band_c = band_c_np
            else:
                conv_k = k + 2 - 1
                evict_slot = (conv_k - 1) % kz_dim
                memset(slab_ptr + evict_slot * slab_slice_size,
                       0xFF, slab_slice_size)
                new_src_z = (conv_k + krz) * str_val
                if new_src_z >= 0 and new_src_z < <int>rz:
                    with nogil:
                        for xi in prange(<int>rx, num_threads=n_threads,
                                         schedule='static'):
                            for yi in range(<int>ry):
                                slab_bit_idx = (
                                    <long long>xi +
                                    <long long>yi * rx_ll +
                                    <long long>new_src_z * rxy)
                                if (src_ptr[slab_bit_idx >> 3] >>
                                        (7 - <int>(slab_bit_idx & 7))) & 1:
                                    slab_ptr[
                                        evict_slot * slab_slice_size +
                                        (xi + pad_x) * slab_ry_dim +
                                        (yi + pad_y)] = 1
                for dk in range(kz_dim):
                    dk_off[dk] = ((conv_k + dk) % kz_dim) * slab_slice_size
                with nogil:
                    for pi in prange(px, num_threads=n_threads,
                                     schedule='dynamic', chunksize=4):
                        for pj in range(py):
                            if (pi == 0 or pi == px - 1 or
                                    pj == 0 or pj == py - 1):
                                slice_c[pi, pj] = -1
                            else:
                                center_sx = ((pi - 1) + krx) * str_val
                                center_sy = ((pj - 1) + kry) * str_val
                                center_val = slab_ptr[
                                    dk_off[krz] + center_sx * slab_ry_dim + center_sy]
                                if (slab_ptr[dk_off[0] + center_sx * slab_ry_dim + center_sy] != center_val
                                    or slab_ptr[dk_off[kz_dim - 1] + center_sx * slab_ry_dim + center_sy] != center_val
                                    or slab_ptr[dk_off[krz] + (pi - 1) * str_val * slab_ry_dim + center_sy] != center_val
                                    or slab_ptr[dk_off[krz] + ((pi - 1) + kx - 1) * str_val * slab_ry_dim + center_sy] != center_val
                                    or slab_ptr[dk_off[krz] + center_sx * slab_ry_dim + (pj - 1) * str_val] != center_val
                                    or slab_ptr[dk_off[krz] + center_sx * slab_ry_dim + ((pj - 1) + ky - 1) * str_val] != center_val):
                                    acc = 0
                                    for di in range(kx):
                                        for dj in range(ky):
                                            for dk in range(kz_dim):
                                                if kernel[di, dj, dk] == 0:
                                                    continue
                                                slab_fi = ((pi - 1) + di) * str_val
                                                slab_fj = ((pj - 1) + dj) * str_val
                                                src_val = slab_ptr[
                                                    dk_off[dk] + slab_fi * slab_ry_dim + slab_fj]
                                                acc = acc + <short>(<short>src_val * <short>kernel[di, dj, dk])
                                    conv_val = <int>acc
                                    if conv_val > 127: conv_val = 127
                                    elif conv_val < -128: conv_val = -128
                                    slice_c[pi, pj] = <signed char>conv_val
                                    band_c[pi, pj] = 1
                                else:
                                    slice_c[pi, pj] = <signed char>(
                                        <int>center_val * 127 + (<int>center_val >> 7))
                                    band_c[pi, pj] = 0


        # Wait for MC to finish
        with nogil:
            pthread_mutex_lock(&mc_args.mutex)
            while mc_args.mc_finished == 0:
                pthread_cond_wait(&mc_args.mc_done, &mc_args.mutex)
            pthread_mutex_unlock(&mc_args.mutex)

        # --- Check capacity, resize if needed (GIL held) ---
        if mc_args.mesh_n_verts > mc_args.mesh_verts_cap * 3 // 4:
            new_size = mc_args.mesh_verts_cap * 2
            verts_np.resize((new_size, 3), refcheck=False)
            verts_mv = verts_np
            mc_args.mesh_verts = &verts_mv[0, 0]
            mc_args.mesh_verts_cap = new_size
        if mc_args.mesh_n_faces > mc_args.mesh_faces_cap * 3 // 4:
            new_size = mc_args.mesh_faces_cap * 2
            faces_np.resize((new_size, 3), refcheck=False)
            faces_mv = faces_np
            mc_args.mesh_faces = &faces_mv[0, 0]
            mc_args.mesh_faces_cap = new_size

        # --- Rotate slices and bands ---
        temp_np = slice_a_np
        slice_a_np = slice_b_np
        slice_b_np = slice_c_np
        slice_c_np = temp_np
        slice_a = slice_a_np
        slice_b = slice_b_np
        slice_c = slice_c_np

        temp_np = band_a_np
        band_a_np = band_b_np
        band_b_np = band_c_np
        band_c_np = temp_np
        band_a = band_a_np
        band_b = band_b_np
        band_c = band_c_np


        # --- Swap face-layers: top->bottom, reset top+z (5bbfac1 pattern) ---
        # CRITICAL: Update ALL struct pointers BEFORE memset
        # Swap float32 coord layers
        tmp_fl_ptr = mc_args.layer_ptrs[0]; mc_args.layer_ptrs[0] = mc_args.layer_ptrs[2]; mc_args.layer_ptrs[2] = tmp_fl_ptr
        tmp_fl_ptr = mc_args.layer_ptrs[1]; mc_args.layer_ptrs[1] = mc_args.layer_ptrs[3]; mc_args.layer_ptrs[3] = tmp_fl_ptr
        # Swap int32 ID layers
        tmp_id_ptr = mc_args.id_layers[0]; mc_args.id_layers[0] = mc_args.id_layers[2]; mc_args.id_layers[2] = tmp_id_ptr
        tmp_id_ptr = mc_args.id_layers[1]; mc_args.id_layers[1] = mc_args.id_layers[3]; mc_args.id_layers[3] = tmp_id_ptr
        # Reset new top layers (NaN for float32, -1 for int32) and z-edges
        memset(mc_args.layer_ptrs[2], 0xFF, (px - 1) * py * 3 * sizeof(float))      # lbx float32
        memset(mc_args.layer_ptrs[3], 0xFF, px * (py - 1) * 3 * sizeof(float))      # lby float32
        memset(mc_args.layer_ptrs[4], 0xFF, px * py * 3 * sizeof(float))             # zed float32
        memset(mc_args.id_layers[2], 0xFF, (px - 1) * py * sizeof(int))              # id_lbx
        memset(mc_args.id_layers[3], 0xFF, px * (py - 1) * sizeof(int))              # id_lby
        memset(mc_args.id_layers[4], 0xFF, px * py * sizeof(int))                    # id_zed

    # --- Teardown pthread ---
    with nogil:
        pthread_mutex_lock(&mc_args.mutex)
        mc_args.terminate = 1
        pthread_cond_signal(&mc_args.slice_ready)
        pthread_mutex_unlock(&mc_args.mutex)
        pthread_join(mc_thread, NULL)
        pthread_mutex_destroy(&mc_args.mutex)
        pthread_cond_destroy(&mc_args.slice_ready)
        pthread_cond_destroy(&mc_args.mc_done)

    return verts_np[:mc_args.mesh_n_verts].copy(), faces_np[:mc_args.mesh_n_faces].copy()
