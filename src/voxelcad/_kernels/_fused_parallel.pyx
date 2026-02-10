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
from cython.parallel cimport prange

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


def _get_optimal_threads(int rz):
    """Determine optimal thread count."""
    cdef int p_cores = _detect_p_cores()
    return min(p_cores, rz)


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
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for Cube.

    Args:
        xcc, ycc, zcc: 1D cell-center coordinate arrays
        cx, cy, cz: cube center coordinates
        half_sx, half_sy, half_sz: half-widths per axis
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
    cdef double x, y, z
    cdef int actual_threads
    cdef long long _slice_bits_check

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

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
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for Sphere.

    Args:
        xcc, ycc, zcc: 1D cell-center coordinate arrays
        cx, cy, cz: sphere center coordinates
        r_sq: radius squared
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
    cdef double dx, dy, dz, dz_sq, dy_sq
    cdef int actual_threads
    cdef long long _slice_bits_check

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

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
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for Cylinder/Cone.

    Args:
        cx, cy, cz: center of cylinder
        h: height
        r1: radius at bottom (z = cz - h/2)
        r2: radius at top (z = cz + h/2)
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
    cdef double dx, dy, Zc, Pz, R, R_sq, dist_sq
    cdef int actual_threads
    cdef long long _slice_bits_check

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

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
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for GyroidCube.

    Precomputes cos/sin per axis so the inner loop has zero trig calls.

    Args:
        ax, ay, az: pi * lattice_param (angular frequency per axis)
        phi_x, phi_y, phi_z: phase shifts
        structure_param: gyroid structure parameter
        thresh1, thresh2: threshold bounds for shell
        use_thresh2: if True, use (thresh1 < F < thresh2); else (0 < F < thresh1)
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

    # Precompute per-axis trig arrays
    cos_x_arr = np.empty(rx, dtype=np.float64)
    sin_x_arr = np.empty(rx, dtype=np.float64)
    cos_y_arr = np.empty(ry, dtype=np.float64)
    sin_y_arr = np.empty(ry, dtype=np.float64)

    cdef double[::1] cos_x = cos_x_arr
    cdef double[::1] sin_x = sin_x_arr
    cdef double[::1] cos_y = cos_y_arr
    cdef double[::1] sin_y = sin_y_arr

    cdef int i, j, k
    cdef long long lin_idx
    cdef double Xa, Ya, cos_z, sin_z, F
    cdef double lo, hi
    cdef int actual_threads
    cdef long long _slice_bits_check

    # Precompute X-axis trig
    for i in range(rx):
        Xa = xcc[i] * ax + phi_x
        cos_x[i] = cos(Xa)
        sin_x[i] = sin(Xa)

    # Precompute Y-axis trig
    for j in range(ry):
        Ya = ycc[j] * ay + phi_y
        cos_y[j] = cos(Ya)
        sin_y[j] = sin(Ya)

    if use_thresh2:
        lo = thresh1
        hi = thresh2
    else:
        lo = 0.0
        hi = thresh1

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

    for k in prange(rz, nogil=True, num_threads=actual_threads, schedule='static'):
        if fabs(zcc[k] - cz) > half_sz:
            continue
        # Per-slice trig (just 2 calls per Z-slice)
        cos_z = cos(zcc[k] * az + phi_z)
        sin_z = sin(zcc[k] * az + phi_z)
        for j in range(ry):
            if fabs(ycc[j] - cy) > half_sy:
                continue
            for i in range(rx):
                if fabs(xcc[i] - cx) > half_sx:
                    continue
                # Gyroid formula with precomputed trig (0 trig calls here)
                F = cos_x[i] * sin_y[j] + \
                    cos_y[j] * sin_z + \
                    cos_z * sin_x[i] - structure_param
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
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for WigglyGyroidCube.

    Precomputes per-axis base trig and wiggle trig. Inner loop still needs
    12 trig calls for Ffunc on modified coordinates.
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

    # Precompute per-axis trig
    cos_x_arr = np.empty(rx, dtype=np.float64)
    sin_x_arr = np.empty(rx, dtype=np.float64)
    cos_bx_arr = np.empty(rx, dtype=np.float64)
    sin_bx_arr = np.empty(rx, dtype=np.float64)
    xa_arr = np.empty(rx, dtype=np.float64)

    cos_y_arr = np.empty(ry, dtype=np.float64)
    sin_y_arr = np.empty(ry, dtype=np.float64)
    cos_by_arr = np.empty(ry, dtype=np.float64)
    sin_by_arr = np.empty(ry, dtype=np.float64)
    ya_arr = np.empty(ry, dtype=np.float64)

    cdef double[::1] cos_x = cos_x_arr, sin_x = sin_x_arr
    cdef double[::1] cos_bx = cos_bx_arr, sin_bx = sin_bx_arr
    cdef double[::1] xa_v = xa_arr

    cdef double[::1] cos_y = cos_y_arr, sin_y = sin_y_arr
    cdef double[::1] cos_by = cos_by_arr, sin_by = sin_by_arr
    cdef double[::1] ya_v = ya_arr

    cdef int i, j, k
    cdef long long lin_idx
    cdef double Xa, Ya, Za, bX, bY, bZ
    cdef double cosX, sinX, cosY, sinY, cosZ, sinZ
    cdef double cos_bZ, sin_bZ
    cdef double gradX, gradY, gradZ
    cdef double wx, wy, wz
    cdef double x1, y1, z1, x2, y2, z2
    cdef double Fw1, Fw2
    cdef int inside
    cdef int actual_threads
    cdef long long _slice_bits_check

    # Precompute X-axis
    for i in range(rx):
        Xa = xcc[i] * ax
        xa_v[i] = Xa
        cos_x[i] = cos(Xa)
        sin_x[i] = sin(Xa)
        bX = w_freq * Xa
        cos_bx[i] = cos(bX)
        sin_bx[i] = sin(bX)

    # Precompute Y-axis
    for j in range(ry):
        Ya = ycc[j] * ay
        ya_v[j] = Ya
        cos_y[j] = cos(Ya)
        sin_y[j] = sin(Ya)
        bY = w_freq * Ya
        cos_by[j] = cos(bY)
        sin_by[j] = sin(bY)

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

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

                # Gradient of gyroid surface
                gradX = cosZ * cosX - sinX * sinY
                gradY = cosX * cosY - sinY * sinZ
                gradZ = cosY * cosZ - sinZ * sinX

                # Wiggle amplitudes
                wx = w_amp * cpow(cos_by[j] * sin_bZ, w_expon)
                wy = w_amp * cpow(sin_bx[i] * cos_bZ, w_expon)
                wz = w_amp * cpow(cos_bx[i] * sin_by[j], w_expon)

                # Ffunc on displaced coordinates (+ and -)
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
    int n_threads=0,
):
    """Fused evaluate + threshold + bit-pack for HyperWigglyGyroidCube.

    Same as WigglyGyroidCube but with double-frequency modulation terms.
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

    # Precompute per-axis trig
    cos_x_arr = np.empty(rx, dtype=np.float64)
    sin_x_arr = np.empty(rx, dtype=np.float64)
    cos_bx_arr = np.empty(rx, dtype=np.float64)
    sin_bx_arr = np.empty(rx, dtype=np.float64)
    cos_3bx_arr = np.empty(rx, dtype=np.float64)
    sin_3bx_arr = np.empty(rx, dtype=np.float64)
    xa_arr = np.empty(rx, dtype=np.float64)

    cos_y_arr = np.empty(ry, dtype=np.float64)
    sin_y_arr = np.empty(ry, dtype=np.float64)
    cos_by_arr = np.empty(ry, dtype=np.float64)
    sin_by_arr = np.empty(ry, dtype=np.float64)
    cos_3by_arr = np.empty(ry, dtype=np.float64)
    sin_3by_arr = np.empty(ry, dtype=np.float64)
    ya_arr = np.empty(ry, dtype=np.float64)

    cdef double[::1] cos_x = cos_x_arr, sin_x = sin_x_arr
    cdef double[::1] cos_bx = cos_bx_arr, sin_bx = sin_bx_arr
    cdef double[::1] cos_3bx = cos_3bx_arr, sin_3bx = sin_3bx_arr
    cdef double[::1] xa_v = xa_arr

    cdef double[::1] cos_y = cos_y_arr, sin_y = sin_y_arr
    cdef double[::1] cos_by = cos_by_arr, sin_by = sin_by_arr
    cdef double[::1] cos_3by = cos_3by_arr, sin_3by = sin_3by_arr
    cdef double[::1] ya_v = ya_arr

    cdef int i, j, k
    cdef long long lin_idx
    cdef double Xa, Ya, Za, bX, bY, bZ
    cdef double cosX, sinX, cosY, sinY, cosZ, sinZ
    cdef double cos_bZ, sin_bZ, cos_3bZ, sin_3bZ
    cdef double gradX, gradY, gradZ
    cdef double wx, wy, wz
    cdef double x1, y1, z1, x2, y2, z2
    cdef double Fw1, Fw2
    cdef int inside
    cdef int actual_threads
    cdef long long _slice_bits_check
    cdef double half_amp = 0.5 * w_amp
    cdef int p1 = w_expon + 1

    # Precompute X-axis
    for i in range(rx):
        Xa = xcc[i] * ax
        xa_v[i] = Xa
        cos_x[i] = cos(Xa)
        sin_x[i] = sin(Xa)
        bX = w_freq * Xa
        cos_bx[i] = cos(bX)
        sin_bx[i] = sin(bX)
        cos_3bx[i] = cos(3.0 * bX)
        sin_3bx[i] = sin(3.0 * bX)

    # Precompute Y-axis
    for j in range(ry):
        Ya = ycc[j] * ay
        ya_v[j] = Ya
        cos_y[j] = cos(Ya)
        sin_y[j] = sin(Ya)
        bY = w_freq * Ya
        cos_by[j] = cos(bY)
        sin_by[j] = sin(bY)
        cos_3by[j] = cos(3.0 * bY)
        sin_3by[j] = sin(3.0 * bY)

    if n_threads <= 0:
        n_threads = _get_optimal_threads(rz)
    actual_threads = n_threads if (slice_bits % 8 == 0) else 1

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

                # HyperWiggle: double-frequency modulation
                wx = w_amp * cpow(cos_by[j] * sin_bZ, w_expon) + \
                     half_amp * cpow(cos_3by[j] * sin_3bZ, p1)
                wy = w_amp * cpow(sin_bx[i] * cos_bZ, w_expon) + \
                     half_amp * cpow(sin_3bx[i] * cos_3bZ, p1)
                wz = w_amp * cpow(cos_bx[i] * sin_by[j], w_expon) + \
                     half_amp * cpow(cos_3bx[i] * sin_3by[j], p1)

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
