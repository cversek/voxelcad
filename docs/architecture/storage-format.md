# Storage Format

How VoxelCAD stores voxel data as packed booleans in Fortran order.

## Packed Boolean Representation

Voxel data is stored as `uint8` arrays where each bit represents one voxel. A 1024^3 grid occupies 128 MB packed vs 1 GB as a boolean array — an 8x reduction.

**Auto-packing** (`voxel_model.py:39-55`): The constructor detects boolean input and packs automatically:

```python
if voxel_data is not None and voxel_data.dtype == np.bool_:
    self._voxel_shape = voxel_data.shape
    self.voxel_data = np.packbits(voxel_data.ravel(order='F'), bitorder='big')
```

All rendering paths produce packed output directly — the bool array never exists in normal operation.

## F-Order (Column-Major) Layout

Data is raveled in Fortran order before packing: `np.packbits(V.ravel(order='F'))`.

In a `(rx, ry, rz)` grid, the linear index for voxel `(i, j, k)` is:

```
lin_idx = i + j * rx + k * rx * ry
```

This makes Z-slices contiguous in the packed array. Slice `k` occupies bytes `[k * slice_bytes : (k+1) * slice_bytes]` where `slice_bytes = (rx * ry + 7) / 8`.

### Why F-Order?

1. **Cache-friendly Z-iteration**: Cython kernels parallelize over Z (`prange(rz)`). Each thread writes to a disjoint byte range — no synchronization needed.
2. **Efficient slice extraction**: `_unpack_slice(k)` reads a contiguous byte range, not scattered bits.
3. **Matches rendering access pattern**: Both Cython and NumPy paths iterate Z-slices in the outer loop.

C-order (the NumPy default) would make the last axis fastest, scattering Z-slice bits across the entire array. Five call sites specify `order='F'` explicitly to maintain this convention.

## Bit Order

Bits within each byte are MSB-first (`bitorder='big'`), matching `np.packbits` default.

Bit position for linear index `lin_idx`:

```
byte_idx = lin_idx >> 3           # lin_idx / 8
bit_pos  = 7 - (lin_idx & 7)     # MSB = position 7, LSB = position 0
```

The Cython `set_bit` helper (`_fused_parallel.pyx:55-59`):

```cython
cdef inline void set_bit(unsigned char *packed, long long lin_idx) noexcept nogil:
    cdef long long byte_idx = lin_idx >> 3
    cdef int bit_pos = 7 - <int>(lin_idx & 7)
    packed[byte_idx] = packed[byte_idx] | <unsigned char>(1 << bit_pos)
```

Reading a bit in NumPy (`voxel_model.py:170-175`):

```python
byte_idx = lin_idx >> 3
bit_idx = 7 - (lin_idx & 7)
bits = (src_data[byte_idx] >> bit_idx) & 1
```

## Unpacking

**Full volume** (`voxel_model.py:61-64`):

```python
def _unpack_volume(self):
    n = int(np.prod(self._voxel_shape))
    return np.unpackbits(self.voxel_data, bitorder='big')[:n].reshape(
        self._voxel_shape, order='F').view(np.bool_)
```

The `[:n]` slice trims padding bits added by `np.packbits` when total bits isn't a multiple of 8.

**Single Z-slice** (`voxel_model.py:66-82`):

```python
def _unpack_slice(self, k):
    rx, ry, rz = self._voxel_shape
    slice_size = rx * ry
    start = k * slice_size
    byte_start = start // 8
    byte_end = (start + slice_size + 7) // 8
    bits = np.unpackbits(self.voxel_data[byte_start:byte_end], bitorder='big')
    bit_offset = start - byte_start * 8
    return bits[bit_offset:bit_offset + slice_size].reshape(rx, ry, order='F').view(np.bool_)
```

Slice extraction reads only the bytes containing that slice, not the full array.

## Memory Footprint

| Resolution | Bool Array | Packed uint8 | Mesh (PyVista) |
|------------|-----------|-------------|----------------|
| 64^3 | 262 KB | 32 KB | ~1 MB |
| 256^3 | 16 MB | 2 MB | ~50 MB |
| 512^3 | 128 MB | 16 MB | ~400 MB |
| 1024^3 | 1 GB | 128 MB | ~3.5 GB |

The mesh column reflects `construct_mesh()` output, which depends on surface complexity. Packed storage keeps the voxel representation within L3 cache at typical resolutions.

## Integer Overflow Guard

Cython kernels use `long long` (64-bit signed) for linear index computations. At 1024^3, `lin_idx` can reach ~10^9; at higher resolutions, 32-bit `int` (max ~2.1 billion) would overflow silently. The `set_bit` helper, `byte_idx`, and `slice_bits` are all `long long`.
