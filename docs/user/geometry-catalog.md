# Geometry Catalog

All primitives inherit from `VoxelModel` and support boolean operations, transforms, plotting, and STL export.

## Sphere

```python
Sphere(r, voxel_size=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `r` | float | Radius |
| `voxel_size` | float | Voxel edge length (default: from `ENV.voxel_size`) |

Centered at origin. Bounding box: `[-r, r]` on all axes.

## Cube

```python
Cube(size, voxel_size=None, center=False)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | float | Edge length |
| `voxel_size` | float | Voxel edge length |
| `center` | bool | If True, center at origin. If False, corner at origin. |

## Cylinder

```python
Cylinder(h, r=None, r1=None, r2=None, center=False, voxel_size=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `h` | float | Height (along Z axis) |
| `r` | float | Radius (uniform). Mutually exclusive with `r1`/`r2`. |
| `r1` | float | Radius at bottom (z=0). Use with `r2` for cones/tapers. |
| `r2` | float | Radius at top (z=h). Set `r2 < r1` for a cone. |
| `center` | bool | If True, center vertically at origin. |

**Cone example**: `Cylinder(h=8, r1=4, r2=0)` creates a cone from radius 4 at the base to a point at the top.

## GyroidCube

```python
GyroidCube(size, lattice_param=[1.0, 1.0, 1.0], structure_param=0.0,
           phi=[0.0, 0.0, 0.0], thresh1=1.0, thresh2=None, **kwargs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | float | Edge length of bounding cube |
| `lattice_param` | list[3] or float | Spatial frequency per axis. Higher values = more cells. |
| `structure_param` | float | Shifts the gyroid level surface. `0.0` = standard gyroid. |
| `phi` | list[3] | Phase offsets per axis (radians). |
| `thresh1` | float | Upper threshold for the gyroid function. |
| `thresh2` | float or None | Lower threshold. If set, creates a shell between `thresh2` and `thresh1`. |

The gyroid surface is defined by:

```
G(x,y,z) = cos(ax)*sin(by) + cos(by)*sin(cz) + cos(cz)*sin(ax)
```

where `a, b, c` are derived from `lattice_param` and `size`.

**Shell example**: `GyroidCube(10, thresh1=0.3, thresh2=-0.3)` produces a thin gyroid shell.

**Dense lattice**: `GyroidCube(10, lattice_param=2.0)` doubles the spatial frequency.

## WigglyGyroidCube

```python
WigglyGyroidCube(size, w_freq=5, w_expon=3, w_amp=0.5, **kwargs)
```

Inherits all `GyroidCube` parameters. Adds sinusoidal modulation to the gyroid amplitude.

| Parameter | Type | Description |
|-----------|------|-------------|
| `w_freq` | float | Frequency of the wiggle modulation |
| `w_expon` | float | Exponent applied to the modulation |
| `w_amp` | float | Amplitude of the wiggle effect |

## HyperWigglyGyroidCube

```python
HyperWigglyGyroidCube(size, w_freq=5, w_expon=3, w_amp=0.5, **kwargs)
```

Same parameters as `WigglyGyroidCube` with a more extreme modulation pattern.

## Common Parameters

All primitives accept `**kwargs` passed through to `VoxelModel`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `voxel_size` | float | Voxel edge length. Smaller = finer detail, more memory. |

If `voxel_size` is omitted, the global default from `voxelcad.ENV.voxel_size` is used.

## Shared Methods

All primitives inherit from `VoxelModel`:

| Method | Description |
|--------|-------------|
| `model.plot()` | Interactive 3D visualization (requires PyVista) |
| `model.export(path)` | Export surface mesh to STL file |
| `model.translate([x, y, z])` | Return translated copy |
| `model.rotate_x(deg)` / `rotate_y` / `rotate_z` | Return rotated copy |
| `model.scale([sx, sy, sz])` | Return scaled copy |
| `model | other` | Boolean union |
| `model & other` | Boolean intersection |
| `model - other` | Boolean difference |
| `model ^ other` | Boolean XOR |
| `~model` | Boolean invert |
