"""Test primitive construction and volume rendering."""
import numpy as np


def test_cube_render(cube32):
    data = cube32.render_volume()
    assert data.dtype == np.uint8
    assert cube32._voxel_shape == (32, 32, 32)
    assert data.sum() > 0


def test_sphere_render(sphere32):
    data = sphere32.render_volume()
    assert data.dtype == np.uint8
    assert sphere32._voxel_shape == (32, 32, 32)
    assert data.sum() > 0


def test_cylinder_render(cylinder32):
    data = cylinder32.render_volume()
    assert data.dtype == np.uint8
    assert cylinder32._voxel_shape == (32, 32, 32)
    assert data.sum() > 0


def test_gyroid_render(gyroid32):
    data = gyroid32.render_volume()
    assert data.dtype == np.uint8
    assert gyroid32._voxel_shape == (32, 32, 32)
    assert data.sum() > 0


def test_unpack_roundtrip(cube32):
    """Pack then unpack preserves geometry."""
    cube32.render_volume()
    V = cube32._unpack_volume()
    assert V.dtype == np.bool_
    assert V.shape == (32, 32, 32)
    # Cube fills entire grid when centered — all voxels should be True
    assert V.all()
