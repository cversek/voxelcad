import pytest
import numpy as np
from voxelcad import Cube, Sphere, Cylinder, GyroidCube
from voxelcad.voxel_model import VoxelModel, CSGModel, TransformedModel
from voxelcad.voxel_grid import VoxelGrid
from voxelcad._kernels import CYTHON_AVAILABLE

# Low resolution for fast tests — 32^3 grid on size=10 objects
LOW_RES_VS = 10.0 / 32


@pytest.fixture
def cube32():
    """Cube centered at origin, 32^3 resolution."""
    return Cube(size=10, voxel_size=LOW_RES_VS, center=True)


@pytest.fixture
def sphere32():
    """Sphere r=5 centered at origin, 32^3 resolution."""
    return Sphere(r=5, voxel_size=LOW_RES_VS)


@pytest.fixture
def cylinder32():
    """Cylinder h=10 r=5 centered, 32^3 resolution."""
    return Cylinder(h=10, r=5, center=True, voxel_size=LOW_RES_VS)


@pytest.fixture
def gyroid32():
    """GyroidCube size=10 centered, 32^3 resolution."""
    return GyroidCube(size=10, voxel_size=LOW_RES_VS, center=True)
