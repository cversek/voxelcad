"""Hello Gyroid Sphere — VoxelCAD quick start.

Creates a gyroid lattice intersected with a sphere and exports it as
an STL file ready for 3D printing.  This is the simplest example that
showcases what makes VoxelCAD unique: boolean operations between
implicit-surface lattices and conventional primitives.

Usage:
    python hello_gyroid_sphere.py
"""
from voxelcad import Sphere, GyroidCube, ENV

# Resolution: smaller voxel_size = finer detail, more memory
# 0.05 mm on a 10 mm sphere -> 200^3 grid (~1 MB packed)
ENV.voxel_size = 0.05  # mm

# A sphere of radius 5 mm
sphere = Sphere(r=5)

# A gyroid lattice filling a 12 mm cube, centered at origin
gyroid = GyroidCube(12, center=True, lattice_param=1.5,
                    thresh1=-0.3, thresh2=0.3)

# Intersect: keep only the gyroid inside the sphere
model = sphere & gyroid

# Export to STL via EDT pipeline (smooth, fast, no meshfix needed)
print("Rendering and exporting gyroid sphere...")
model.export("hello_gyroid_sphere.stl", only_largest_component=True)
print("Saved: hello_gyroid_sphere.stl")

# Visualize the smooth EDT surface
model.plot(mode="surf")
