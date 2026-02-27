"""Hello Gyroid Sphere — VoxelCAD quick start.

Creates a gyroid lattice intersected with a sphere and exports it as
an STL file ready for 3D printing.  This is the simplest example that
showcases what makes VoxelCAD unique: boolean operations between
implicit-surface lattices and conventional primitives.

Usage:
    python hello_gyroid_sphere.py
"""
from voxelcad import Sphere, GyroidCube, ENV

# Resolution: 0.025 mm on a 12 mm cube -> ~480^3 grid
ENV.voxel_size = 0.025  # mm

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

# Export a decimated version for smaller file size (90% triangle reduction)
model.export("hello_gyroid_sphere_decimated.stl",
             only_largest_component=True, target_reduction=0.9)
print("Saved: hello_gyroid_sphere_decimated.stl")

# Visualize: full resolution surface with edges
model.plot(mode="surf", show_edges=True)

# Visualize: decimated surface with edges for comparison
model.plot(mode="surf", show_edges=True, target_reduction=0.9)
