"""Transforms Demo — translate, rotate, scale, and compose.

Shows how to position and orient primitives using VoxelCAD's lazy
transform system.  Transforms compose into a single matrix — no
intermediate volumes are materialized until export.

Usage:
    python transforms_demo.py
"""
from voxelcad import Cube, Cylinder, Sphere, ENV

ENV.voxel_size = 0.1  # mm

# --- Translate: stack two cubes ---
base = Cube(10)
top = Cube(6).translate([2, 2, 10])
stacked = base | top

# --- Rotate: tilt a cylinder 45 degrees ---
pillar = Cylinder(h=15, r=2).rotate_x(45)

# --- Scale: flatten a sphere into a disk ---
disk = Sphere(r=5).scale([1, 1, 0.3])

# --- Compose: combine everything into one model ---
model = stacked | pillar.translate([0, 0, 5]) | disk.translate([0, 0, 20])

print("Rendering and exporting transforms demo...")
model.export("transforms_demo.stl")
print("Saved: transforms_demo.stl")
