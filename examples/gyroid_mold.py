"""Gyroid Mold — multi-component assembly with CSG difference.

Builds a bottle-shaped gyroid object (body + base + neck), then
subtracts it from a solid cylinder to create a negative mold.
Demonstrates union, intersection, difference, and translate.

Usage:
    python gyroid_mold.py
"""
from voxelcad import GyroidCube, Cylinder, ENV

ENV.voxel_size = 6 / 64  # ~0.094 mm, equivalent to res=64 on a 6mm cube

BODY_H = 5
BODY_R = 3

BASE_H = 1
BASE_R = BODY_R

NECK_H = 4
NECK_R = 1

MOLD_R = BODY_R + 1
MOLD_H = BODY_H + NECK_H - 5

gc = GyroidCube(6, center=True, lattice_param=1.0,
                thresh1=0.75, thresh2=3.0).translate([0, 0, BODY_H / 2.0])

body = Cylinder(h=BODY_H, r=BODY_R) & gc
base = Cylinder(h=BASE_H, r=BASE_R).translate([0, 0, BODY_H - 0.1])
neck = Cylinder(h=NECK_H, r=NECK_R).translate([0, 0, BODY_H + BASE_H - 0.1])

model = (base | neck) | body

mold = Cylinder(h=MOLD_H, r=MOLD_R) - model.translate([0, 0, -2])


def export(filename="gyroid_mold.stl", show=False):
    print("Rendering surface model...")
    model_surf = mold.render_surface_mesh(
        smooth_iters=500,
        only_largest_component=True,
    )
    model_surf.save(filename)
    print(f"Saved: {filename}")
    if show:
        model_surf.plot(color='white', show_edges=True)


if __name__ == "__main__":
    export()
