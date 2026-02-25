"""Multi Union — stacking primitives with union_all().

Creates a vertical stack of cubes using list comprehension and
union_all(), demonstrating how to combine many primitives efficiently.

Usage:
    python multi_union_test.py
"""
from voxelcad import Cube, union_all, ENV

ENV.voxel_size = 1  # mm

CUBE_SIZE = 10
Z_SPACING = 2 * CUBE_SIZE
NUM = 5

cubes = [Cube(CUBE_SIZE).translate([0, 0, i * Z_SPACING]) for i in range(NUM)]
model = union_all(cubes)


def export(filename="multi_union.stl", show=False):
    print("Rendering surface model...")
    model_surf = model.render_surface_mesh()
    model_surf.save(filename)
    print(f"Saved: {filename}")
    if show:
        model_surf.plot(color='white', show_edges=True)


if __name__ == "__main__":
    export()