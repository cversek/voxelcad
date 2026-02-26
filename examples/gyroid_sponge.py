"""Gyroid Sponge — high-resolution lattice slab for 3D printing.

Creates a 10x10x10 mm gyroid sponge at 1024^3 resolution with
Laplacian smoothing and mesh downsampling for clean STL output.

Usage:
    python -c "from gyroid_sponge import export; export()"
"""
from voxelcad import GyroidCube, ENV

RES = 1024

X = 10.0 # mm, with H=1mm defines a part of 1 cm^2 surface area
Y = X
H = 10.0   # mm
GC_LATTICE_PARAM = 1.0 
GC_THRESH_PARAM  = 0.2

STL_FILENAME = f"gyroid_sponge_{X:0.2f}x{Y:0.2f}x{H:0.2f}mm_lp{GC_LATTICE_PARAM:0.2f}_th{GC_THRESH_PARAM:0.2f}_res{RES}.stl"

#let's choose the longest dimension to choose a good voxel size based on the specified RES
ENV.voxel_size = X/RES #set it as global default

model = GyroidCube([X,X,H],
                   center=True,
                   lattice_param=GC_LATTICE_PARAM,
                   thresh1=-GC_THRESH_PARAM,
                   thresh2=GC_THRESH_PARAM,
                   ).translate([0,0,H/2.0]) #center on X, Y, not Z



#call this to generate the STL
def export(filename = STL_FILENAME, show=True):
    print("Rendering surface model via EDT...")
    model_surf = model.render_surface_mesh_edt(only_largest_component=True)
    model_surf.save(filename)
    print(f"Saved: {filename}")
    if show:
        model_surf.plot(color='white', show_edges=True)
