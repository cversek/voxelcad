from voxelcad.gyroid_cube import GyroidCube
from voxelcad.cylinder import Cylinder

import voxelcad.environment as ENV

RES = 512
SMOOTH_ITERS = 1000
DOWNSAMPLE_TIMES = 2

D = 10.0 # mm, with H=1mm defines a part of 1 cm^2 surface area
H = 5.0   # mm
GC_LATTICE_PARAM = 1.0 
GC_THRESH_PARAM  = 0.2

STL_FILENAME = f"gyroid_sponge_D{D:0.2f}xH{H:0.2f}mm_lp{GC_LATTICE_PARAM:0.2f}_th{GC_THRESH_PARAM:0.2f}_res{RES}_ds{DOWNSAMPLE_TIMES}.stl"

#let's choose the longest dimension to choose a good voxel size based on the specified RES
ENV.voxel_size = D/RES #set it as global default

gc = GyroidCube([D,D,H],
                   center=True,
                   lattice_param=GC_LATTICE_PARAM,
                   thresh1=-GC_THRESH_PARAM,
                   thresh2=GC_THRESH_PARAM,
                   ).translate([0,0,H/2.0]) #center on X, Y, not Z

model  = Cylinder(h=H,r=D/2.0) & gc #intersection of gyroid and cylinder

#call this to generate the STL
def export(filename = STL_FILENAME, show=False):
    print("Rendering surface model...")
    model_surf = model.render_surface_mesh(
        smooth_iters = SMOOTH_ITERS,
        downscale_times = DOWNSAMPLE_TIMES,
        only_largest_component = True,
    )
    model_surf.save(filename)
    if show:
        model_surf.plot(color='white',show_edges=True)
