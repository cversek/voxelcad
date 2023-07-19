from voxelcad.gyroid_cube import GyroidCube

import voxelcad.environment as ENV

RES = 1024
SMOOTH_ITERS = 1000
DOWNSAMPLE_TIMES = 3

X = 10.0 # mm, with H=1mm defines a part of 1 cm^2 surface area
Y = X
H = 10.0   # mm
GC_LATTICE_PARAM = 1.0 
GC_THRESH_PARAM  = 0.2

STL_FILENAME = f"gyroid_sponge_{X:0.2f}x{Y:0.2f}x{H:0.2f}mm_lp{GC_LATTICE_PARAM:0.2f}_th{GC_THRESH_PARAM:0.2f}_res{RES}_ds{DOWNSAMPLE_TIMES}.stl"

#let's choose the longest dimension to choose a good voxel size based on the specified RES
ENV.voxel_size = X/RES #set it as global default

model = GyroidCube([X,X,H],
                   center=True,
                   lattice_param=GC_LATTICE_PARAM,
                   thresh1=-GC_THRESH_PARAM,
                   thresh2=GC_THRESH_PARAM,
                   ).translate([0,0,H/2.0]) #center on X, Y, not Z



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
