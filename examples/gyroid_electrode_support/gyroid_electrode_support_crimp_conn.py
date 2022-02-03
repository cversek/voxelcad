from voxelcad.gyroid_cube import GyroidCube
from voxelcad.cylinder import Cylinder
import voxelcad.environment as ENV

RES = 256
SMOOTH_ITERS     = 0
DOWNSAMPLE_TIMES = 0

BODY_H = 5
BODY_R = 3

GC_SIZE = BODY_R*2

PLUG_R = 1.8/2 #must fit inside a 2.0mm inner diam tube
PLUG_TAPER_H = 1
PLUG_TAPER_R1 = PLUG_R
PLUG_TAPER_R2 = PLUG_TAPER_R1 - 0.5
PLUG_H = 5 - PLUG_TAPER_H

STEM_H  = BODY_H/2
STEM_R1 = 1.5*PLUG_R
STEM_R2 = PLUG_R

#let's add up the longest dimesion to choose a good voxel size based on the specified RES
ENV.voxel_size = (BODY_H + PLUG_H + PLUG_TAPER_H)/RES
NUDGE_L = 1.75*ENV.voxel_size #used for connecting volumes



gc = GyroidCube(GC_SIZE,center=True,lattice_param=0.75,thresh1=-0.2,thresh2=0.2).translate([0,0,GC_SIZE/2.0]) #center on X, Y, not Z
body = Cylinder(h=BODY_H,r=BODY_R) & gc                                    #intersection of gyroid and cylinder
stem = Cylinder(h=STEM_H,r1=STEM_R1,r2=STEM_R2).translate([0,0,BODY_H/2+NUDGE_L])
plug = Cylinder(h=PLUG_H,r=PLUG_R).translate([0,0,BODY_H])
plug_taper = Cylinder(h=PLUG_TAPER_H,r1=PLUG_TAPER_R1,r2=PLUG_TAPER_R2).translate([0,0,BODY_H + PLUG_H - NUDGE_L])

# #combine part components with union
# print("Combining components...")
model = plug_taper | plug | stem | body  #righthand side is the bottom of the Z stack

def export(filename = "gyroid_electrode_support_crimp_conn.stl", show=False):
    print("Rendering surface model...")
    model_surf = model.render_surface_mesh(
        smooth_iters = SMOOTH_ITERS,
        downscale_times = DOWNSAMPLE_TIMES,
        only_largest_component = True,
    )
    model_surf.save(filename)
    if show:
        model_surf.plot(color='white',show_edges=True)
