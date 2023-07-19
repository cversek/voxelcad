from voxelcad.gyroid_cube import GyroidCube
from voxelcad.cylinder import Cylinder

RES = 64

BODY_H = 5
BODY_R = 3

BASE_H = 1
BASE_R = BODY_R

NECK_H = 4
NECK_R = 1

MOLD_R = BODY_R + 1
MOLD_H = BODY_H + NECK_H - 5

gc = GyroidCube(6,res=RES,center=True,lattice_param=1.0,thresh1=0.75,thresh2=3.0).translate([0,0,BODY_H/2.0])

body = Cylinder(h=BODY_H,r=BODY_R,res=RES) & gc
base = Cylinder(h=BASE_H,r=BASE_R,res=RES).translate([0,0,BODY_H-0.1])
neck = Cylinder(h=NECK_H,r=NECK_R,res=RES).translate([0,0,BODY_H+BASE_H-0.1])


model = (base | neck) | body 

mold = Cylinder(h=MOLD_H,r=MOLD_R,res=RES) - model.translate([0,0,-2])

#model.export("model.stl") #makes a gyroid cube, intersects with cylinder, exports as STL
