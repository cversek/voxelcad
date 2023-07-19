from voxelcad.debug import currentframe, DEBUG_TAG, DEBUG_EMBED, LOGGER
LOGGER.setLevel("DEBUG")


from voxelcad import Cube, union_all
import voxelcad.environment as ENV






#let's add up the longest dimesion to choose a good voxel size based on the specified RES
ENV.voxel_size = 1
NUDGE_L = 1.75*ENV.voxel_size #used for connecting volumes

CUBE_SIZE = 10
Z_SPACING = 2*CUBE_SIZE
NUM = 5
M = [Cube(CUBE_SIZE).translate([0,0,i*Z_SPACING]) for i in range(NUM)]
M = union_all(M)
M.plot(color='white',show_edges=True)


