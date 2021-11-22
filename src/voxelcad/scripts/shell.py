################################################################################
import logging, sys, argparse
LOGGER = logging.getLogger(__name__)

import IPython

################################################################################
__BANNER  = ['*'*80,
             '* VoxelCAD Shell',
             '*     author: cversek@gmail.com',
             '*'*80]
__BANNER  = '\n'.join(__BANNER)


def launch_shell(debug=False):
    try:
        print(__BANNER)
        #configure and pop open ipython terminal
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            LOGGER.debug("DEBUG MODE ON")

        else:
            logging.basicConfig(level=logging.INFO)
               
        from traitlets.config.loader import Config
        cfg = Config()
        #choosing a dedicate profile separates history and config from other IPython sessions
        cfg.BaseIPythonApplication.profile='voxelcad_shell' 
        cfg.InteractiveShellApp.exec_lines = [
             #"%pylab tk",
             "import voxelcad.environment as ENV",
             "from voxelcad.cube import Cube",
             "from voxelcad.sphere import Sphere",
             "from voxelcad.cylinder import Cylinder",
             "from voxelcad.gyroid_cube import GyroidCube, WigglyGyroidCube",
             "from voxelcad.voxel_model import union_all",
        ]
        IPython.start_ipython(argv=['--pylab'],config = cfg)
    finally:
        LOGGER.setLevel(logging.DEBUG)
        
###############################################################################
# Main
from voxelcad.errors import handleCrash

@handleCrash
def main():
    #---------------------------------------------------------------------------
    #parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
                        dest    = "debug",
                        default = False,
                        action  = 'store_true',
                        help    = "trigger debugging mode"
                       )
    args = parser.parse_args()
    launch_shell(debug=args.debug)