################################################################################
import logging, sys, argparse

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
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
               
        from traitlets.config.loader import Config
        cfg = Config()
        cfg.InteractiveShellApp.exec_lines = [
             #"%pylab tk",
             "from voxelcad.sphere import Sphere",
             "from voxelcad.cube import Cube",
             "from voxelcad.gyroid_cube import GyroidCube",
        ]
        IPython.start_ipython(argv=['--pylab'],config = cfg)
    finally:
        logging.getLogger().setLevel(logging.DEBUG)
        
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