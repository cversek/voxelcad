#!/usr/bin/python
"""   
desc:  Setup script for 'voxelcad' package.
auth:  Craig Wm. Versek (cversek@gmail.com)
date:  2021-11-10
notes: Install with "python setup.py install".
"""
import platform, os, shutil, glob
from setuptools import setup, find_packages, Extension

#make safe for Python 3
try:
    raw_input
except NameError:
    raw_input = input

PACKAGE_METADATA = {
    'name'         : 'voxelcad',
    'version'      : 'dev',
    'author'       : "Craig Wm. Versek",
    'author_email' : "cversek@gmail.com",
}
    
PACKAGE_SOURCE_DIR = 'src'
MAIN_PACKAGE_DIR   = 'voxelcad'
MAIN_PACKAGE_PATH  = os.path.abspath(os.sep.join((PACKAGE_SOURCE_DIR,MAIN_PACKAGE_DIR)))

#dependencies
INSTALL_REQUIRES = [
                    'numpy >= 1.1.0',
                    'matplotlib >= 0.98',
                    ]

#scripts and plugins
ENTRY_POINTS =  { 'gui_scripts':     [
                                     ],
                  'console_scripts': [
                                      'voxelcad_shell  = voxelcad.scripts.shell:main',
                                     ],
                }


if __name__ == "__main__":
    #complete the setup using setuptools
    setup(package_dir      = {'':PACKAGE_SOURCE_DIR},
          packages         = find_packages(PACKAGE_SOURCE_DIR),
          entry_points     = ENTRY_POINTS,
          #non-code files
          package_data     =   {'': ['*.kv']},
          include_package_data = True,
          **PACKAGE_METADATA
         )
