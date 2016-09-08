"""Module for dealing with whiski data.


This directory, or one like it, should be on sys.path:
/home/chris/Downloads/whisk-1.1.0d-Linux/share/whisk/python

It's not ideal because that directory contains a lot of files with common
names (and no __init__.py), so probably put it on the end of the path.
"""

# Import whiski files from where they live.
# This will trigger a load of default.parameters, always in the directory
# in which the calling script lives, NOT the module directory.
# How to fix that??
try:
    import traj
    import trace
except ImportError:
    pass

import db

# Import the functions for analyzing data
from base import *


try:
    import output_video
except ImportError:
    pass
