"""Module for dealing with whiski data.

In order to be able to trace:
    This directory, or one like it, should be on sys.path:
    /home/chris/Downloads/whisk-1.1.0d-Linux/share/whisk/python

    It's not ideal because that directory contains a lot of files with common
    names (and no __init__.py), so probably put it on the end of the path.

Importing this module will automatically trigger a linkage to the
mouse-cloud django project. This is done by:
* Adding ~/dev/mouse-cloud to sys.path
* Setting the environment variable DJANGO_SETTINGS_MODULE to mouse2.settings
* Calling django.setup()

It would probably be better to move the above linkage to a distinct
module that specifically handles the behavioral and video databases.

The django module whisk_video is then imported, which will trigger
a call to local_settings, which will set DATABASE_URL. So the database 
we connect to depends on the branch currently active in the 
django project path.
"""
import os
import sys
import django

## Load the interface with the django database
def _setup_django():
    """Imports django settings from mouse2 project
    
    http://stackoverflow.com/questions/8047204/django-script-to-access-model-objects-without-using-manage-py-shell
    
    In the long run it may be better to rewrite this using sqlalchemy
    and direct URLs to the database. For one thing, we are dependent on
    the branch currently in place on the Dropbox.
    """
    # Test if this trick has already been used
    dsm_val = os.environ.get('DJANGO_SETTINGS_MODULE')    
    if dsm_val == 'mouse2.settings':
        # Seems to already have been done. Maybe we are running from manage.py
        #~ print "warning: DJANGO_SETTINGS_MODULE already set"
        return
    if dsm_val is not None:
        # Already set to some other module. This will not work
        raise ValueError("DJANGO_SETTINGS_MODULE already set to %s" % dsm_val)
    
    # Add to path
    django_project_path = os.path.expanduser('~/dev/mouse-cloud')
    if django_project_path not in sys.path:
        sys.path.append(django_project_path)

    # Set environment variable
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mouse2.settings")
    
    # Setup django
    django.setup()

try:
    _setup_django()
except ImportError:
    # this happens if mouse-cloud doesn't exist at the expected pat
    pass

# Now we can import the django modules
try:
    import whisk_video
    import django_db
except ImportError:
    pass

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

import plotting


try:
    import output_video
except ImportError:
    pass
