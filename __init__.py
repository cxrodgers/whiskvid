"""Module for dealing with whiski data.


This directory, or one like it, should be on sys.path:
/home/chris/Downloads/whisk-1.1.0d-Linux/share/whisk/python

It's not ideal because that directory contains a lot of files with common
names (and no __init__.py), so probably put it on the end of the path.
"""
import os
import sys
import django

## Load the interface with the django database
def setup_django():
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
    django_project_path = os.path.expanduser(
        '~/Dropbox/django/mouse2')
    if django_project_path not in sys.path:
        sys.path.append(django_project_path)

    # Set environment variable
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mouse2.settings")
    
    # Setup django
    django.setup()

setup_django()

# Now we can import the django modules
import whisk_video
import django_db


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
