"""Module for interacting with django whisker video database."""

import os
import pandas
import numpy as np
import django
import sys
import whisk_video
from Handlers import *

# Search in the following order for the session
root_directory_search_list = [
    '/mnt/fast/data/whisker/processed',
    '/home/chris/whisker_video',
    os.path.expanduser('~/mnt/nas2_home/whisker/processed'),
]
neural_root_directory_search_list = [
    '/mnt/fast/data/neural',
    '/home/chris/data',
    os.path.expanduser('~/mnt/nas2_home/neural'),
]

class HandlerHolder(object):
    pass

class VideoSession(object):
    """Interface to all of the data about a video session.
    
    It is initialized from a database object from whisk_video.models.
    """
    def __init__(self, django_object, forced_root_directory=None):
        """Initialize a new video session from database object
        
        forced_root_directory : force root to be in a certain location
            Otherwise the search tree is followed
        """
        # Store a reference to the django_object
        self._django_object = django_object

        # Set the session directory
        self._set_session_path(forced_root_directory)
        
        # Install the handlers
        self.data = HandlerHolder()
        for handler_class in [TacHandler, AllEdgesHandler, EdgeSummaryHandler,
            MonitorVideoHandler, ClusteredTacHandler, 
            ColorizedWhiskerEndsHandler,
            ContactsSummaryHandler, MaskedWhiskerEndsHandler, 
            VideoColorizedWhiskersHandler, VideoTrackedWhiskersHandler,
            WhiskersTableHandler]:
            # Init the handler
            handler = handler_class(self)
            
            if hasattr(self.data, handler._name):
                raise ValueError("handler %s already installed" % handler._name)
            
            # Store
            setattr(self.data, handler._name, handler)

    ## Initializing shortcut
    @classmethod
    def from_name(self, name, forced_root_directory=None, **kwargs):
        """Query database for session name and initialize VideoSession"""
        # Load django object
        django_vsession = whisk_video.models.VideoSession.objects.filter(
            name=name).first()
        
        # Initialize object from that
        return VideoSession(django_vsession, 
            forced_root_directory=forced_root_directory, **kwargs)

    ## Link to appropriate session directory
    def _set_session_path(self, forced_root_directory=None):
        """Find the appropriate session directory
        
        Tries each path in the search list. Usually this is just called
        once during initialization.
        """
        # Depends on if we are forcing a root directory
        if forced_root_directory is None:
            # Try each root directory in turn
            for try_root_dir in root_directory_search_list:
                # Try the next one
                self._root_directory = try_root_dir
                self._session_directory = os.path.join(self._root_directory,
                    self.name)
                
                # Found it, break
                if os.path.exists(self._session_directory):
                    break
        else:
            # Use the requested root directory
            self._root_directory = forced_root_directory
            self._session_directory = os.path.join(self._root_directory,
                self.name)

        # Ensure we found the session
        if not os.path.exists(self._session_directory):
            raise IOError("cannot find session directory at %s" %
                self._session_directory)        
    
    @property
    def session_path(self):
        """Path to the directory containing all session files"""
        return self._session_directory

    ## Shortcut accessor methods to commonly used fields in django object
    # Other properties may be accessed like
    # vs._django_object.field_name
    @property
    def frame_height(self):
        return self._django_object.frame_height
    
    @property
    def frame_width(self):
        return self._django_object.frame_width

    @property
    def name(self):
        return self._django_object.name
    
    @property
    def frame_rate(self):
        return self._django_object.frame_rate
    
    # Other accessors that are not simple short cuts
    @property
    def bsession_name(self):
        return self._django_object.bsession.name
    
    @property
    def fit_b2v(self):
        return np.array([
            self._django_object.fit_b2v0,
            self._django_object.fit_b2v1,
        ])

    @property
    def fit_v2b(self):
        return np.array([
            self._django_object.fit_v2b0,
            self._django_object.fit_v2b1,
        ])
    
    @property
    def fol_range_x(self):
        return np.array([
            self._django_object.param_fol_x0,
            self._django_object.param_fol_x1,
        ])

    @property
    def fol_range_y(self):
        return np.array([
            self._django_object.param_fol_y0,
            self._django_object.param_fol_y1,
        ])

class NeuralSession(object):
    """Interface to all of the data about a neural session.
    
    It is initialized from a database object from neural_sessions.models.
    Each field is available as a hidden attribute beginning with "_field"
    
    The preferred way to access data stored on disk are as follows:
        get_path : filenames to data stored on disk
            Will return null ('') if no filename stored in database
        load_data : loading methods for data stored on disk
            Will raise IOError if fails to load
    """
    def __init__(self, django_object):
        # Copy values
        for field in django_object._meta.fields:
            setattr(self, '_field_' + field.name, 
                getattr(django_object, field.name))
        
        # Unhide simple properties
        simple_properties_l = ['name', 'sort_name',]
        for simple_property in simple_properties_l:
            try:
                value = getattr(self, '_field_' + simple_property)
            except AttributeError:
                continue
            setattr(self, simple_property, value)

        # _field_bsession is a django object, for now
        # at least extract the name
        try:
            self.bsession_name = self._field_bsession.name
        except AttributeError:
            self.bsession_name = None

        # PathJoiner (uses simple properties like name)
        self.get_path = NeuralPathJoiner(self)

    @property
    def fit_b2n(self):
        return np.array([
            self._field_fit_b2n0,
            self._field_fit_b2n1,
        ])

    @property
    def fit_n2b(self):
        return np.array([
            self._field_fit_n2b0,
            self._field_fit_n2b1,
        ])