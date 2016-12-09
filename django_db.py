"""Module for interacting with django whisker video database."""

import os
#~ import shutil
import datetime
import pandas
import numpy as np
import django
import sys
import whisk_video
from Handlers import *
import runner.models

# For syncing
import BeWatch


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
            WhiskersTableHandler, ColorizedContactsSummaryHandler]:
            # Init the handler
            handler = handler_class(self)
            
            if hasattr(self.data, handler._name):
                raise ValueError("handler %s already installed" % handler._name)
            
            # Store
            setattr(self.data, handler._name, handler)

    def __str__(self):
        return "Python VideoSession for %s" % self._django_object

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
    
    @property
    def color2whisker(self):
        whisker_color_l = self._django_object.whisker_colors.split()
        res = {0: '?'}
        for n, s in enumerate(whisker_color_l):
            res[n + 1] = s
        return res
    
    
    ## Other methods
    # These are things that don't make sense for a Handler
    # Not sure how to encapsulate them exactly
    def find_and_set_bsession(self, force=False):
        """Find the matching behavioral session
        
        By convention the name should be DATESTRING_MOUSENAME. We use
        these values to find the matching behavioral session and set
        the field 'bsession'.
        
        #~ We also copy the logfile into the video session and set the
        #~ field 'bsession_logfilename'. This is a bit redundant.
        """
        # Skip if it already exists
        if not force and self._django_object.bsession is not None:
            return
        
        # Try splitting into date and mouse
        sess_split = self.name.split('_')
        if len(sess_split) != 2:
            raise ValueError("cannot split session name: %s" % session)
        date_string, mouse_name = sess_split
        
        # Try to put the date_string to a date
        session_date = datetime.datetime.strptime(date_string, '%y%m%d').date()
        
        # Try to find a matching behavioral session in the database
        qs = runner.models.Session.objects.filter(
            date_time_start__date=session_date,
            mouse__name=mouse_name)
        if qs.count() != 1:
            raise ValueError("found 0 or 2+ matching behavioral sessions")
        bsession = qs.first()
        
        # Problem is that the below code needs the BeWatch magic to
        # set the path correctly for the current locale.
        #~ # Copy the behavioral file into the video directory
        #~ bfile = bsession.logfile
        #~ bfilename = os.path.split(bfile)[1]
        #~ new_bfile = os.path.join(self.session_path, bfilename)
        #~ print "copying %s to %s" % (bfile, new_bfile)        
        #~ shutil.copyfile(bfile, new_bfile)
        
        # Save the name of the behavioral session in the database
        self._django_object.bsession = bsession
        
        # Save the name of the behavioral file
        #~ self._django_object.bsession_logfilename = bfilename
        
        # Save to db
        self._django_object.save()
    
    def calculate_sync(self, light_delta=30, diffsize=2, refrac=50,):
        """Sync the behavior file with the monitor video
        
        Requires 'bsession' and 'monitor_video'
        Sets 'fit_b2v0' etc.
        """
        # Use BeWatch to get behavior file name locale-specific
        bdf = BeWatch.db.get_behavior_df()
        bfile = bdf.set_index('session').loc[self.bsession_name, 'filename']
        
        # Monitor video
        video_file = self.data.monitor_video.get_path
        
        # Sync it
        res = BeWatch.syncing.sync_video_with_behavior(
            bfile=bfile,
            lums=None, 
            video_file=video_file, 
            light_delta=light_delta,
            diffsize=diffsize, 
            refrac=refrac, 
            assumed_fps=30.,
            error_if_no_fit=True,
        )
        
        # Set sync
        1/0

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