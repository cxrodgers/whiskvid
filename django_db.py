"""Module for interacting with django whisker video database."""

import os
import pandas
import numpy as np
import django
import sys
import whisk_video

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


class CalculationHandler(object):
    """Generic object for handling results of a calculation
    
    Derived objects must set these attributes:
        _name
        _db_field_path
        _default_filename
    """
    def __init__(self, video_session):
        """Initalize a new handler for this video session"""
        self.video_session = video_session
        
    @property
    def get_path(self):
        """Full path to file using session directory and database field
        
        Raises ValueError if the field is not set
        Raises IOError if the field is set but the file does not exist
        
        Returns: full path to file
        """
        # Refresh db
        # This may not really be necessary if it causes slow downs
        self.video_session._django_object.refresh_from_db()
        
        # Get the filename from the db
        short_filename = getattr(self.video_session._django_object,
            self._db_field_path)
        
        # Test if null
        if short_filename is None or short_filename == '':
            raise ValueError("%s is not set in db for %s" % (
                self._db_field_path, str(self.video_session)))
            return None
        
        # Raise exception if file doesn't exist?
        full_filename = os.path.join(
            self.video_session.get_path.session,
            short_filename,
        )
        if not os.path.exists(full_filename):
            raise IOError("file does not exist: %s" % full_filename)
        
        # Return
        return full_filename
    
    @property
    def new_path(self):
        """Generate a new filename
        
        This will not be a full path, but a short filename, suitable
        for setting the database with.
        """
        return self._default_filename

    def set_path(self, new_path=None):
        """Set file in database
        
        By default it is set to the default value given by self.new_path
        If another path is desired, specify `new_path`. It should be a
        short filename, not a full path.
        
        Triggers a database save.
        
        Returns: new_path
        """
        if new_path is None:
            new_path = self.new_path
        
        setattr(self.video_session._django_object, self._db_field_path, 
            new_path)
        self.video_session._django_object.save()
        
        return new_path

    def load_data(self):
        """Read data or raise IOError"""
        filename = self.get_path
        try:
            data = pandas.read_pickle(filename)
        except IOError:
            raise IOError("cannot read pickle at %s" % filename)
        return data        

    def calculate(self, **kwargs):
        pass
    
    def set_manual_params_if_needed(self, **kwargs):
        pass

class Tac(CalculationHandler):
    _db_field_path = 'tac_filename'
    _name = 'tac'
    _default_filename = 'tac'
    
    def calculate(self, **kwargs):
        pass
    
    def set_manual_params_if_needed(self, **kwargs):
        pass
    
    

class NeuralPathJoiner(object):
    def __init__(self, neural_session, root_directory=None):
        # Figure out which root directory we're using
        if root_directory is None:
            # Try each root directory in turn
            for try_root_dir in neural_root_directory_search_list:
                self._root_directory = try_root_dir
                self._session_directory = os.path.join(self._root_directory,
                    neural_session._field_name)
                
                if os.path.exists(self._session_directory):
                    break
        else:
            # Use the requested root directory
            self._root_directory = try_root_dir
            self._session_directory = os.path.join(self._root_directory,
                neural_session._field_name)            
        
        # Ensure we found the session
        if not os.path.exists(self._session_directory):
            raise IOError("cannot find session directory at %s" %
                self._session_directory)
        
        self._neural_session = neural_session
        self._neural_session_name = neural_session.name
    
    @property
    def session(self):
        return self._session_directory
    
    @property
    def sort(self):
        return os.path.join(self._session_directory, 
            self._neural_session.sort_name)
    
    @property
    def kwik(self):
        return os.path.join(self.sort, self._neural_session._field_kwik_filename)

    @property
    def kwx(self):
        return os.path.join(self.sort, self._neural_session._field_kwx_filename)

class PathJoiner(object):
    """Joins paths to data on disk relating to a video session.
    
    This just deals with generating the appropriate root and session
    directory. Something else will have to generate the appropriate filenames
    from scratch when they don't yet exist.
    """
    def __init__(self, video_session, root_directory=None):
        # Figure out which root directory we're using
        if root_directory is None:
            # Try each root directory in turn
            for try_root_dir in root_directory_search_list:
                self._root_directory = try_root_dir
                self._session_directory = os.path.join(self._root_directory,
                    video_session.name)
                
                if os.path.exists(self._session_directory):
                    break
        else:
            # Use the requested root directory
            self._root_directory = try_root_dir
            self._session_directory = os.path.join(self._root_directory,
                video_session.name)            
        
        # Ensure we found the session
        if not os.path.exists(self._session_directory):
            raise IOError("cannot find session directory at %s" %
                self._session_directory)
        
        self._video_session = video_session
        self._video_session_name = video_session.name
    
    @property
    def session(self):
        return self._session_directory
    
    @property
    def tac(self):
        return os.path.join(self._session_directory, 
            self._video_session._field_tac_filename)

    @property
    def all_edges(self):
        return os.path.join(self._session_directory, 
            self._video_session._field_all_edges_filename)

    @property
    def edge_summary(self):
        return os.path.join(self._session_directory, 
            self._video_session._field_edge_summary_filename)

    @property
    def video_monitor(self):
        return os.path.join(self._session_directory,    
            self._video_session._field_monitor_video)
    
    @property
    def bsession_logfilename_copied(self):
        return os.path.join(self._session_directory,    
            self._video_session._field_bsession_logfilename)
    
    # Filenames that need to be put into the database
    # So that we can easily check which need to be generated
    # For now they are hard-coded
    @property
    def masked_whisker_ends(self):
        return os.path.join(self._session_directory, 'masked_whisker_ends')
    
    @property
    def clustered_tac(self):
        return os.path.join(self._session_directory, 'clustered_tac')
    
    @property
    def colorized_whisker_ends(self):
        return os.path.join(self._session_directory, 'colorized_whisker_ends')
    
    @property
    def contacts_summary(self):
        return os.path.join(self._session_directory, 'contacts_summary')
    
    @property
    def video_tracked_whiskers(self):
        return os.path.join(self._session_directory,    
            '%s.tracked_whiskers.mkv' % self._video_session_name)

    @property
    def video_colorized_whiskers(self):
        return os.path.join(self._session_directory,    
            '%s_colorized_video.mp4' % self._video_session_name)


class DataLoader(object):
    """Loads data from disk"""
    def __init__(self, video_session):
        self._video_session = video_session
    
    def _pandas_read_pickle(self, attr_name):
        """Generic function that tries to read a pickle using pathjoiner"""
        filename = getattr(self._video_session.get_path, attr_name)
        try:
            data = pandas.read_pickle(filename)
        except IOError:
            raise IOError("cannot read pickle for %s at %s" % (
                attr_name, filename))
        return data
    
    @property
    def tac(self):
        return self._pandas_read_pickle('tac')
    
    @property
    def all_edges(self):
        filename = self._video_session.get_path.all_edges
        try:
            data = np.load(filename)
        except IOError:
            raise IOError("no all_edges found at %s" % filename)
        return data
    
    @property
    def edge_summary(self):
        return self._pandas_read_pickle('edge_summary')

    @property
    def masked_whisker_ends(self):
        return self._pandas_read_pickle('masked_whisker_ends')

    @property
    def clustered_tac(self):
        return self._pandas_read_pickle('clustered_tac')
  
    @property
    def colorized_whisker_ends(self):
        return self._pandas_read_pickle('colorized_whisker_ends')
    
    @property
    def contacts_summary(self):
        return self._pandas_read_pickle('contacts_summary')


class VideoSession(object):
    """Interface to all of the data about a video session.
    
    It is initialized from a database object from whisk_video.models.
    Each field is available as a hidden attribute beginning with "_field"
    
    The preferred way to access data stored on disk are as follows:
        get_path : filenames to data stored on disk
            Will return null ('') if no filename stored in database
        load_data : loading methods for data stored on disk
            Will raise IOError if fails to load
    """
    def __init__(self, django_object):
        # Store a reference to the django_object
        self._django_object = django_object

        # _field_bsession is a django object, for now
        # at least extract the name
        try:
            self.bsession_name = self._field_bsession.name
        except AttributeError:
            self.bsession_name = None

        # PathJoiner (uses simple properties like name)
        self.get_path = PathJoiner(self)
        
        # DataLoader (uses self.get_path)
        self.load_data = DataLoader(self)
        
        # Install the handlers
        self.tac = Tac(self)
        # self.get_path.tac = self.tac.get_path
        # self.data_load.tac = self.tac.data_load

    ## Initializing shortcut
    @classmethod
    def from_name(self, name):
        """Query database for session name and initialize VideoSession"""
        # Load django object
        django_vsession = whisk_video.models.VideoSession.objects.filter(
            name=name).first()
        
        # Initialize object from that
        return VideoSession(django_vsession)
    
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
        
    @property
    def fit_b2v(self):
        return np.array([
            self._field_fit_b2v0,
            self._field_fit_b2v1,
        ])

    @property
    def fit_v2b(self):
        return np.array([
            self._field_fit_v2b0,
            self._field_fit_v2b1,
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