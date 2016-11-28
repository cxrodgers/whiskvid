import pandas
import os

class CalculationHandler(object):
    """Generic object for handling results of a calculation
    
    Derived objects must set these attributes:
        _name
        _db_field_path
    
    _db_field_path must be a CharField
    """
    def __init__(self, video_session):
        """Initalize a new handler for this video session"""
        self.video_session = video_session
    
    def __str__(self):
        return "%s handler for %s" % (self._name, str(self.video_session))
    
    def _get_field(self):
        """Return the value stored at _db_field_path"""
        # Refresh db
        # This may not really be necessary if it causes slow downs
        self.video_session._django_object.refresh_from_db()
        
        # Get the filename from the db
        short_filename = getattr(self.video_session._django_object,
            self._db_field_path)
        
        return short_filename
    
    @property
    def field_is_null(self):
        """Returns True if the value stored at _db_field_path is None or ''"""
        value = self._get_field()
        return value is None or value == ''
    
    @property
    def get_path(self):
        """Full path to file using session directory and database field
        
        Raises ValueError if the field is not set
        Raises IOError if the field is set but the file does not exist
        
        Returns: full path to file
        """
        # Get the filename from the db
        short_filename = self._get_field()
        
        # Test if null
        if short_filename is None or short_filename == '':
            raise ValueError("%s is not set in db for %s" % (
                self._db_field_path, str(self.video_session)))
        
        # Raise exception if file doesn't exist?
        full_filename = os.path.join(
            self.video_session.session_path,
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
        return self._name

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

    def set_path_if_exists(self):
        """If the field is null and new_path exists, set_path
        
        If the field is already set, nothing happens. Otherwise,
        we call new_path to find out what the path would be. Then we
        check if it exists on disk. If it does, then we set_path.
        """
        if self.field_is_null:
            # Generate the full new path
            new_path = os.path.join(self.video_session.session_path, 
                self.new_path)

            # If that file exists, then set it
            if os.path.exists(new_path):
                self.set_path()
            
                # This should now work
                return self.get_path

    def load_data(self):
        """Read data or raise IOError"""
        filename = self.get_path
        try:
            data = pandas.read_pickle(filename)
        except (KeyError, IOError):
            # KeyError sometimes raise when unpickling a non-pickle
            raise IOError("cannot read pickle at %s" % filename)
        return data        

    def calculate(self, **kwargs):
        pass
    
    def set_manual_params_if_needed(self, **kwargs):
        pass