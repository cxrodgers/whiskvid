import os
import pandas

class CalculationHandler(object):
    """Generic object for handling results of a calculation
    
    Derived objects must set these attributes:
        _name
        _db_field_path
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