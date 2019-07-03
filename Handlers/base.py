import pandas
import os
import datetime
import shutil

class Error(Exception):
    """Base class for exceptions in this module"""
    pass

class FieldNotSetError(Error):
    """When we try to get_path but the path hasn't been set"""
    def __init__(self, field_name, vsession_s):
        message = '%s / %s: field is unset' % (
            vsession_s, field_name)
        super(FieldNotSetError, self).__init__(message)

class FileDoesNotExistError(Error):
    """When the field is set but the file doesn't exist"""
    def __init__(self, field_name, full_filename, vsession_s):
        message = '%s / %s: file does not exist at %s' % (
            vsession_s, field_name, full_filename)
        super(FileDoesNotExistError, self).__init__(message)

class RequiredFieldsNotSetError(Error):
    """When we try to run a calculation but the prereqs are unset"""
    def __init__(self, handler):
        message = '%s: Some required fields are unset' % (
            handler._name,
        )
        super(FieldNotSetError, self).__init__(message)    

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
        
        This is the recommended way to check if the data is "available"
        without actually loading it.
        
        Raises FieldNotSetError if the field is not set
        Raises FileDoesNotExistError if the field is set but the file 
            does not exist
        
        Returns: full path to file
        """
        # Get the filename from the db
        short_filename = self._get_field()
        
        # Test if null
        if short_filename is None or short_filename == '':
            raise FieldNotSetError(self._db_field_path, str(self.video_session))
        
        # Raise exception if file doesn't exist?
        full_filename = os.path.join(self.video_session.session_path,
            short_filename)
        if not os.path.exists(full_filename):
            raise FileDoesNotExistError(self._db_field_path, full_filename,
                str(self.video_session))
        
        # Return
        return full_filename
    
    @property
    def new_path(self):
        """Generate a new filename
        
        This will not be a full path, but a short filename, suitable
        for setting the database with. By default we use self._name,
        but this can be overridden if desired.
        """
        return self._name
    
    @property
    def new_path_full(self):
        """Full new path including session directory"""
        return os.path.join(self.video_session.session_path,
            self.new_path)

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
            new_path = self.new_path_full

            # If that file exists, then set it
            if os.path.exists(new_path):
                self.set_path()
            
                # This should now work
                return self.get_path

    def load_data(self, filename=None):
        """Read data or raise IOError
        
        filename : Use this to override the normal filename
            If None, uses self.get_path
        
        This implementation uses pandas.read_pickle, as does save_data.
        Override if another method is desired.
        """
        if filename is None:
            filename = self.get_path
        
        try:
            data = pandas.read_pickle(filename)
        except (KeyError, IOError):
            # KeyError sometimes raise when unpickling a non-pickle
            raise IOError("cannot read pickle at %s" % filename)
        return data   
    
    def save_data(self, data, clobber_action='backup', clobber_warn=True):
        """Save data to disk and set the database path.
        
        Saves data to the location specified in self.new_path. Then calls
        self.set_path() to set the database field. Finally return
        self.get_path, which should be the new full filename, and will
        raise FileNotFoundError if something went wrong.
        
        This implementation uses pandas.to_pickle. Override if another
        method is desired.
        
        
        data : the data to save

        clobber_action : string
            This only matters if the file already exists.
            If 'error' : raise exception
            If 'backup' : backup the old file
            If 'clobber' : overwrite the old file

        clobber_warn : boolean
            If False, silently performs clobber_action
            If True, print warning before performing clobber_action
        
        
        Returns: string, full path to written file
        """
        filename = self.new_path_full
        
        # Determine if we are clobbering
        if os.path.exists(filename):
            if clobber_action == 'error':
                raise IOError("file %s already exists" % filename)
            
            elif clobber_action == 'backup':
                # Generate backup filename
                now_string = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                backup_filename = filename + '.%s.backup' % now_string
                
                # Warn
                if clobber_warn:
                    print "warning: backing up %s" % backup_filename
                
                # Do the backup
                shutil.copyfile(filename, backup_filename)
                
            elif clobber_action == 'clobber':
                # Warn
                if clobber_warn:
                    print "warning: clobbering %s" % filename
            
            else:
                raise ValueError("unknown clobber action: %s" % clobber_action)
        
        # Save
        try:
            data.to_pickle(filename)
        except AttributeError:
            raise ValueError("data cannot be pickled")
        except IOError:
            raise IOError("cannot write pickle at %s" % filename)
        
        # Set path
        self.set_path()

        # This should now work
        return self.get_path

    def calculate(self, **kwargs):
        pass
    
    def set_manual_params_if_needed(self, **kwargs):
        pass

    def _check_if_manual_params_set(self):
        """Returns True if all manual params are already set in the db.
        
        These are the manual params that this Handler is capable of setting,
        not the ones that are required for it to run (if any).
        """
        all_params_set = True
        if hasattr(self, '_manual_param_fields'):
            for attr in self._manual_param_fields:
                if pandas.isnull(getattr(
                    self.video_session._django_object, attr)): 
                    all_params_set = False
        return all_params_set
    
    def _check_if_required_fields_for_calculate_set(self):
        """Returns True if _required_fields_for_calculate are all set"""
        all_params_set = True
        if hasattr(self, '_required_fields_for_calculate'):
            for attr in self._required_fields_for_calculate:
                if pandas.isnull(getattr(
                    self.video_session._django_object, attr)): 
                    all_params_set = False
        return all_params_set     