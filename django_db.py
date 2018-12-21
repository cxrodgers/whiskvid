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
import MCwatch.behavior
import my.misc

# for video_range_bbase
import my.video

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
        for handler_class in [
            TacHandler, 
            AllEdgesHandler, 
            EdgeSummaryHandler,
            MonitorVideoHandler, 
            ClusteredTacHandler, 
            ColorizedWhiskerEndsHandler,
            ContactsSummaryHandler,
            VideoTrackedWhiskersHandler,
            WhiskersTableHandler, 
            ColorizedContactsSummaryHandler,
            ColorizationKeystoneInfoHandler,
            ColorizationCuratedNum2Name,
            ColorizationCurated,
            ColorizationPredictions,
            ColorizationHeldoutResults,
            ColorizationRepairedResults,
            CWE_with_kappa,
            FrameMeanLuminances,
            ]:
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
        
        if django_vsession is None:
            raise ValueError(
                "cannot find django VideoSession with name %s" % name)
        
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
        """Get name of bsession. Often used to get trial matrix."""
        #return self._django_object.bsession.name
        return self._django_object.grand_session.session.name
    
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
    
    @property
    def whisker_order(self):
        """List of whiskers in anatomical order"""
        whiskers = self._django_object.whisker_colors.split()
        
        # Sort: first lowercase greeks, then uppers, then '?'
        lowers, uppers = [], []
        for whisker in sorted(whiskers):
            if whisker.islower():
                lowers.append(whisker)
            else:
                uppers.append(whisker)
        res = lowers + uppers + ['?']
        
        return res
    
    @property
    def whiskers_and_colors(self):
        """Try to replace color2whisker and whisker_order with this
        
        Result is something like this:
           color_group whisker disp_color_k disp_color_w  anatomical_order
        0            0       ?            k        white                 4
        1            1       b            b            b                 0
        2            2      C1            g            g                 1
        3            3      C2            r            r                 2
        4            4      C3            c            c                 3
        """
        # Form a dataframe of whisker names, indexed by color_group
        whisker_color_l = self._django_object.whisker_colors.split()
        res = pandas.DataFrame.from_dict({
            'whisker': ['?'] + whisker_color_l,
            'color_group': range(0, len(whisker_color_l) + 1)
            })
        
        # Add 'disp_color_k' and 'disp_color_w' colummns
        res['disp_color_k'] = whiskvid.WHISKER_COLOR_ORDER_K[:len(res)]
        res['disp_color_w'] = whiskvid.WHISKER_COLOR_ORDER_W[:len(res)]
        
        # Add an anatomical_order column
        # Sort: first lowercase greeks, then uppers
        lowers, uppers = [], []
        for whisker in sorted(whisker_color_l):
            if whisker.islower():
                lowers.append(whisker)
            else:
                uppers.append(whisker)
        anatomical_order = lowers + uppers + ['?']
        res['anatomical_order'] = [
            anatomical_order.index(wname) for wname in res['whisker']]
        
        return res
    
    @property
    def video_range_bbase(self):
        """Return the start and stop times of the video in behavioral timebase.
        
        Gets the duration of the monitor video and uses self.fit_v2b to
        convert.
        
        Returns: array of (start, stop) times
        """
        # Get duration of monitor video in spurious timebase
        video_duration_s = my.video.get_video_duration2(
            self.data.monitor_video.get_path)
        
        # Convert to bbase
        video_range_bbase = np.polyval(self.fit_v2b, [0, video_duration_s])    
        
        return video_range_bbase
    
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
        # This should be done at the grand session level now
        1/0
        
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
        
        # Problem is that the below code needs the MCwatch.behavior magic to
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
    
    def calculate_sync(self, light_delta=30, diffsize=2, refrac=50, 
        verbose=False, force=False):
        """Sync the behavior file with the monitor video
        
        The actual syncing is done by 
          MCwatch.behavior.syncing.sync_video_with_behavior
        
        Requires self.bsession_name and lums. If lums does not exist,
        it will be calculated, which requires monitor_video.
        Sets these fields:
            fit_v2b0, fit_b2v1, fit_v2b0, fit_v2b1
        
        verbose : print out frame number for every chunk as it is processed
        force : bool
            If True and the sync already exists, return immediately
            Else it will be calculated
            'force' does not propagate to calculation of lums
        """
        # Skip if it already exists
        already_done = (
            (self._django_object.fit_v2b0 is not None) and
            (self._django_object.fit_b2v1 is not None) and
            (self._django_object.fit_v2b0 is not None) and
            (self._django_object.fit_v2b1 is not None)
        )
        if not force and already_done:
            return

        # Get trial matrix
        trial_matrix = MCwatch.behavior.db.get_trial_matrix(
            self.bsession_name, False)
        
        # Calculate lums if they don't exist
        try:
            self.data.lums.get_path
        except IOError:
            print "calculating lums"
            self.data.lums.calculate(verbose=verbose)
        
        # Load lums
        lums = self.data.lums.load_data()
        
        # Sync it
        sync_res = MCwatch.behavior.syncing.sync_video_with_behavior(
            trial_matrix=trial_matrix,
            lums=lums, 
            light_delta=light_delta,
            diffsize=diffsize, 
            refrac=refrac, 
            assumed_fps=30.,
            error_if_no_fit=False,
            verbose=verbose,
            return_all_data=True,
            refit_data=True,
        )
        res = sync_res['b2v_fit']
        
        # Warning if no fit found
        if res is None:
            print "warning: no fit found"
            return
        
        
        ## Set sync
        self._django_object.fit_b2v0 = res[0]
        self._django_object.fit_b2v1 = res[1]

        fit_v2b = my.misc.invert_linear_poly(res)
        self._django_object.fit_v2b0 = fit_v2b[0]
        self._django_object.fit_v2b1 = fit_v2b[1]
    
        # Save to db
        self._django_object.save()

def identify_sync_problems(sync_res):
    """Function to identify major syncing issues, eg camera reboot
    
    sync_res : result of MCwatch.behavior.syncing.sync_video_with_behavior
    
    Identifies which trials are off by more than 20 ms
    Right now this can only handle the case where all the trials are good,
    or the camera fails after the midpoint and the later trials are bad
    
    Returns : first_bad_trial, max_good_vtime
        These are both None if there is no problem.
    """
    ## Identify sync problems
    b2v_fit = sync_res['b2v_fit']
    lums = sync_res['lums']
    video_flash_x = sync_res['video_flash_x']
    behavior_flash_y = sync_res['behavior_flash_y']

    # Do the matching to identify where in the video the sync fails (if anywhere)
    # This is the index of the first btrial in the video.
    # If this is negative then there are extra flashes in the video (e.g., from
    # another session.)
    b_trial_start = sync_res['y_start'] - sync_res['x_start']
    assert b_trial_start >= 0

    # The last btrial might be nan, but nothing else should be
    last_btrial_is_nan = np.isnan(behavior_flash_y[-1])
    if last_btrial_is_nan:
        assert not np.any(np.isnan(behavior_flash_y[:-1]))
    else:
        assert not np.any(np.isnan(behavior_flash_y))

    # Match the trials and fit
    # Really should refit here, in case not all the data was used originally
    # Because longest_unique_fit adds trials symetrically at beginning and end
    btimes_matched = behavior_flash_y[b_trial_start:b_trial_start + len(video_flash_x)]
    btimes_matched_vbase = np.polyval(sync_res['b2v_fit'], btimes_matched)
    residuals = btimes_matched_vbase - video_flash_x

    # Convert residuals to real seconds, not spurious timebase
    residuals = residuals / sync_res['b2v_fit'][0]

    # Identify the bad trials
    # 20 ms errors are not uncommon!
    bad_trials = np.abs(residuals) > .02

    # The only cases that are handled right now are:
    # All trials good
    # Trials good until some point, then all trials bad afterward (camera reboot)
    if np.all(~bad_trials):
        # all trials good
        first_bad_trial = None
        max_good_vtime = None
    else:
        # camera reboot after the midpoint
        first_bad_trial = np.where(bad_trials)[0][0]
        assert np.all(bad_trials[first_bad_trial:])
        
        # the maximum good time is the last good trial start
        max_good_vtime = btimes_matched_vbase[first_bad_trial-1]        
    
    return first_bad_trial, max_good_vtime

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