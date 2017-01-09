"""Handles the calculating of touches and contacts"""
from base import *
import numpy as np
import pandas
import my
import BeWatch

class TacHandler(CalculationHandler):
    _db_field_path = 'tac_filename'
    _name = 'tac'

    # The params this handler helps set
    _manual_param_fields = (
        'param_fol_x0', 'param_fol_x1', 'param_fol_y0', 'param_fol_y1',
    )

    # The fields that are required before calculate can run
    _required_fields_for_calculate = (
        'monitor_video', 'all_edges_filename', 'param_face_side',
        'whiskers_table_filename',
    )

    def calculate(self, force=False, verbose=True, **kwargs):
        """Wrapper around calculate_contacts"""
        # Return if force=False and the data exists
        if not force:
            # Check if data available
            data_available = True
            warn_about_field = False
            try:
                self.get_path
            except FieldNotSetError:
                # not calculated yet
                data_available = False
            except FileDoesNotExistError:
                data_available = False
                warn_about_field = True
            
            # Return if it is
            if data_available:
                return
            
            # Warn if we couldn't load data but we were supposed to be able to
            if warn_about_field:
                print (("warning: %s was set " % self._db_field_path) + 
                    "but could not load data, recalculating" 
                )
        
        # We are going to try to calculate
        # Ensure required fields are set
        if not self._check_if_required_fields_for_calculate_set():
            raise RequiredFieldsNotSetError(self)

        # Load required data
        mwe = self.video_session.data.whiskers.load_data()
        edge_a = self.video_session.data.all_edges.load_data()
        
        # Calculate contacts and save
        tac = calculate_contacts(mwe, edge_a, verbose=verbose, **kwargs)
        self.save_data(tac)
    
    def set_manual_params_if_needed(self, force=False, 
        n_frames=4, interactive=True, **kwargs):
        """Display a subset of video frames to set fol_x and fol_y"""
        # Return if force=False and all params already set
        if not force and self._check_if_manual_params_set():
            return
        
        # Choose the follicle
        vfile = self.video_session.data.monitor_video.get_path
        print "Enter the ROI coordinates of the follicle."
        res = my.video.choose_rectangular_ROI(vfile, n_frames=n_frames, 
            interactive=interactive)

        # Shortcut
        vs_obj = self.video_session._django_object
    
        # Store
        vs_obj.param_fol_x0 = res['x0']
        vs_obj.param_fol_x1 = res['x1']
        vs_obj.param_fol_y0 = res['y0']
        vs_obj.param_fol_y1 = res['y1']
        vs_obj.save()

    def load_data(self, add_btime=False, add_trial_info=False, 
        min_t=None, max_t=None):
        """Load tac data and optionally process a bit
        
        add_btime : if True, loads sync and convert 'frame' to 'btime'
        
        add_trial_info : if True, associates each tac with a trial, and
            adds information about that trial. We associate based on
            the 'start_time' column of trial_matrix. 
            
            These columns are added: ['rwin_time', 'trial', 'rewside', 
                'outcome', 'choice_time', 'isrnd', 'choice', 'relative_t']
            
            'relative_t' is the time wrt the rwin_time for that trial,
            though this could easily be edited to be wrt choice_time.
            
            This implies add_btime = True
        
        min_t, max_t : discard tac rows with a 'relative_t' outside of
            this range. If both are None, nothing happens. If one is None,
            assumes the other is 0.
            
            This implies both add_btime and add_trial_info are True.
        """
        # Parent class to load the raw data
        tac = super(TacHandler, self).load_data()
        
        # Need btime if we want to add trials
        if add_trial_info or min_t is not None or max_t is not None:
            add_btime = True
        
        # Need trial info if we want to filter
        if min_t is not None or max_t is not None:
            add_trial_info = True
        
        # Optionally add btime
        if add_btime:
            # Get sync
            v2b_fit = self.video_session.fit_v2b

            # "vtime" is in the spurious 30fps timebase
            # the fits take this into account
            tac['vtime'] = tac['frame'] / 30.
            tac['btime'] = np.polyval(v2b_fit, tac['vtime'].values)
        
        # Optionally add trial labels
        if add_trial_info:
            # Get trial matrix
            bsession = self.video_session.bsession_name
            trial_matrix = BeWatch.db.get_trial_matrix(bsession, True)

            # Associate each trial with a tac
            tac['trial'] = trial_matrix.index[
                np.searchsorted(trial_matrix['start_time'].values, 
                    tac['btime'].values) - 1]    

            # Add rewside and outcome to tac
            tac = tac.join(trial_matrix[[
                'rewside', 'outcome', 'choice_time', 'isrnd', 'choice',
                'rwin_time',]], 
                on='trial')
            tac['relative_t'] = tac['btime'] - tac['rwin_time']
    
        # Optionally filter in a window around each trial
        if min_t is not None or max_t is not None:
            # Defaults
            if min_t is None:
                min_t = 0
            if max_t is None:
                max_t = 0
            
            # Filter
            tac = tac[
                (tac['relative_t'] > min_t) &
                (tac['relative_t'] <= max_t)
            ].copy()
        
        return tac

def calculate_contacts(mwe, edge_a, contact_dist_thresh=10, verbose=True):
    """Calculate whisker-shape contacts by proximity
    
    mwe : masked whisker ends
    edge_a : all edges
    contact_dist_thresh : distance to call a contact
    verbose : print frame number every 1000
    """
    # Find the contacts
    # For every frame, iterate through whiskers and compare to shape
    contacts_l = []
    for frame, frame_tips in mwe.groupby('frame'):
        # Use the fact that edge_a goes from frame 0 to end
        edge_frame = edge_a[frame]
        if edge_frame is None:
            continue

        if verbose and np.mod(frame, 1000) == 0:
            print frame
        
        for idx, frame_tip in frame_tips.iterrows():
            dists = np.sqrt(
                (edge_frame[:, 1] - frame_tip['tip_x']) ** 2 + 
                (edge_frame[:, 0] - frame_tip['tip_y']) ** 2)
            closest_edge_idx = np.argmin(dists)
            closest_dist = dists[closest_edge_idx]
            contacts_l.append({'index': idx, 'closest_dist': closest_dist,
                'closest_edge_idx': closest_edge_idx})
    contacts_df = pandas.DataFrame.from_records(contacts_l)

    if len(contacts_df) == 0:
        # Not sure how to form a nice empty dataframe here
        raise ValueError("no contacts found")

    # Join
    tips_and_contacts = mwe.join(contacts_df.set_index('index'))
    tips_and_contacts = tips_and_contacts[
        tips_and_contacts.closest_dist < contact_dist_thresh]
    return tips_and_contacts


## For clustering contacts
class ClusteredTacHandler(CalculationHandler):
    """Clustering proximity events into contacts"""
    _db_field_path = 'clustered_tac_filename'
    _name = 'clustered_tac'

    # The fields that are required before calculate can run
    _required_fields_for_calculate = (
        'tac_filename',
    )

    def calculate(self, force=False, verbose=True, **kwargs):
        """Wrapper around cluster_contacts_nodb"""
        # Return if force=False and the data exists
        if not force:
            # Check if data available
            data_available = True
            warn_about_field = False
            try:
                self.get_path
            except FieldNotSetError:
                # not calculated yet
                data_available = False
            except FileDoesNotExistError:
                data_available = False
                warn_about_field = True
            
            # Return if it is
            if data_available:
                return
            
            # Warn if we couldn't load data but we were supposed to be able to
            if warn_about_field:
                print (("warning: %s was set " % self._db_field_path) + 
                    "but could not load data, recalculating" 
                )
        
        # We are going to try to calculate
        # Ensure required fields are set
        if not self._check_if_required_fields_for_calculate_set():
            raise RequiredFieldsNotSetError(self)

        # Load required data
        tac = self.video_session.data.tac.load_data()
        
        # Cluster it
        tac_clustered = cluster_contacts_nodb(tac, **kwargs)

        # Save it
        self.save_data(tac_clustered)

def label_greedy(tac, n_contig=15, x_contig=5):
    """Group together contact times within a certain window of each other.
    
    Begin with the first tac. Group it with all future tacs that are separated
    by no more than n_contig frames and x_contig pixels. Continue.
    
    Returns: a new tac2, with a column "group".
    """
    # Initialize the groups
    tac2 = tac.copy().sort_values(by='frame')
    tac2['group'] = 0
    n_groups = 0
    
    # Iterate over tacs
    for idx in tac2.index:
        # Get group of this row, if any
        mygroup = tac2.loc[idx, 'group']
        if mygroup == 0:
            # Make a new group
            n_groups = n_groups + 1
            tac2.loc[idx, 'group'] = n_groups
            mygroup = n_groups
        
        # Find all points with temporal window
        dist = (
            np.abs(tac2.tip_x - tac2.loc[idx, 'tip_x']) + 
            np.abs(tac2.tip_y - tac2.loc[idx, 'tip_y']))
        neighbors = (
            (tac2.group == 0) &
            (tac2.frame < tac2.loc[idx, 'frame'] + n_contig) &
            (dist < x_contig)
            )
        tac2.loc[neighbors, 'group'] = mygroup
    
    return tac2

def cluster_contacts_nodb(tac, max_contacts_per_frame=50, n_contig=3,
    x_contig=100):
    """Cluster contacts by frame into discrete contact events.
    
    max_contacts_per_frame : drop frames with more contacts than this
        Typically these are artefacts on black frames
    n_contig, x_contig : passed to label_greedy
    
    Returns: tac_clustered
    """
    # Group by frame
    tac_gframe = tac.groupby('frame')

    # Get rid of the messed up frames with >50 contacts
    n_contacts_per_frame = tac_gframe.apply(len)
    bad_frames = n_contacts_per_frame.index[
        n_contacts_per_frame > max_contacts_per_frame]
    print "dropping %d bad frames" % len(bad_frames)
    tac = tac[~tac.frame.isin(bad_frames)]
    tac_gframe = tac.groupby('frame')

    # Cluster them
    print "clustering"
    tac_clustered = label_greedy(tac, n_contig=n_contig, x_contig=x_contig)
    
    return tac_clustered
