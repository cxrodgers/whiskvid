"""Handles the calculating of touches and contacts"""
from base import *
import numpy as np
import pandas
import my

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
