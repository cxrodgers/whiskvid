"""Handlers for summarized and colorized contacts"""
from base import *
import numpy as np
import pandas
import whiskvid
import MCwatch

class ContactsSummaryHandler(CalculationHandler):
    """Contacts summmary"""
    _db_field_path = 'contacts_summary_filename'
    _name = 'contacts_summary'
    
    def calculate(self, force=False, verbose=True, **kwargs):
        """Wrapper around summarize_contacts_nodb"""
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
        tac_clustered = self.video_session.data.clustered_tac.load_data()
        
        # Calculate
        contacts_summary = summarize_contacts_nodb(tac_clustered)

        # Store
        self.save_data(contacts_summary)


class ColorizedContactsSummaryHandler(CalculationHandler):
    """Colorized contacts summmary"""
    _db_field_path = 'colorized_contacts_summary_filename'
    _name = 'colorized_contacts_summary'
    
    def calculate(self, force=False, save=True):
        """Colorize the contacts in cs using colorized_whisker_ends
        
        See colorize_contacts_summary_nodb for algorithm. This loads
        data from disk and stores result.
        
        Returns : colorized_contacts_summary
        """
        # Return if force=False and we can load the data
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
        
        ## Begin handler-specific stuff
        # Load necessary data
        ctac = self.video_session.data.clustered_tac.load_data()
        cs = self.video_session.data.contacts_summary.load_data(
            add_trial_info=False)
        cwe = self.video_session.data.colorized_whisker_ends.load_data()
        
        # Calculate
        ccs = colorize_contacts_summary_nodb(ctac, cs, cwe)
        ## End handler-specific stuff
        
        # Store
        if save:
            self.save_data(ccs)
        
        return ccs
    
    def load_data(self, add_trial_info=True, columns_to_join=None,
        trial_matrix=None):
        """Load ccs and optionally add trial info"""
        # Parent class to load the raw data
        res = super(ColorizedContactsSummaryHandler, self).load_data()

        # Optionally add trial info
        if add_trial_info:
            # Add a trial column, based on where the contact start_frame
            # fits in the trial_matrix['exact_start_frame'] column
            res['trial'] = trial_matrix.index[
                np.searchsorted(
                    trial_matrix['exact_start_frame'].values, 
                    res['frame_start'].values
                ) - 1]               
            
            if (res['trial'] == -1).any():
                print "warning: dropping contacts before first trial started"
                res = res[res['trial'] != -1].copy()

            # Join columns
            if columns_to_join is None:
                columns_to_join = [
                    'rewside', 'servo_pos', 'stepper_pos', 
                    'isrnd', 'choice', 'outcome',
                ]

            res = res.join(trial_matrix[columns_to_join], on='trial')
        
        return res 

def colorize_contacts_summary_nodb(ctac, cs, cwe):
    """Colorize the contacts in cs using colorized_whisker_ends

    We first load the colorized_whisker_ends, contacts_summary,
    and clustered_tac. We colorize each tac, and then propagate this
    color to cs. For the last step, we take the most common color of
    tac within that clustered_tac. 
    
    Finally, we use the mapping between color_group and whisker in
    cwe to add a 'whisker' column in the result.

    Returns: colorized_contacts_summary
        This is contacts_summary with another column for color
    """
    # Copy so we can assign a new column
    cs = cs.copy()
    ctac = ctac.copy()
    
    # For some reason ctac group is sometimes float
    if 'group' in ctac.select_dtypes([np.float]).columns:
        print "warning: ctac group is float"
        ctac['group'] = ctac['group'].astype(np.int)
    
        # Likely cs.index will also be float in this case
        cs.index = cs.index.astype(np.int)
    
    # Drop rows in ctac that are not in cwe
    # ctac shares an index with cwe
    # Primarily this would be whiskers that are too short to be colorized
    # But why are there contacts from these whiskers anyway?
    ctac_not_in_cwe = ~ctac.index.isin(cwe.index)
    if ctac_not_in_cwe.sum() > 0:
        print "warning: dropping %d/%d touches from ctac not in cwe" % (
            ctac_not_in_cwe.sum(), len(ctac))
        ctac = ctac[~ctac_not_in_cwe].copy()
        
        # Also drop the corresponding rows in cs
        cs = cs[cs.index.isin(ctac['group'])].copy()
    
    # Assign the color from cwe to clustered_tac
    ctac['color'] = cwe.loc[ctac.index, 'color_group']

    # in debugging, cwe may be truncated
    # so only select the tacs for which we have entries in cwe
    if ctac.color.isnull().any():
        # This should no longer happen now that the above check is done
        1/0
        print "warning: dropping nan in ctac before colorizing cs"
        ctac = ctac.dropna()

    # Group the ctac by 'group' (index into cs)
    # Choose the most common color within that gruop
    grouped_ctac = ctac.groupby('group')['color']

    # Choose the color for each group
    def choose_color_group(color_series):
        """Identify the most common color in color_series
        
        color_series : Series where the values are colors
        
        Returns: dict
            n_cg : number of unique colors
            nnz_cg : number of non-zero unique colors
            cg : most common color
        """
        vc = color_series.value_counts()
        return {
            'n_cg': len(vc), 
            'cg': vc.idxmax(), 
            'nnz_cg': len(vc.drop(0, errors='ignore'))
        }
    chosen_colors = grouped_ctac.apply(choose_color_group).unstack()    
    
    # Determine how well-assigned the colors were
    print ("Of %d, %d were unique, %d were unique after dropping 0" % (
        len(chosen_colors), 
        (chosen_colors.n_cg == 1).sum(), 
        (chosen_colors.nnz_cg == 1).sum(),) +
        " and %d were assigned 0" % (
        chosen_colors.cg == 0).sum()
    )    
    
    # Use the fac that ctac.group is the index of cs
    cs['color'] = chosen_colors['cg']     
    assert not np.any(cs['color'].isnull())

    
    ## Also add whisker name
    # Check that it is unique (other than color group 0 which is unlabeled)
    nunique = cwe.groupby('color_group')['whisker'].nunique()
    if 0 in nunique.index:
        nunique = nunique.drop(0)
    assert (nunique == 1).all()
    
    # Infer the names
    cg2whisker = cwe.groupby('color_group')['whisker'].first()
    cg2whisker.loc[0] = '?'
    
    # Apply
    cs['whisker'] = cs['color'].map(cg2whisker)
    assert not np.any(cs['whisker'].isnull())
    
    assert cs.index.dtype == np.dtype('int64')
    return cs

def summarize_contacts_nodb(tac_clustered):
    """Summarize the timing and location of clustered_tac
    
    Returns : contacts_summary
    """
    rec_l = []
    for tacnum, cluster in tac_clustered.groupby('group'):
        rec = {'cluster': tacnum}
        
        # Start and stop of cluster
        rec['frame_start'] = cluster['frame'].min()
        rec['frame_stop'] = cluster['frame'].max()
        rec['duration'] = rec['frame_stop'] - rec['frame_start'] + 1
        
        # Mean tip and fol of cluster
        rec['tip_x'] = cluster['tip_x'].mean()
        rec['tip_y'] = cluster['tip_y'].mean()
        rec['fol_x'] = cluster['fol_x'].mean()
        rec['fol_y'] = cluster['fol_y'].mean()
        rec['pixlen'] = np.sqrt(
            (rec['tip_y'] - rec['fol_y']) ** 2 +
            (rec['tip_x'] - rec['fol_x']) ** 2)
        
        rec_l.append(rec)
    contacts_summary = pandas.DataFrame.from_records(rec_l).set_index('cluster')
    
    return contacts_summary
