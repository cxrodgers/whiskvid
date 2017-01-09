"""Handlers for summarized and colorized contacts"""
from base import *
import numpy as np
import pandas

class ContactsSummaryHandler(CalculationHandler):
    """Contacts summmary"""
    _db_field_path = 'contacts_summary_filename'
    _name = 'contacts_summary'

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
            failed_to_read_data = False
            try:
                data = self.load_data()
            except (FieldNotSetError, FileDoesNotExistError):
                # Failed to read, probably not calculated
                failed_to_read_data = True
            
            # Return data if we were able to load it
            if not failed_to_read_data:
                return data
            
            # Warn if we couldn't load data but we were supposed to be able to
            if not self.field_is_null:
                print (("warning: %s was set " % self._db_field_name) + 
                    "but could not load data, recalculating" 
                )
        
        ## Begin handler-specific stuff
        # Load necessary data
        ctac = self.video_session.data.clustered_tac.load_data()
        cs = self.video_session.data.contacts_summary.load_data()
        cwe = self.video_session.data.colorized_whisker_ends.load_data()
        
        # Calculate
        ccs = colorize_contacts_summary_nodb(ctac, cs, cwe)
        ## End handler-specific stuff
        
        # Store
        if save:
            self.save_data(ccs)
        
        return ccs

def colorize_contacts_summary_nodb(ctac, cs, cwe):
    """Colorize the contacts in cs using colorized_whisker_ends

    We first load the colorized_whisker_ends, contacts_summary,
    and clustered_tac. We colorize each tac, and then propagate this
    color to cs. For the last step, we take the most common color of
    tac within that clustered_tac. 

    Returns: colorized_contacts_summary
        This is contacts_summary with another column for color
    """
    # Copy so we can assign a new column
    cs = cs.copy()
    
    # Assign the color from cwe to clustered_tac
    # ctac shares an index with cwe
    ctac['color'] = cwe.loc[ctac.index, 'color_group']

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
    
    return cs


def summarize_contacts(vsession):
    """Summarize the contacts using the database and save to disk
    
    If a file called "contacts_summary" already exists in the session 
    directory, it is loaded and returned immediately.
    
    Otherwise, the clustered_tac is loaded, and summarize_contacts_nodb()
    is called. Then the contacts_summary is saved to disk and returned.
    """    
    # Determine the filename
    db = whiskvid.db.load_db()
    contacts_summary_name = os.path.join(db.loc[vsession, 'session_dir'],
        'contacts_summary')

    # Return if already exists
    if os.path.exists(contacts_summary_name):
        print "loading cached contacts summary"
        contacts_summary = pandas.read_pickle(contacts_summary_name)
        return contacts_summary

    # Otherwise compute
    tac_clustered_name = os.path.join(db.loc[vsession, 'session_dir'],
        'clustered_tac')
    tac_clustered = pandas.read_pickle(tac_clustered_name)
    contacts_summary = summarize_contacts_nodb(tac_clustered)

    # Store
    contacts_summary.to_pickle(contacts_summary_name)
    
    return contacts_summary

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
