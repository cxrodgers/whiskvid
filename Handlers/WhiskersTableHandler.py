"""Module for handling whiskers table"""
import numpy as np
import pandas
from base import CalculationHandler
try:
    import tables
except ImportError:
    pass

class WhiskersTableHandler(CalculationHandler):
    """Handler for whiskers table (HDF5)"""
    _db_field_path = 'whiskers_table_filename'
    _name = 'whiskers'
    
    def load_data(self, side='right', length_thresh=75, verbose=True,
        add_angle=True, add_sync=True):
        """Load masked whisker ends.
        
        The data is read from the whisker_table HDF5 file.
        
        side : face side. TODO: save this in database
        length_thresh, verbose, add_angle : 
            passed to get_masked_whisker_ends_nodb
    
        add_sync: uses db sync to add vtime and btime columns
        """
        # Get path to HDF5 file
        filename = self.get_path
        
        # Get parameters from video session
        fol_range_x = self.video_session.fol_range_x
        fol_range_y = self.video_session.fol_range_y
        
        # Need a standard way to check necessary parameters are not null
        required_attrs = ['param_fol_x0', 'param_fol_x1', 'param_fol_y0', 
            'param_fol_y1']
        for attr in required_attrs:
            if pandas.isnull(getattr(self.video_session._django_object, attr)):
                raise AttributeError("param %s must not be null" % attr)
        
        # Get the whiskers
        mwe = get_masked_whisker_ends_nodb(filename, 
            side=side,
            fol_range_x=fol_range_x,
            fol_range_y=fol_range_y, 
            length_thresh=length_thresh, 
            verbose=verbose,
            add_angle=add_angle,
        )

        # Add syncing information (requires info from database)
        if add_sync:
            # Get sync
            v2b_fit = self.video_session.fit_v2b
            
            if np.any(np.isnan(v2b_fit)):
                print "warning: v2b_fit is null, cannot sync"
            else:
                # Apply sync
                mwe['vtime'] = mwe['frame'] / 30.
                mwe['btime'] = np.polyval(v2b_fit, mwe.vtime.values)   

        return mwe

    @property
    def new_path(self):
        """Filename of whiskers table (overriding _name which is 'whiskers')
        
        """
        return 'whiskers.h5'

def get_masked_whisker_ends_nodb(h5_filename, side, 
    fol_range_x, fol_range_y, length_thresh=75, 
    verbose=True, add_angle=True):
    """Return a table of whiskers that has been masked by follicle and length

    add_angle: uses the arctan2 method to add an angle column
    """
    # Get the ends
    resdf = get_whisker_ends_hdf5(h5_filename, side=side)
    if verbose:
        print "whisker rows: %d" % len(resdf)

    # Drop everything < thresh
    resdf = resdf[resdf['length'] >= length_thresh]
    if verbose:
        print "whisker rows after length: %d" % len(resdf)

    # Follicle mask
    resdf = resdf[
        (resdf['fol_x'] > fol_range_x[0]) & (resdf['fol_x'] < fol_range_x[1]) &
        (resdf['fol_y'] > fol_range_y[0]) & (resdf['fol_y'] < fol_range_y[1])]
    if verbose:
        print "whisker rows after follicle mask: %d" % len(resdf)    

    if add_angle:
        # Get angle on each whisker
        resdf['angle'] = np.arctan2(
            -(resdf['fol_y'].values - resdf['tip_y'].values),
            resdf['fol_x'].values - resdf['tip_x'].values) * 180 / np.pi

    return resdf

def get_whisker_ends_hdf5(hdf5_file=None, side=None, 
    also_calculate_length=True):
    """Reimplement get_whisker_ends on hdf5 file"""
    # Get the summary
    with tables.open_file(hdf5_file) as fi:
        summary = pandas.DataFrame.from_records(fi.root.summary.read())
    
    # Rename
    summary = summary.rename(columns={'time': 'frame', 'id': 'seg'})
    
    # Assign tip and follicle
    if side == 'left':
        # Identify which are backwards
        switch_mask = summary['tip_x'] < summary['fol_x']
        
        # Switch those rows
        new_summary = summary.copy()
        new_summary.loc[switch_mask, 'tip_x'] = summary.loc[switch_mask, 'fol_x']
        new_summary.loc[switch_mask, 'fol_x'] = summary.loc[switch_mask, 'tip_x']
        new_summary.loc[switch_mask, 'tip_y'] = summary.loc[switch_mask, 'fol_y']
        new_summary.loc[switch_mask, 'fol_y'] = summary.loc[switch_mask, 'tip_y']
        summary = new_summary
    elif side == 'right':
        # Like left, but x is switched
        
        # Identify which are backwards
        switch_mask = summary['tip_x'] > summary['fol_x']
        
        # Switch those rows
        new_summary = summary.copy()
        new_summary.loc[switch_mask, 'tip_x'] = summary.loc[switch_mask, 'fol_x']
        new_summary.loc[switch_mask, 'fol_x'] = summary.loc[switch_mask, 'tip_x']
        new_summary.loc[switch_mask, 'tip_y'] = summary.loc[switch_mask, 'fol_y']
        new_summary.loc[switch_mask, 'fol_y'] = summary.loc[switch_mask, 'tip_y']
        summary = new_summary        
    elif side == 'top':
        # Identify which are backwards (0 at the top (?))
        switch_mask = summary['tip_y'] < summary['fol_y']
        
        # Switch those rows
        new_summary = summary.copy()
        new_summary.loc[switch_mask, 'tip_x'] = summary.loc[switch_mask, 'fol_x']
        new_summary.loc[switch_mask, 'fol_x'] = summary.loc[switch_mask, 'tip_x']
        new_summary.loc[switch_mask, 'tip_y'] = summary.loc[switch_mask, 'fol_y']
        new_summary.loc[switch_mask, 'fol_y'] = summary.loc[switch_mask, 'tip_y']
        summary = new_summary        
    elif side is None:
        pass
    else:
        raise NotImplementedError

    # length
    if also_calculate_length:
        summary['length'] = np.sqrt(
            (summary['tip_y'] - summary['fol_y']) ** 2 + 
            (summary['tip_x'] - summary['fol_x']) ** 2)
    
    return summary    

## More HDF5 stuff
def get_summary(h5file):
    """Return summary metadata of all whiskers"""
    return pandas.DataFrame.from_records(h5file.root.summary.read())

def get_x_pixel_handle(h5file):
    return h5file.root.pixels_x

def get_y_pixel_handle(h5file):
    return h5file.root.pixels_y

def select_pixels(h5file, **kwargs):
    summary = get_summary(h5file)
    mask = my.pick(summary, **kwargs)
    
    # For some reason, pixels_x[fancy] is slow
    res = [
        np.array([
            h5file.root.pixels_x[idx], 
            h5file.root.pixels_y[idx], 
            ])
        for idx in mask]
    return res
## End HDF5 stuff