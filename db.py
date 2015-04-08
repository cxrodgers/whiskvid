"""For dealing with db of high-speed video stuff"""
import os
import numpy as np
import pandas
import ArduFSM
import my

def load_everything_from_session(session, db):
    """Load all data from specified row of db
    
    TODO: just specify session, and automatically read db here
    """
    row = db.ix[session]

    # Fit from previous file
    b2v_fit = np.loadtxt(os.path.join(row['root_dir'], session, row['fit']))
    v2b_fit = my.misc.invert_linear_poly(b2v_fit)

    # Get behavior df
    trial_matrix = ArduFSM.TrialMatrix.make_trial_matrix_from_file(
        os.path.join(row['root_dir'], session, row['bfile']))

    # Get tips and contacts
    tac = pandas.read_pickle(os.path.join(row['root_dir'], session, row['tac']))
    
    # Get edge_a
    edge_a = np.load(os.path.join(row['root_dir'], session, row['edge']))
    edge_summary = pandas.read_pickle(os.path.join(
        row['root_dir'], session, row['edge_summary']))
        
    # Get overlays
    overlay_image = np.load(os.path.join(row['root_dir'], session, 
        row['overlay_image']))
        

    
    return {'b2v_fit': b2v_fit, 'v2b_fit': v2b_fit, 
        'trial_matrix': trial_matrix, 'tac': tac,
        'edge_a': edge_a, 'edge_summary': edge_summary, 
        'overlay_image': overlay_image}

def add_trials_to_tac(tac, v2b_fit, trial_matrix, drop_late_contacts=False):
    """Add the trial numbers to tac and return it
    
    trial_matrix should already have "choice_time" column from BeWatch.misc
    
    Also adds "vtime" (in the spurious 30fps timebase)
    and "btime" (using fit)
    and "t_wrt_choice" (using choice_time)
    
    If `drop_late_contacts`: drops every contact that occurred after choice
    """
    # "vtime" is in the spurious 30fps timebase
    # the fits take this into account
    tac['vtime'] = tac['frame'] / 30.
    tac['btime'] = np.polyval(v2b_fit, tac['vtime'].values)

    # Add trial info to tac
    tac['trial'] = trial_matrix.index[
        np.searchsorted(trial_matrix['start_time'].values, 
            tac['btime'].values) - 1]    

    # Add rewside and outcome to tac
    tac = tac.join(trial_matrix[[
        'rewside', 'outcome', 'choice_time', 'isrnd', 'choice']], 
        on='trial')
    tac['t_wrt_choice'] = tac['btime'] - tac['choice_time']
    
    if drop_late_contacts:
        tac = tac[tac.t_wrt_choice < 0]

    return tac


def dump_edge_summary(trial_matrix, edge_a, b2v_fit, v_width, v_height,
    edge_summary_filename=None,
    hist_pix_w=2, hist_pix_h=2, vid_fps=30, offset=-.5):
    """Extract edges at choice times for each trial type and dump
    
    2d-histograms at choice times and saves the resulting histogram
    
    trial_matrix : must have choice time added in already
    edge_a : array of edge at every frame
    offset : time relative to choice time at which frame is dumped
    edge_summary_filename : where to dump results, if anywhere
    
    Returns: {
        'row_edges': row_edges, 'col_edges': col_edges, 
        'H_l': H_l, 'rewside_l': rwsd_l, 'srvpos_l': srvpos_l}    
    """
    # Convert choice time to frames using b2v_fit
    choice_btime = np.polyval(b2v_fit, trial_matrix['choice_time'])
    choice_btime = choice_btime + offset
    trial_matrix['choice_bframe'] = np.rint(choice_btime * vid_fps)
    
    # hist2d the edges for each rewside * servo_pos
    gobj = trial_matrix.groupby(['rewside', 'servo_pos'])
    rwsd_l, srvpos_l, H_l = [], [], []
    col_edges = np.arange(0, v_width, hist_pix_w)
    row_edges = np.arange(0, v_height, hist_pix_h)    
    for (rwsd, srvpos), subtm in gobj:
        # Extract the edges at choice time from all trials of this type
        n_bad_edges = 0
        sub_edge_a = []
        for frame in subtm['choice_bframe'].values:
            # Skip ones outside the video
            if frame < 0 or frame >= len(edge_a) or np.isnan(frame):
                continue
            
            # Count the ones for which no edge was detected
            elif edge_a[frame] is None:
                n_bad_edges = n_bad_edges + 1
                continue
            
            else:
                sub_edge_a.append(edge_a[int(frame)])

        # Warn
        if n_bad_edges > 0:
            print "warning: some edge_a entries are None at choice time"
        if len(sub_edge_a) == 0:
            print "warning: could not extract any edges for " \
                "rwsd %s and srvpos %d" % (rwsd, srvpos)
            continue
        
        # Extract rows and cols from sub_edge_a
        col_coords = np.concatenate([edg[:, 0] for edg in sub_edge_a])
        row_coords = np.concatenate([edg[:, 1] for edg in sub_edge_a])
        
        # Histogram it .. note H is X in first dim and Y in second dim
        H, xedges, yedges = np.histogram2d(row_coords, col_coords,
            bins=[col_edges, row_edges])
        
        # Store
        rwsd_l.append(rwsd)
        srvpos_l.append(srvpos)
        H_l.append(H.T)
    
    # Save
    res = {
        'row_edges': row_edges, 'col_edges': col_edges, 
        'H_l': H_l, 'rewside_l': rwsd_l, 'srvpos_l': srvpos_l}
    if edge_summary_filename is not None:
        my.misc.pickle_dump(res, edge_summary_filename)
    return res
