"""For dealing with db of high-speed video stuff

Stages of processing
0.  Acquiring / demodulation / ffmpeg / copying to server
1.  Creating and filling a session directory
2.  Cropping video (semi-manual)
3.  Tracing
4.  Converting trace output to hdf5
    whiskvid.put_whiskers_into_hdf5(whisk_file, 'test.h5', 
        truncate_seg=1000)
5.  Edging (semi-manual, set lum threshold and shape roi)
    whiskvid.get_all_edges_from_video(video_file, 
        n_frames=n_frames, verbose=True,
        lum_threshold=50, roi_x=(200, width), roi_y=(0, 200), 
        return_frames_instead=return_frames_instead,
        meth='largest_in_roi')
6.  Finding contacts (semi-manual, set follicle range, length thresh)
    tac = whiskvid.calculate_contacts(h5_filename, edge_file, side, 
        tac_filename=tac_filename,
        length_thresh=75, contact_dist_thresh=10,
        fol_range_x=(0, 80), fol_range_y=(100, 300))
7.  Synchronizing to behavior (semi-manual)
8.  Making overlays at retraction time, edge summarizing
    whiskvid.db.dump_edge_summary(res['trial_matrix'], edge_a, 
        res['b2v_fit'], row['v_width'], row['v_height'],
        edge_summary_filename=edge_summary_filename,
        hist_pix_w=2, hist_pix_h=2, vid_fps=30, offset=-.5)
9.  Some debugging plots. Distribution of contacts with overlays.
    Video of contacts / edge summary

4 and 5 can be done simultaneously. 7 can be done earlier.

Try to make each stage done by functions:
    * A low-level one that doesn't know about the db and just gets 
      filenames and parameters
    * A db-level one that just takes a session name and perhaps
      deals with caching
    * One that iterates over the db and does it wherever necessary,
      but it's unclear whether this should be done depth- or breadth-
      first
"""
import os
import glob
import numpy as np
import pandas
import ArduFSM
import my


## Functions to create a db from scratch
def only_one_or_none(l):
    """Returns only_one(l), or None if error"""
    try:
        val = my.misc.only_one(l)
        return val
    except my.misc.UniquenessError:
        return None

def globjoin(dirname, pattern):
    return glob.glob(os.path.join(dirname, pattern))

def remove_skipped(hits, skip_if_includes_l):
    """Returns those hits that don't include a string from skip_if_includes_l"""
    res = []
    for s in hits:
        include_me = True
        for skip_s in skip_if_includes_l:
            if skip_s in s:
                include_me = False
                break
        if include_me:
            res.append(s)
    return res

class FileFinder:
    """Parent class for each file in the session"""
    skip_if_includes_l = []
    always_skip_if_includes_l = ['.bak.']
    
    @classmethod
    def find(self, dirname, debug=False):
        hits = globjoin(dirname, self.glob_pattern)
        hits = remove_skipped(hits, self.skip_if_includes_l)
        hits = remove_skipped(hits, self.always_skip_if_includes_l)
        return only_one_or_none(hits)

class RawVideo(FileFinder):
    """Finds raw video files"""
    glob_pattern = '*.mp4'
    skip_if_includes_l = ['edge_tac_overlay.mp4']

class Whiskers(FileFinder):
    """Finds raw video files"""
    glob_pattern = '*.whiskers'

class EdgesAll(FileFinder):
    """Finds files that contain all edges by frame"""
    glob_pattern = '*.edge_a.*'
    skip_if_includes_l = ['.bak']

class WhiskersHDF5(FileFinder):
    """Finds HDF5-formatted whiskers file"""
    glob_pattern = '*.wseg.h5'

class Contacts(FileFinder):
    """Finds dataframe of contact times and locations"""
    glob_pattern = '*.tac'

class Fit(FileFinder):
    """Finds fit from behavior to video"""
    glob_pattern = '*.fit'
    
    @classmethod
    def load(self, filename):
        res = np.loadtxt(filename)
        return res

class TrialFramesDir(FileFinder):
    """Finds directory containing frames at time of retraction"""
    glob_pattern = '*frames*'

class EdgesSummary(FileFinder):
    """Finds pickle of histogrammed edges for each trial type"""
    glob_pattern = '*.edge_summary.pickle'

class TrialFramesByType(FileFinder):
    """Finds dataframe of trial frames, meaned by type"""
    glob_pattern = '*.overlays.df'

class TrialFramesAllTypes(FileFinder):
    """Finds image array of all overlaid trial type frames"""
    glob_pattern = '*.all_overlays.npy'

class BehaviorLog(FileFinder):
    """Finds log of behavior for session"""
    glob_pattern = 'ardulines.*'

class ContactVideo(FileFinder):
    """Finds video with overlaid edges and contacts"""
    glob_pattern = '*edge_tac_overlay.mp4'


def create_db_from_root_dir(root_dir='/home/chris/whisker_video',
    savename=None):
    """Searches root dir and forms db of existing result files
    
    The directory name, not including the root_dir, is defined as the
    `session`.
    
    Looks for all files specified by a class derived from FileFinder. 
    If multiple hits are found, skips. Stores the results as fields.
    
    Also adds the following as fields:
        v_width, v_height
        b2v_fit0, v2b_fit0, etc
        date_s : first 6 chars of the session
    
    Optionally writes database to disk. Read in like this:
    # pandas.read_csv('db.csv', index_col='session')

    Returns db as a DataFrame, indexed by session.
    """
    session_dir_list = sorted(glob.glob(os.path.join(root_dir, '*/')))
    rec_l = []
    for session_fulldir in session_dir_list:
        # Get the session dir and check root dir
        root_dir_check, session_dir = os.path.split(
            os.path.normpath(session_fulldir))
        assert root_dir == root_dir_check
        
        # Get files that match known endings
        print "processing", session_dir
        video_file = RawVideo.find(session_fulldir)
        whiskers_file = Whiskers.find(session_fulldir)
        edge_a_file = EdgesAll.find(session_fulldir)
        wseg_h5_file = WhiskersHDF5.find(session_fulldir)
        tac_file = Contacts.find(session_fulldir)
        fit_file = Fit.find(session_fulldir)
        frames_dir = TrialFramesDir.find(session_fulldir)
        edge_summary_file = EdgesSummary.find(session_fulldir)
        overlays_file = TrialFramesByType.find(session_fulldir)
        all_overlays_file = TrialFramesAllTypes.find(session_fulldir)
        bfile = BehaviorLog.find(session_fulldir)
        contact_video = ContactVideo.find(session_fulldir)

        # always do side=left for now
        side = 'left'
        
        # video aspect
        if video_file is None:
            v_width, v_height = 0, 0
            duration = None
        else:
            v_width, v_height = my.misc.get_video_aspect(video_file)
            duration = my.misc.get_video_duration(video_file)

        
        # fit
        if fit_file is None:
            fit_arr = [None, None]
            ifit_arr = [None, None]
        else:
            fit_arr = Fit.load(fit_file)
            ifit_arr = my.misc.invert_linear_poly(fit_arr)
        
        # Form record
        rec = {'root_dir': root_dir, 'session': session_dir,
            'session_dir': session_fulldir,
            'whiskers': whiskers_file, 'edge': edge_a_file, 'wseg_h5': wseg_h5_file,
            'tac': tac_file, 'side': side, 'fit': fit_file,
            'fit_b2v0': fit_arr[0], 'fit_b2v1': fit_arr[1],
            'fit_v2b0': ifit_arr[0], 'fit_v2b1': ifit_arr[1],
            'frames': frames_dir, 'bfile': bfile, 'vfile': video_file,
            'edge_summary': edge_summary_file, 'overlays': overlays_file,
            'overlay_image': all_overlays_file,
            'v_width': v_width, 'v_height': v_height, 'duration': duration,
            'contact_video': contact_video,
            'date_s': session_dir[:6]}
        rec_l.append(rec)

    df = pandas.DataFrame.from_records(rec_l).set_index('session').sort()

    if savename is not None:
        df.to_csv(savename)
    
    return df
## End functions to create a db from scratch

## Functions to create a new session directory
def find_closest_bfile(date_string, 
    behavior_dir='/home/chris/runmice/L0/logfiles'):
    """Given a date string like 150313, find the bfile from that day"""
    bfile_l = glob.glob(os.path.join(behavior_dir, 'ardulines.*%s*.*' % 
        date_string))
    
    try:
        return my.misc.only_one(bfile_l)
    except my.misc.UniquenessError:
        return None

def add_bfile_to_session(session, db):
    date_string = db.loc[session, 'date_s']
    session_dir = os.path.join(db.loc[session, 'root_dir'], session)
    add_bfile_to_session_nodb(date_s, session_dir)

def add_bfile_to_session_nodb(date_s, session_dir):
    bfile = find_closest_bfile(date_s)
    
    if bfile is None:
        print "cannot find behavior file for session"
    else:
        bfile_splitname = os.path.split(bfile)[1]
        dst = os.path.join(session_dir, bfile_splitname)
        shutil.copyfile(bfile, dst)
## End functions to create a new session directory

## Load db functions
def save_db(filename='/home/chris/dev/whisker_db/db.csv'):
    # deal with floating point and column order here
    pass

def load_db(filename='/home/chris/dev/whisker_db/db.csv'):
    return pandas.read_csv(filename, index_col='session')

def flush_db(filename='/home/chris/dev/whisker_db/db.csv'):
    db = create_db_from_root_dir(savename=filename)
    return db
## End load db functions

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
    
    Check if there is a bug here when the edge is in the last row and is
    not in the histogram.
    
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
