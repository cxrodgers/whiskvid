"""Functions for analyzing whiski data

* Wrappers around whiski loading functions
* Various methods for extracting angle information from whiski data
* Methods for processing video to extract shape contours
* HDF5 file format stuff
* The high-speed video analysis pipeline. Tried to have a generic system
  for each step in the pipeline. So, for instance there is:
    whiskvid.base.edge_frames_manual_params
        To set the manual parameters necessary for this step
    whiskvid.base.edge_frames_manual_params_db
        Same as above but use the db defined in whiskvid.db
    whiskvid.base.edge_frames_nodb
        Run the step without relying on the db
    whiskvid.base.edge_frames
        Same as above but save to db
* Plotting stuff, basically part of the pipeline
"""
try:
    import traj, trace
except ImportError:
    pass
import numpy as np, pandas
import os
import scipy.ndimage
import my
import ArduFSM
import MCwatch.behavior
import whiskvid
import WhiskiWrap
import matplotlib.pyplot as plt
import pandas
import kkpandas

try:
    import tables
except ImportError:
    pass

# whisker colors
WHISKER_COLOR_ORDER_W = [
    'white', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange']
WHISKER_COLOR_ORDER_K = [
    'k', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange']

def load_whisker_traces(whisk_file):
    """Load the traces, return as frame2segment_id2whisker_seg"""
    frame2segment_id2whisker_seg = trace.Load_Whiskers(whisk_file)
    return frame2segment_id2whisker_seg

def load_whisker_identities(measure_file):
    """Load the correspondence between traces and identified whiskers
    
    Return as whisker_id2frame2segment_id
    """
    tmt = traj.MeasurementsTable(measure_file)
    whisker_id2frame2segment_id = tmt.get_trajectories()
    return whisker_id2frame2segment_id

def load_whisker_positions(whisk_file, measure_file, side='left'):
    """Load whisker data and return angle at every frame.
    
    This algorithm needs some work. Not sure the best way to convert to
    an angle. See comments.
    
    Whisker ids, compared with color in whiski GUI:
    (This may differ with the total number of whiskers??)
        -1  orange, one of the unidentified traces
        0   red
        1   yellow
        2   green
        3   cyan
        4   blue
        5   magenta
    
    Uses `side` to disambiguate some edge cases.
    
    Returns DataFrame `angl_df` with columns:
        frame: frame #
        wid: whisker #
        angle: angle calculated by fitting a polynomial to trace
        angle2: angle calculated by slope between endpoints.

    `angle2` is noisier overall but may be more robust to edge cases.
    
    You may wish to pivot:
    piv_angle = angl_df.pivot_table(rows='frame', cols='wid', 
        values=['angle', 'angle2'])    
    """
    
    
    # Load whisker traces and identities
    frame2segment_id2whisker_seg = load_whisker_traces(whisk_file)
    whisker_id2frame2segment_id = load_whisker_identities(measure_file)
    
    # It looks like it numbers them from Bottom to Top for side == 'left'
    # whiski colors them R, G, B

    # Iterate over whiskers
    rec_l = []
    for wid, frame2segment_id in whisker_id2frame2segment_id.items():
        # Iterate over frames
        for frame, segment_id in frame2segment_id.items():
            # Get the actual segment for this whisker and frame
            ws = frame2segment_id2whisker_seg[frame][segment_id]
            
            # Fit angle two ways
            angle = angle_meth1(ws.x, ws.y, side)
            angle2 = angle_meth2(ws.x, ws.y, side)

            # Store
            rec_l.append({
                'frame': frame, 'wid': wid, 'angle': angle, 'angle2': angle2})

    # DataFrame it
    angl_df = pandas.DataFrame.from_records(rec_l)
    #~ piv_angle = angl_df.pivot_table(rows='frame', cols='wid', 
        #~ values=['angle', 'angle2'])
    
    return angl_df

def angle_meth1(wsx, wsy, side):
    """Fit angle by lstsqs line fit, then arctan, then pin.
    
    This will fail for slopes close to vertical.
    """
    # fit a line and calculate angle of whisker
    # in the video, (0, 0) is upper left, so we need to take negative of slope
    # This will fail for slopes close to vertical, for instance if it
    # has this shape: (  because least-squares fails here
    # eg Frame 5328 in 0509A_cropped_truncated_4
    p = np.polyfit(wsx, wsy, deg=1)
    slope = -p[0]

    # Arctan gives values between -90 and 90
    # Basically, we cannot discriminate a SSW whisker from a NNE whisker
    # Can't simply use diff_x because the endpoints can be noisy
    # Similar problem occurs with ESE and WNW, and then diff_y is noisy
    # Easiest way to do it is just pin the data to a known range
    angle = np.arctan(slope) * 180 / np.pi    
    
    # pin
    pinned_angle = pin_angle(angle, side)
    
    return pinned_angle

def angle_meth2(wsx, wsy, side):
    """Fit angle by arctan of tip vs follicle, then pin"""
    # Separate angle measurement: tip vs follicle
    # This will be noisier
    # Remember to flip up/down here
    # Also remember that ws.x and ws.y go from tip to follicle (I think?)
    # Actually the go from tip to follicle in one video and from follicle
    # to tip in the other; and then occasional exceptions on individual frames
    angle = np.arctan2(
        -(wsy[0] - wsy[-1]), wsx[0] - wsx[-1]) * 180 / np.pi

    # On rare occasions it seems to be flipped, 
    # eg Frame 9 in 0509A_cropped_truncated_4
    # So apply the same fix, even though it shouldn't be necessary here
    # pin
    pinned_angle = pin_angle(angle, side)
    
    return pinned_angle    

def pin_angle(angle, side):
    """Pins angle to normal range, based on side"""
    # side = left, so theta ~-90 to +90
    # side = top, so theta ~ -180 to 0    
    
    if side == 'top':
        if angle > 0:
            return angle - 180
    elif side == 'left':
        if angle > 90:
            return angle - 180
    return angle
    
def assign_tip_and_follicle(x0, x1, y0, y1, side=None):
    """Decide which end is the tip.
    
    The side of the screen that is closest to the face is used to determine
    the follicle. For example, if the face is along the left, then the
    left-most end is the follicle.
    
    We assume (0, 0) is in the upper left corner, and so "top" means that
    the face lies near row zero.
    
    Returns: fol_x, tip_x, fol_y, tip_y
        If side is None, return x0, x1, y0, y1
    """
    if side is None:
        return x0, x1, y0, y1
    elif side in ['left', 'right', 'top', 'bottom']:
        # Is it correctly oriented, ie, 0 is fol and 1 is tip
        is_correct = (
            (side == 'left' and x0 < x1) or 
            (side == 'right' and x1 < x0) or 
            (side == 'top' and y0 < y1) or 
            (side == 'bottom' and y1 < y0))
        
        # Return normal or swapped
        if is_correct:
            return x0, x1, y0, y1
        else:
            return x1, x0, y1, y0
    else:
        raise ValueError("unknown value for side: %s" % side)

def get_whisker_ends(whisk_file=None, frame2segment_id2whisker_seg=None,
    side=None, also_calculate_length=True):
    """Returns dataframe with both ends of every whisker
    
    Provide either whisk_file or frame2segment_id2whisker_seg
    side : used to determine which end is which
    
    Returns a DataFrame with columns:
        'fol_x', 'fol_y', 'frame', 'seg', 'tip_x', 'tip_y', 'length'
    """
    # Load traces
    if frame2segment_id2whisker_seg is None:
        frame2segment_id2whisker_seg = load_whisker_traces(whisk_file)
    
    # Get tips and follicles
    res_l = []
    for frame, segment_id2whisker_seg in frame2segment_id2whisker_seg.items():
        for segment_id, whisker_seg in segment_id2whisker_seg.items():
            # Get x and y of both ends
            x0, x1 = whisker_seg.x[[0, -1]]
            y0, y1 = whisker_seg.y[[0, -1]]
            
            # Pin
            fol_x, tip_x, fol_y, tip_y = assign_tip_and_follicle(x0, x1, y0, y1, 
                side=side)
            
            # Stores
            res_l.append({
                'frame': frame, 'seg': segment_id,
                'tip_x': tip_x, 'tip_y': tip_y,
                'fol_x': fol_x, 'fol_y': fol_y})

    # DataFrame
    resdf = pandas.DataFrame.from_records(res_l)

    # length
    if also_calculate_length:
        resdf['length'] = np.sqrt(
            (resdf['tip_y'] - resdf['fol_y']) ** 2 + 
            (resdf['tip_x'] - resdf['fol_x']) ** 2)
    
    return resdf


## Begin stuff for putting whisker data into HDF5
try:
    class WhiskerSeg(tables.IsDescription):
        time = tables.UInt32Col()
        id = tables.UInt16Col()
        tip_x = tables.Float32Col()
        tip_y = tables.Float32Col()
        fol_x = tables.Float32Col()
        fol_y = tables.Float32Col()
        pixlen = tables.UInt16Col()
except NameError:
    pass

def put_whiskers_into_hdf5(session, db=None, **kwargs):
    """Puts whiskers for session and updates db"""
    if db is None:
        db = whiskvid.db.load_db()
    row = db.ix[session]
    
    # Generate output file name
    if pandas.isnull(db.loc[session, 'wseg_h5']):
        output_file = os.path.join(row['session_dir'], session + '.wseg.h5')
        db.loc[session, 'wseg_h5'] = output_file

    # Save immediately to avoid race
    whiskvid.db.save_db(db)      

    put_whiskers_into_hdf5_nodb(row['whiskers'], db.loc[session, 'wseg_h5'],
        **kwargs)

def put_whiskers_into_hdf5_nodb(whisk_filename, h5_filename, verbose=True,
    flush_interval=100000, truncate_seg=None):
    """Load data from whisk_file and put it into an hdf5 file
    
    The HDF5 file will have two basic components:
        /summary : A table with the following columns:
            time, id, fol_x, fol_y, tip_x, tip_y, pixlen
            These are all directly taken from the whisk file
        /pixels_x : A vlarray of the same length as summary but with the
            entire array of x-coordinates of each segment.
        /pixels_y : Same but for y-coordinates
    
    truncate_seg : for debugging, stop after this many segments
    """
    import tables
    
    ## Load it, so we know what expectedrows is
    # This loads all whisker info into C data types
    # wv is like an array of trace.LP_cWhisker_Seg
    # Each entry is a trace.cWhisker_Seg and can be converted to
    # a python object via: wseg = trace.Whisker_Seg(wv[idx])
    # The python object responds to .time and .id (integers) and .x and .y (numpy
    # float arrays).
    wv, nwhisk = trace.Debug_Load_Whiskers(whisk_filename)
    if truncate_seg is not None:
        nwhisk = truncate_seg

    # Open file
    h5file = tables.open_file(h5_filename, mode="w")

    # A group for the normal data
    table = h5file.create_table(h5file.root, "summary", WhiskerSeg, 
        "Summary data about each whisker segment",
        expectedrows=nwhisk)

    # Put the contour here
    xpixels_vlarray = h5file.create_vlarray(
        h5file.root, 'pixels_x', 
        tables.Float32Atom(shape=()),
        title='Every pixel of each whisker (x-coordinate)',
        expectedrows=nwhisk)
    ypixels_vlarray = h5file.create_vlarray(
        h5file.root, 'pixels_y', 
        tables.Float32Atom(shape=()),
        title='Every pixel of each whisker (y-coordinate)',
        expectedrows=nwhisk)


    ## Iterate over rows and store
    h5seg = table.row
    for idx in range(nwhisk):
        # Announce
        if verbose and np.mod(idx, 10000) == 0:
            print idx

        # Get the C object and convert to python
        # I suspect this is the bottleneck in speed
        cws = wv[idx]
        wseg = trace.Whisker_Seg(cws)

        # Write to the table
        h5seg['time'] = wseg.time
        h5seg['id'] = wseg.id
        h5seg['fol_x'] = wseg.x[0]
        h5seg['fol_y'] = wseg.y[0]
        h5seg['tip_x'] = wseg.x[-1]
        h5seg['tip_y'] = wseg.y[-1]
        h5seg['pixlen'] = len(wseg.x)
        assert len(wseg.x) == len(wseg.y)
        h5seg.append()
        
        # Write x
        xpixels_vlarray.append(wseg.x)
        ypixels_vlarray.append(wseg.y)

        if np.mod(idx, flush_interval) == 0:
            table.flush()

    h5file.close()    

## tracing
def trace_session(session, db=None, create_monitor_video=False, 
    chunk_size=200, stop_after_frame=None, n_trace_processes=8,
    monitor_video_kwargs=None):
    """Runs trace on session using WhiskiWrap.
    
    Currently this only works on modulated mat files.
    It first writes them out as tiffs to trace
        trace_write_chunked_tiffs_nodb
        Optionally at this point a monitor video can also be created.
    And then traces them
        trace_session_nodb
    If tiffs_to_trace directory already exists, the first step is skipped.
    
    session : name of session to trace
    create_monitor_video : Whether to create a monitor video
        This could be useful for subsequent analysis (eg, shapes)
    monitor_video_kwargs : dict of kwargs for trace_write_chunked_tiffs_nodb
        Default: {'vcodec': 'libx264', 'qp': 15}
        For lossless, use {'vcodec': 'libx264', 'qp': 0}

    chunk_size, stop_after_frame : passed to trace_write_chunked_tiffs_nodb
    
    """
    if db is None:
        db = whiskvid.db.load_db()

    # Extract some info from the db
    whisker_session_directory = db.loc[session, 'session_dir']
    
    # Error check that matfile_directory exists
    # Later rewrite this to run on raw videos too
    if pandas.isnull(db.loc[session, 'matfile_directory']):
        raise ValueError("trace only supports matfile directory for now")

    # Store the wseg_h5_fn in the db if necessary
    if pandas.isnull(db.loc[session, 'wseg_h5']):
        # Create a wseg h5 filename
        db.loc[session, 'wseg_h5'] = whiskvid.db.WhiskersHDF5.generate_name(
            whisker_session_directory)
        
        # Save right away, to avoid stale db
        whiskvid.db.save_db(db)  

    # Run the trace if the file doesn't exist
    if not os.path.exists(db.loc[session, 'wseg_h5']):
        # Where to put tiff stacks and timestamps and monitor video
        tiffs_to_trace_directory = os.path.join(whisker_session_directory, 
            'tiffs_to_trace')
        timestamps_filename = os.path.join(whisker_session_directory, 
            'tiff_timestamps.npy')
        if create_monitor_video:
            monitor_video = os.path.join(whisker_session_directory,
                session + '.mkv')
            if monitor_video_kwargs is None:
                monitor_video_kwargs = {'vcodec': 'libx264', 'qp': 21}
        else:
            monitor_video = None
            monitor_video_kwargs = {}
        
        # Skip writing tiffs if the directory already exists
        # This is a bit of a hack because tiffs_to_trace is not in the db
        if not os.path.exists(tiffs_to_trace_directory):
            # Create the directory and run trace_write_chunked_tiffs_nodb
            os.mkdir(tiffs_to_trace_directory)
            frame_width, frame_height = trace_write_chunked_tiffs_nodb(
                matfile_directory=db.loc[session, 'matfile_directory'],
                tiffs_to_trace_directory=tiffs_to_trace_directory,
                timestamps_filename=timestamps_filename,
                monitor_video=monitor_video, 
                monitor_video_kwargs=monitor_video_kwargs,
                chunk_size=chunk_size,
                stop_after_frame=stop_after_frame,
                )
            
            # Store the frame_width and frame_height
            db = whiskvid.db.load_db()
            if pandas.isnull(db.loc[session, 'v_width']):
                db.loc[session, 'v_width'] = frame_width
                db.loc[session, 'v_height'] = frame_height
                whiskvid.db.save_db(db)
        
        # Tiffs have been written
        # Now trace the session
        trace_session_nodb(
            h5_filename=db.loc[session, 'wseg_h5'],
            tiffs_to_trace_directory=tiffs_to_trace_directory,
            n_trace_processes=n_trace_processes,
            )

def trace_write_chunked_tiffs_nodb(matfile_directory, tiffs_to_trace_directory,
    timestamps_filename=None, monitor_video=None, monitor_video_kwargs=None,
    chunk_size=None, stop_after_frame=None):
    """Generate a PF reader and call WhiskiWrap.write_video_as_chunked_tiffs
    
    Returns: frame_width, frame_height
    """
    # Generate a PF reader
    pfr = WhiskiWrap.PFReader(matfile_directory)

    # Write the video
    ctw = WhiskiWrap.write_video_as_chunked_tiffs(pfr, tiffs_to_trace_directory,
        chunk_size=chunk_size,
        stop_after_frame=stop_after_frame, 
        monitor_video=monitor_video,
        timestamps_filename=timestamps_filename,
        monitor_video_kwargs=monitor_video_kwargs)    
    
    return pfr.frame_width, pfr.frame_height

def trace_session_nodb(h5_filename, tiffs_to_trace_directory,
    n_trace_processes=8):
    """Trace whiskers from input to output"""
    WhiskiWrap.trace_chunked_tiffs(
        h5_filename=h5_filename,
        input_tiff_directory=tiffs_to_trace_directory,
        n_trace_processes=n_trace_processes,
        )

## Overlays
def make_overlay_image(session, db=None, verbose=True, ax=None):
    """Generates trial_frames_by_type and trial_frames_all_types for session
    
    This is a wrapper around make_overlay_image_nodb that extracts metadata
    and works with the db.

    Calculates, saves, and returns the following:
    
    sess_meaned_frames : pandas dataframe
        containing the meaned image over all trials of each type
        AKA TrialFramesByType
    
    overlay_image_name : 3d color array of the overlays
        This is the sum of all the types in trial_frames_by_type, colorized
        by rewarded side.
        AKA TrialFramesAllTypes
    
    trialnum2frame : dict of trial number to frame

    
    Returns: trialnum2frame, sess_meaned_frames, C
    """
    if db is None:
        db = whiskvid.db.load_db()    

    # Get behavior df
    behavior_filename = db.loc[session, 'bfile']
    lines = ArduFSM.TrialSpeak.read_lines_from_file(db.loc[session, 'bfile'])
    trial_matrix = ArduFSM.TrialSpeak.make_trials_matrix_from_logfile_lines2(lines)
    trial_matrix = ArduFSM.TrialSpeak.translate_trial_matrix(trial_matrix)
    video_filename = db.loc[session, 'vfile']
    b2v_fit = [db.loc[session, 'fit_b2v0'], db.loc[session, 'fit_b2v1']]

    def get_or_generate_filename(file_class):
        db_changed = False
        if pandas.isnull(db.loc[session, file_class.db_column]):
            db.loc[session, file_class.db_column] = \
                file_class.generate_name(db.loc[session, 'session_dir'])
            db_changed = True
        filename = db.loc[session, file_class.db_column]
        
        return filename, db_changed

    # Set up filenames for each
    overlay_image_name, db_changed1 = get_or_generate_filename(
        whiskvid.db.TrialFramesAllTypes)
    trial_frames_by_type_filename, db_changed2 = get_or_generate_filename(
        whiskvid.db.TrialFramesByType)
    trialnum2frame_filename = os.path.join(db.loc[session, 'session_dir'],
        'trialnum2frame.pickle')

    # Load from cache if possible
    if os.path.exists(trialnum2frame_filename):
        if verbose:
            print "loading cached trialnum2frame"
        trialnum2frame = my.misc.pickle_load(trialnum2frame_filename)
    else:
        trialnum2frame = None

    # Call make_overlay_image_nodb
    trialnum2frame, sess_meaned_frames, C = make_overlay_image_nodb(
        trialnum2frame,
        behavior_filename, video_filename, 
        b2v_fit, trial_matrix, verbose=verbose, ax=ax)
    
    # Save
    my.misc.pickle_dump(trialnum2frame, trialnum2frame_filename)
    whiskvid.db.TrialFramesByType.save(trial_frames_by_type_filename,
        sess_meaned_frames)
    whiskvid.db.TrialFramesAllTypes.save(overlay_image_name,
        C)
    
    # Update db
    db = whiskvid.db.load_db()    
    db.loc[session, 'overlays'] = trial_frames_by_type_filename
    db.loc[session, 'frames'] = trialnum2frame_filename
    db.loc[session, 'overlay_image'] = overlay_image_name
    whiskvid.db.save_db(db)     
    
    return trialnum2frame, sess_meaned_frames, C
    
def make_overlay_image_nodb(trialnum2frame=None,
    behavior_filename=None, video_filename=None, 
    b2v_fit=None, trial_matrix=None, verbose=True, ax=None):
    """Make overlays of shapes to show positioning.
    
    Wrapper over the methods in MCwatch.behavior.overlays

    trialnum2frame : if known
        Otherwise, provide behavior_filename, video_filename, and b2v_fit

    Returns:
        trialnum2frame, sess_meaned_frames (DataFrame), C (array)
    """
    # Get trialnum2frame
    if trialnum2frame is None:
        if verbose:
            print "calculating trialnum2frame"
        trialnum2frame = MCwatch.behavior.overlays.extract_frames_at_retraction_times(
            behavior_filename=behavior_filename, 
            video_filename=video_filename, 
            b2v_fit=b2v_fit, 
            verbose=verbose)

    # Calculate sess_meaned_frames
    sess_meaned_frames = MCwatch.behavior.overlays.calculate_sess_meaned_frames(
        trialnum2frame, trial_matrix)

    #~ # Save trial_frames_by_type
    #~ whiskvid.db.TrialFramesByType.save(trial_frames_by_type_filename, resdf)

    # Make figure window
    if ax is None:
        f, ax = plt.subplots(figsize=(6.4, 6.2))

    # Make the trial_frames_all_types and save it
    C = MCwatch.behavior.overlays.make_overlay(sess_meaned_frames, ax, meth='all')
    #~ whiskvid.db.TrialFramesAllTypes.save(overlay_image_name, C)
    
    return trialnum2frame, sess_meaned_frames, C
## End overlays

## 
# correlation with contact
def plot_perf_vs_contacts(session):
    db = whiskvid.db.load_db()
    
    # Load stuff
    res = whiskvid.db.load_everything_from_session(session, db)
    tac = res['tac']
    trial_matrix = res['trial_matrix']
    v2b_fit = res['v2b_fit']

    # Get trial timings
    trial_matrix['choice_time'] = MCwatch.behavior.misc.get_choice_times(
        db.loc[session, 'bfile'])

    # Add trials
    tac = whiskvid.db.add_trials_to_tac(tac, v2b_fit, trial_matrix, 
        drop_late_contacts=True)
    
    # Add # of contacts to trial_matrix
    trial_matrix['n_contacts'] = tac.groupby('trial').apply(len)
    trial_matrix.loc[trial_matrix['n_contacts'].isnull(), 'n_contacts'] = 0

    # Plot histogram of contacts vs hit or error
    f, ax = plt.subplots()
    
    # Split on hits and errors and draw hist for each
    tm_hit = my.pick_rows(trial_matrix, outcome='hit', isrnd=True)
    tm_err = my.pick_rows(trial_matrix, outcome='error', isrnd=True)
    ax.hist([
        np.sqrt(tm_hit.n_contacts.values), 
        np.sqrt(tm_err.n_contacts.values),
        ])
    ax.set_title(session)

    # Plot perf vs some or none contacts
    f, ax = plt.subplots()
    
    # Split on whether contacts occurred
    tm_n_contacts = trial_matrix[
        (trial_matrix.n_contacts == 0) &
        trial_matrix.outcome.isin(['hit', 'error']) &
        trial_matrix.isrnd]
    tm_y_contacts = trial_matrix[
        (trial_matrix.n_contacts > 0) &
        trial_matrix.outcome.isin(['hit', 'error']) &
        trial_matrix.isrnd]    
    
    perf_n_contacts = tm_n_contacts.outcome == 'hit'
    perf_y_contacts = tm_y_contacts.outcome == 'hit'
    data = [perf_n_contacts, perf_y_contacts]
    
    my.plot.vert_bar(ax=ax,
        bar_lengths=map(np.mean, data),
        bar_errs=map(np.std, data),
        bar_colors=('b', 'r'),
        bar_labels=('none', 'some'),
        tick_labels_rotation=0,
        )
    ax.set_ylim((0, 1))
    ax.set_title(session)

def logreg_perf_vs_contacts(session):
    trial_matrix = ArduFSM.TrialMatrix.make_trial_matrix_from_file(
        db.loc[session, 'bfile'])
    tac = whiskvid.db.Contacts.load(db.loc[session, 'tac'])
    v2b_fit = db.loc[session, ['fit_v2b0', 'fit_v2b1']]
    b2v_fit = db.loc[session, ['fit_b2v0', 'fit_b2v1']]
    
    if np.any(pandas.isnull(v2b_fit.values)):
        1/0

    # Get trial timings
    trial_matrix['choice_time'] = MCwatch.behavior.misc.get_choice_times(
        db.loc[session, 'bfile'])
    trial_matrix['vchoice_time'] = np.polyval(b2v_fit, trial_matrix['choice_time'])

    # Add trials
    tac = whiskvid.db.add_trials_to_tac(tac, v2b_fit, trial_matrix, 
        drop_late_contacts=True)

    # Add # of contacts to trial_matrix
    trial_matrix['n_contacts'] = tac.groupby('trial').apply(len)
    trial_matrix.loc[trial_matrix['n_contacts'].isnull(), 'n_contacts'] = 0

    # Drop the ones before video started
    trial_matrix = trial_matrix[trial_matrix.vchoice_time > 0]

    # Choose the random hits
    lr_tm = my.pick_rows(trial_matrix, outcome=['hit', 'error'], isrnd=True)

    # Choose the regularizations
    C_l = [1, .1, .01]

    # Setup input / output
    input = lr_tm['n_contacts'].values[:, None]
    output = (lr_tm['outcome'].values == 'hit').astype(np.int)

    # Transform the input
    input = np.sqrt(input)

    # Values for plotting the decision function 
    plotl = np.linspace(0, input.max(), 100)

    # Bins for actual data
    bins = np.sqrt([0, 1, 4, 8, 16, 32, 64, 128])
    #~ bins = np.linspace(0, input.max(), 4)
    bin_centers = bins[:-1] + 0.5

    # Extract perf of each bin of trials based on # of contacts
    binned_input = np.searchsorted(bins, input.flatten())
    bin_mean_l, bin_err_l = [], []
    for nbin, bin in enumerate(bins):
        mask = binned_input == nbin
        if np.sum(mask) == 0:
            bin_mean_l.append(np.nan)
            bin_err_l.append(np.nan)
        else:
            hits = output[mask]
            bin_mean_l.append(np.mean(hits))
            bin_err_l.append(np.std(hits))
        

    f, axa = plt.subplots(1, len(C_l), figsize=(12, 4))
    for C, ax in zip(C_l, axa):
        lr = scikits.learn.linear_model.LogisticRegression(C=C)
        lr.fit(input, output)#, class_weight='auto')
        ax.plot(plotl, lr.predict_proba(plotl[:, None])[:, 1])
        ax.plot(plotl, np.ones_like(plotl) * 0.5)
        ax.set_ylim((0, 1))
        
        # plot data
        ax.errorbar(x=bins, y=bin_mean_l, yerr=bin_err_l)
    f.suptitle(session)
    plt.show()    

##


## for classifying whiskers
def classify_whiskers_by_follicle_order(mwe, max_whiskers=5,
    fol_y_cutoff=400, short_pixlen_thresh=55, long_pixlen_thresh=150,
    subsample_frame=1, rank_foly_ascending=True,
    oof_y_thresh=5, oof_y_bonus=200):
    """Classify the whiskers by their position on the face
    
    oof_y_thresh : whiskers with a tip_y greater than this will have
        oof_y_bonus added to their length
    rank_foly_ascending : if True, the lowest color is given to the
        larget fol_y (nearest top of frame)
    
    First we apply two length thresholds (one for posterior and one
    for anterior). Then we rank the remaining whisker objects in each
    frame from back to front. 
    
    mwe is returned with a new column 'color_group' with these ranks.
    0 means that the whisker is not in a group.
    1 is the one with minimal y-coordinate.
    Ranks greater than max_whiskers are set to 0.
    
    Debug plots:
    bins = np.arange(orig_mwe.fol_y.min(), orig_mwe.fol_y.max(), 1)
    f, ax = plt.subplots()
    for color, submwe in orig_mwe[orig_mwe.frame < 100000].groupby('color_group'):
        ax.hist(submwe.fol_y.values, bins=bins, histtype='step')

    bins = np.arange(orig_mwe.pixlen.min(), orig_mwe.pixlen.max(), 1)
    f, ax = plt.subplots()
    for color, submwe in orig_mwe[orig_mwe.frame < 100000].groupby('color_group'):
        ax.hist(submwe.pixlen.values, bins=bins, histtype='step')
    
    f, ax = plt.subplots()
    for color, submwe in orig_mwe[orig_mwe.frame < 100000].groupby('color_group'):
        ax.plot(submwe.angle.values, submwe.fol_y.values, ',')    
    """
    print "copying data"
    # Make changes to the copy to avoid SettingWithCopyWarning
    mwe_copy = mwe.copy()

    # Out of frame bonus
    mwe_copy.loc[mwe_copy.tip_y < oof_y_thresh, 'pixlen'] += oof_y_bonus

    # Apply various thresholds
    # Make a second copy here
    mwe_copy2 = mwe_copy[
        ((mwe_copy.pixlen >= long_pixlen_thresh) & 
            (mwe_copy.fol_y < fol_y_cutoff)) | 
        ((mwe_copy.pixlen >= short_pixlen_thresh) & 
            (mwe_copy.fol_y >= fol_y_cutoff))
    ].copy()

    # Subsample to save time
    mwe_copy2 = mwe_copy2[mwe_copy2.frame.mod(subsample_frame) == 0]

    # Argsort each frame
    print "sorting whiskers in order"
    
    # No need to add 1 because rank starts with 1
    mwe_copy2['ordinal'] = mwe_copy2.groupby('frame')['fol_y'].apply(
        lambda ser: ser.rank(method='first', ascending=rank_foly_ascending))

    # Anything beyond C4 is not real
    mwe_copy2.loc[mwe_copy2['ordinal'] > max_whiskers, 'ordinal'] = 0

    # Store the results in the first copy
    mwe_copy['color_group'] = 0
    mwe_copy.loc[mwe_copy2.index, 'color_group'] = \
        mwe_copy2['ordinal'].astype(np.int)
    
    return mwe_copy

##

# This should be moved to edge_summary handler
def normalize_edge_summary(edge_summary):
    """Normalize each trial type in edge_summary and mean.
    
    Also dumps edges with y < 100 because those usually are all the way
    at the top of the frame
    """
    # Sum over each type of trial
    # Normalize each trial type to its max, and then mean
    normalized_es_l = []
    for es in edge_summary['H_l']:
        # Drop everything for which y < 100 (which is the top of the frame)
        es = es.copy()
        es[edge_summary['row_edges'][:-1] < 100] = 0
        
        # Normalize to max
        normalized_es_l.append(es / es.max())
    edge_hist2d = np.mean(normalized_es_l, axis=0)
    
    return edge_hist2d

def add_trial_info_to_video_dataframe(df, trial_matrix, v2b_fit,
    df_column='frame', columns_to_join=None):
    """Join trial information onto each row.
    
    df : dataframe with a column called 'frame'
    trial_matrix : trial matrix for that session
        Should contain a column called 'start_time'
    v2b_fit : fit between video and behavior
    df_column : name of column in df if not 'frame'
    columns_to_join : columns from trial_matrix to join onto df
        If None, the following is used: ['rewside', 'outcome', 'choice_time', 
            'isrnd', 'choice', 'rwin_time',]
    
    The frame numbers will be converted to behavior time. Then each
    row will be associated with a trial based on trial_matrix.start_time.
    Finally the other columns in trial_matrix are joined onto df.
    
    Returns : copy of df with those columns added
    """
    # Columns to join
    if columns_to_join is None:
        columns_to_join = ['rewside', 'outcome', 'choice_time', 
            'isrnd', 'choice', 'rwin_time',]
    
    # Copy
    df = df.copy()

    # Convert to behavior time
    # "vtime" is in the spurious 30fps timebase
    # the fits take this into account
    df['vtime'] = df[df_column] / 30.
    df['btime'] = np.polyval(v2b_fit, df['vtime'].values)
    
    # Associate each row in df with a trial
    df['trial'] = trial_matrix.index[
        np.searchsorted(trial_matrix['start_time'].values, 
            df['btime'].values) - 1]    

    # Add rewside and outcome to df
    df = df.join(trial_matrix[columns_to_join], on='trial')

    return df

def bin_ccs(ccs, locking_times, trial_labels,
    t_start=-2, t_stop=1, nbins=None, binwidth=.005, smoothing_window=None, 
    video_range_bbase=None,
    ):
    """Bin contact times into starts and touching
    
    ccs : colorized_contacts_summary
        Needs to have a column 'whisker' to group by
        Needs to have 'start_btime' and 'stop_btime' to fold on
    
    locking_times : time of each trial to lock on
    
    trial_labels : labels of each trial
    
    t_start, t_stop : time around each locking time to take
    
    nbins : use this many bins
    
    binwidth : if nbins is None, calculate it using this
    
    smoothing_window : if not None, then uses GaussianSmoother with this
        smoothing param instead of np.histogram
    
    video_range_bbase : tuple (start, stop)
        This is the time that the video started and stopped in the
        behavioral timebase. Trials that occurred outside of this range
        are discarded.
    
    Returns: DataFrame
        The columns are a MultiIndex on (metric, whisker, time)
        metric: 'start' or 'touching'
        The index is the trial_labels
    """
    ## Generate bins
    if nbins is None:
        nbins = int(np.rint((t_stop - t_start) / binwidth)) + 1
    bins = np.linspace(t_start, t_stop, nbins)

    # Choose smoothing meth
    if pandas.isnull(smoothing_window):
        meth = np.histogram
    else:
        gs = kkpandas.base.GaussianSmoother(
            smoothing_window=smoothing_window)
        meth = gs.smooth    
    
    def fold_on_trial_times(flat):
        """Helper function to fold starts and stops in the same way"""
        # Fold
        folded = kkpandas.Folded.from_flat(
            flat=flat,
            centers=locking_times,
            dstart=t_start, dstop=t_stop,
            flat_range=video_range_bbase,
            labels=trial_labels
        )
        
        # Bin
        binned = kkpandas.Binned.from_folded_by_trial(
            folded, bins=bins, meth=meth)
        
        # Label rate
        # Because we're folding by trial, # trials is always 1
        # This will be int if meth is np.histogram
        binned_rate = binned.counts
        binned_rate.index = binned.t
        binned_rate.index.name = 'time'
        binned_rate.columns.name = 'trial'

        return binned_rate

    ## Iterate over whiskers and generate starts and touching for each
    starts_l, touching_l, whisker_l = [], [], []
    for whisker, whisker_ccs in ccs.groupby('whisker'):
        # Bin the starts
        binned_starts_rate = fold_on_trial_times(
            np.sort(whisker_ccs.start_btime.values))
        
        # Bin the stops
        binned_stops_rate = fold_on_trial_times(
            np.sort(whisker_ccs.stop_btime.values))

        # Construct touching by cumsumming
        # Need to shift stops so that contacts wholly contained in one bin
        # will still show up
        touching = (binned_starts_rate.cumsum() - 
            binned_stops_rate.cumsum().shift().fillna(0))
        
        # This would fail if a touch was already in progress at the
        # beginning of the window because it might go negative
        assert (touching >= -1e-10).values.all()
        
        # Convert back to int (the shift converts to float)
        if meth is np.histogram:
            touching = touching.astype(np.int)

        # Store
        starts_l.append(binned_starts_rate)
        touching_l.append(touching)
        whisker_l.append(whisker)
    
    ## Form a dataframe indexed by trial and whisker with bins on columns
    starts_df = pandas.concat(starts_l, 
        keys=whisker_l, verify_integrity=True, names=['whisker', 'time'])
    touching_df = pandas.concat(touching_l, 
        keys=whisker_l, verify_integrity=True, names=['whisker', 'time'])   

    # Concatenate all metrics and put trials on the rows
    binned_df = pandas.concat(
        [starts_df, touching_df],
        axis=0, verify_integrity=True,
        keys=['start', 'touching'],
        names=['metric', 'whisker', 'time'],
    ).T
    
    return binned_df
