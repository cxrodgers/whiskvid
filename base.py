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
import BeWatch
import whiskvid
import WhiskiWrap
import matplotlib.pyplot as plt
import pandas

try:
    import tables
except ImportError:
    pass

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

## Syncing
def sync_with_behavior(session, light_delta=30, diffsize=2, refrac=50, 
    **kwargs):
    """Sync video with behavioral file and store in db
    
    Uses decrements in luminance and the backlight signal to do the sync.
    Assumes the backlight decrement is at the time of entry to state 1.
    Assumes video frame rates is 30fps, regardless of actual frame rate.
    And fits the behavior to the video based on that.
    """
    db = whiskvid.db.load_db()
    video_file = db.loc[session, 'vfile']
    bfile = db.loc[session, 'bfile']

    b2v_fit = sync_with_behavior_nodb(
        video_file=video_file,
        bfile=bfile,
        light_delta=light_delta,
        diffsize=diffsize,
        refrac=refrac,
        **kwargs)

    # Save the sync
    db = whiskvid.db.load_db()
    db.loc[session, ['fit_b2v0', 'fit_b2v1']] = b2v_fit
    db.loc[session, ['fit_v2b0', 'fit_v2b1']] = my.misc.invert_linear_poly(
        b2v_fit)
    whiskvid.db.save_db(db)    

def sync_with_behavior_nodb(video_file, bfile, light_delta, diffsize, refrac):
    """Sync video with behavioral file
    
    This got moved to BeWatch.syncing
    """    
    return BeWatch.syncing.sync_video_with_behavior(bfile=bfile,
        lums=None, video_file=video_file, light_delta=light_delta,
        diffsize=diffsize, refrac=refrac, assumed_fps=30.,
        error_if_no_fit=True)

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
    
    Wrapper over the methods in BeWatch.overlays

    trialnum2frame : if known
        Otherwise, provide behavior_filename, video_filename, and b2v_fit

    Returns:
        trialnum2frame, sess_meaned_frames (DataFrame), C (array)
    """
    # Get trialnum2frame
    if trialnum2frame is None:
        if verbose:
            print "calculating trialnum2frame"
        trialnum2frame = BeWatch.overlays.extract_frames_at_retraction_times(
            behavior_filename=behavior_filename, 
            video_filename=video_filename, 
            b2v_fit=b2v_fit, 
            verbose=verbose)

    # Calculate sess_meaned_frames
    sess_meaned_frames = BeWatch.overlays.calculate_sess_meaned_frames(
        trialnum2frame, trial_matrix)

    #~ # Save trial_frames_by_type
    #~ whiskvid.db.TrialFramesByType.save(trial_frames_by_type_filename, resdf)

    # Make figure window
    if ax is None:
        f, ax = plt.subplots(figsize=(6.4, 6.2))

    # Make the trial_frames_all_types and save it
    C = BeWatch.overlays.make_overlay(sess_meaned_frames, ax, meth='all')
    #~ whiskvid.db.TrialFramesAllTypes.save(overlay_image_name, C)
    
    return trialnum2frame, sess_meaned_frames, C
## End overlays


## edge_summary + tac
def plot_tac(video_session, ax=None, versus='rewside', min_t=-2.0, max_t=None,
    **plot_kwargs):
    """Plot the contact locations based on which trial type or response.
    
    whiskvid.db.add_trials_to_tac is used to connect the contact times
    to the behavioral data.
    
    t_min and t_max are passed to tac
    """
    # Get the tac filtered by time
    tac = video_session.data.tac.load_data(min_t=min_t, max_t=max_t)
    
    # Get the video size to set the plot limits
    xlims = (0, video_session.frame_width)
    ylims = (video_session.frame_height, 0)
    
    # Plot defaults
    if 'marker' not in plot_kwargs:
        plot_kwargs['marker'] = 'o'
    if 'mec' not in plot_kwargs:
        plot_kwargs['mec'] = 'none'
    if 'ls' not in plot_kwargs:
        plot_kwargs['ls'] = 'none'

    # Figure handles
    if ax is None:
        f, ax = plt.subplots()
    
    # Two types of plots
    if versus == 'rewside':
        # Plot tac vs rewside
        rewside2color = {'left': 'b', 'right': 'r'}
        gobj = my.pick_rows(tac, 
            choice=['left', 'right'], outcome='hit', isrnd=True).groupby('rewside')
        for rewside, subtac in gobj:
            ax.plot(subtac['tip_x'], subtac['tip_y'],
                color=rewside2color[rewside], **plot_kwargs)
    
    elif versus == 'choice':
        # Plot tac vs choice
        choice2color = {'left': 'b', 'right': 'r'}
        gobj = my.pick_rows(tac, 
            choice=['left', 'right'], isrnd=True).groupby('choice')
        for rewside, subtac in gobj:
            ax.plot(subtac['tip_x'], subtac['tip_y'],
                color=rewside2color[rewside], **plot_kwargs)
    
    else:
        raise ValueError("bad versus: %s" % versus)

    # pretty
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.axis('equal')
    my.plot.rescue_tick(ax=ax, x=4, y=4)
    plt.show()
    
    return ax

def plot_edge_summary(video_session, ax=None, **kwargs):
    """Plot the 2d histogram of edge locations
    
    kwargs are passed to imshow, eg clim or alpha
    The image is stretched to video width and height, regardless of
    histogram edges.
    
    Returns: typical_edges_hist2d, typical_edges_row, typical_edges_col
    """
    # Load the edge summary
    edge_summary = video_session.data.edge_summary.load_data()

    # Get the video size to set the plot limits
    xlims = (0, video_session.frame_width)
    ylims = (0, video_session.frame_height)

    # Get hists
    typical_edges_hist2d = np.sum(edge_summary['H_l'], axis=0)
    typical_edges_row = edge_summary['row_edges']
    typical_edges_col = edge_summary['col_edges']

    # Figure handles
    if ax is None:
        f, ax = plt.subplots()

    # Plot H_l
    im = my.plot.imshow(typical_edges_hist2d, ax=ax,
        xd_range=xlims, yd_range=ylims,
        axis_call='image', cmap=plt.cm.gray_r, **kwargs)
    
    return typical_edges_hist2d, typical_edges_row, typical_edges_col

def video_edge_tac(session, d_temporal=5, d_spatial=1, stop_after_trial=None,
    **kwargs):
    """Make a video with the overlaid edging and contact locations"""
    db = whiskvid.db.load_db()
    
    everything = whiskvid.db.load_everything_from_session(session, db)
    tac = everything['tac']   
    trial_matrix = everything['trial_matrix']
    trial_matrix['choice_time'] = BeWatch.misc.get_choice_times(
        db.loc[session, 'bfile'])
    choice_btime = np.polyval(everything['b2v_fit'], trial_matrix['choice_time'])
    trial_matrix['choice_bframe'] = np.rint(choice_btime * 30)    

    # Get hists
    typical_edges_hist2d = np.sum(everything['edge_summary']['H_l'], axis=0)
    typical_edges_row = everything['edge_summary']['row_edges']
    typical_edges_col = everything['edge_summary']['col_edges']

    video_filename = db.loc[session, 'vfile']
    output_filename = whiskvid.db.ContactVideo.generate_name(
        db.loc[session, 'session_dir'])
    
    frame_triggers = trial_matrix['choice_bframe'].values
    if stop_after_trial is not None:
        frame_triggers = frame_triggers[:stop_after_trial]

    whiskvid.output_video.dump_video_with_edge_and_tac(
        video_filename, typical_edges_hist2d, tac, everything['edge_a'],
        output_filename, frame_triggers,
        d_temporal=d_temporal, d_spatial=d_spatial, **kwargs)
    
    db.loc[session, 'contact_video'] = output_filename
    whiskvid.db.save_db(db)

def write_video_with_overlays(session):
    """Wrapper around output_video.write_video_with_overlays"""
    pass

## end edge_summary + tac

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
    trial_matrix['choice_time'] = BeWatch.misc.get_choice_times(
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
    trial_matrix['choice_time'] = BeWatch.misc.get_choice_times(
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
    subsample_frame=1):
    """Classify the whiskers by their position on the face
    
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
    orig_mwe = mwe.copy()

    # Apply various thresholds
    mwe = mwe[
        ((mwe.pixlen >= long_pixlen_thresh) & (mwe.fol_y < fol_y_cutoff)) | 
        ((mwe.pixlen >= short_pixlen_thresh) & (mwe.fol_y >= fol_y_cutoff))
    ]

    # Subsample to save time
    mwe = mwe[mwe.frame.mod(subsample_frame) == 0]

    # Argsort each frame
    print "sorting whiskers in order"
    
    # No need to add 1 because rank starts with 1
    mwe['ordinal'] = mwe.groupby('frame')['fol_y'].apply(
        lambda ser: ser.rank(method='first'))

    # Anything beyond C4 is not real
    mwe.loc[mwe['ordinal'] > max_whiskers, 'ordinal'] = 0

    orig_mwe['color_group'] = 0
    orig_mwe.loc[mwe.index, 'color_group'] = mwe['ordinal'].astype(np.int)
    
    return orig_mwe

##

def get_triggered_whisker_angle(vsession, **kwargs):
    """Load the whisker angle from mwe and trigger on trial times
    
    This is a wrapper around get_triggered_whisker_angle_nodb
    """
    # Get masked whisker ends
    mwe = vsession.data.whiskers.load_data()

    # Get sync
    v2b_fit = vsession.fit_v2b
    
    # Get trial matrix
    bsession = vsession.bsession_name
    tm = BeWatch.db.get_trial_matrix(bsession, True)
    
    # Trigger
    twa = get_triggered_whisker_angle_nodb(mwe, v2b_fit, tm, **kwargs)
    
    return twa

def get_triggered_whisker_angle_nodb(mwe, v2b_fit, tm, relative_time_bins=None):
    """Load the whisker angle from mwe and trigger on trial times
    
    The angle is meaned over whiskers by frame.
    
    relative_time_bins: timepoints at which to infer whisker angle
    """
    if relative_time_bins is None:
        relative_time_bins = np.arange(-3.5, 5, .05)
    
    ## mean angle over whiskers by frame
    angle_by_frame = mwe.groupby('frame')['angle'].mean()
    angle_vtime = angle_by_frame.index.values / 30.
    angle_btime = np.polyval(v2b_fit, angle_vtime)

    ## Now extract mean angle for each RWIN open time
    # convert rwin_open_time to seconds
    rwin_open_times_by_trial = tm['rwin_time']

    # Index the angle based on the btime
    angle_by_btime = pandas.Series(index=angle_btime, 
        data=angle_by_frame.values)
    angle_by_btime.index.name = 'btime'

    ## Interpolate angle_by_btime at the new time bins that we want
    # Get absolute time bins
    trigger_times = rwin_open_times_by_trial.dropna()
    absolute_time_bins_l = []
    for trial, trigger_time in trigger_times.iteritems():
        # Get time bins relative to trigger
        absolute_time_bins = relative_time_bins + trigger_time
        absolute_time_bins_l.append(absolute_time_bins)
    
    # Drop the ones before and after data
    # By default pandas interpolate fills forward but not backward
    absolute_time_bins_a = np.concatenate(absolute_time_bins_l)
    absolute_time_bins_a_in_range = absolute_time_bins_a[
        (absolute_time_bins_a < angle_by_btime.index.values.max()) &
        (absolute_time_bins_a > angle_by_btime.index.values.min())
    ].copy()
    
    # Make a bigger index with positions for each of the desired time bins
    # Ensure it doesn't contain duplicates
    new_index = (angle_by_btime.index | 
        pandas.Index(absolute_time_bins_a_in_range))
    new_index = new_index.drop_duplicates()
    
    # Interpolate
    resampled_session = angle_by_btime.reindex(new_index).interpolate('index')
    assert not np.any(resampled_session.isnull())

    ## Extract interpolated times for each trial
    # Take interpolated values at each of the absolute time bins
    # Will be NaN before and after the data
    interpolated = resampled_session.ix[absolute_time_bins_a]
    assert interpolated.shape == absolute_time_bins_a.shape

    ## Reshape
    # One column per trial
    twa = pandas.DataFrame(
        interpolated.values.reshape(
            (len(trigger_times), len(relative_time_bins))).T,
        index=relative_time_bins, columns=trigger_times.index
    )

    # Drop trials with any missing data (should be at beginning and end)
    twa = twa.dropna(1)
    return twa


## For clustering contacts
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

def cluster_contacts(vsession, **kwargs):
    """Cluster the contacts using the database and save to disk
    
    If a file called "clustered_tac" already exists in the session 
    directory, it is loaded and returned immediately.
    
    Otherwise, the tac is loaded, and cluster_contacts_nodb() is called.
    Then the clustered_tac is saved to disk and returned.
    """
    # Determine the filename
    db = whiskvid.db.load_db()
    tac_clustered_name = os.path.join(db.loc[vsession, 'session_dir'],
        'clustered_tac')
    
    # If it exists, return
    if os.path.exists(tac_clustered_name):
        print "using cached clustered_tac"
        tac_clustered = pandas.read_pickle(tac_clustered_name)
        return tac_clustered

    # Otherwise do the clustering
    tac = whiskvid.get_tac(vsession)
    tac_clustered = cluster_contacts_nodb(tac, **kwargs)

    # Store
    print "saving clustered_tac to", tac_clustered_name
    tac_clustered.to_pickle(tac_clustered_name)
    
    return tac_clustered

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