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
import traj, trace
import numpy as np, pandas
import os
import scipy.ndimage
import my
import ArduFSM
import BeWatch
import whiskvid
import matplotlib.pyplot as plt
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



## Functions for extracting objects from video

def get_object_size_and_centroid(objects):
    """Returns size and centroid of every object.
    
    objects : first result from scipy.ndimage.label
    
    Returns: szs, centroids
        szs : array of object sizes, starting with the object labeled 0
            (which is usually the background) and continuing through all
            available objects
        centroids : same, but for the centroids. This will be a Nx2 array.
    """
    # Find out which objects are contained
    object_ids = np.unique(objects)
    assert np.all(object_ids == np.arange(len(object_ids)))
    
    # Get size and centroid of each
    szs, centroids = [], []
    for object_id in object_ids:
        # Get its size and centroid and store
        sz = np.sum(objects == object_id)
        szs.append(sz)

    # Get the center of mass
    # This is the bottleneck step of the whole process
    # center_of_mass is not really any faster than manually calculating
    # we switch x and y for backwards compat
    # Maybe (arr.mean(0) * arr.sum(0)).mean() to get a weighted average in y?
    centroids2 = np.asarray(scipy.ndimage.center_of_mass(
        objects, objects, object_ids))
    centroids3 = centroids2[:, [1, 0]]
    
    return np.asarray(szs), centroids3

def is_centroid_in_roi(centroid, roi_x, roi_y):
    """Returns True if the centroid is in the ROI.
    
    centroid : x, y
    roi_x : x_min, x_max
    roi_y : y_min, y_max
    """
    return (
        centroid[0] >= roi_x[0] and centroid[0] < roi_x[1] and
        centroid[1] >= roi_y[0] and centroid[1] < roi_y[1]
        )

def get_left_edge(object_mask):
    """Return the left edge of the object.
    
    Currently, for each row, we take the left most nonzero value. We
    return (row, col) for each such pixel. However, this doesn't work for
    horizontal parts of the edge.
    """
    contour = []
    for nrow, row in enumerate(object_mask):
        true_cols = np.where(row)[0]
        if len(true_cols) == 0:
            continue
        else:
            contour.append((nrow, true_cols[0]))
    return np.asarray(contour)

def get_bottom_edge(object_mask):
    """Return the bottom edge of the object.
    
    Currently, for each column, we take the bottom most nonzero value. We
    return (row, col) for each such pixel. However, this doesn't work for
    vertical parts of the edge.
    """
    contour = []
    for ncol, col in enumerate(object_mask.T):
        true_rows = np.where(col)[0]
        if len(true_rows) == 0:
            continue
        else:
            contour.append((true_rows[-1], ncol))
    return np.asarray(contour)

def plot_all_objects(objects, nobjects):
    f, axa = plt.subplots(1, nobjects)
    for object_id in range(nobjects):
        axa[object_id].imshow(objects == object_id)
    plt.show()

def find_edge_of_shape(frame, lum_threshold=30, roi_x=(320, 640),
    roi_y=(0, 480), size_threshold=10000, edge_getter=get_bottom_edge,
    meth='largest_in_roi', split_iters=10, debug=False):
    """Find the left edge of the shape in frame.
    
    This is a wrapper around the other utility functions
    
    1. Thresholds image to find dark spots
    2. Segments the dark spots using scipy.ndimage.label
    3. Chooses the largest dark spot within the ROI
    4. Finds the left edge of this spot
    
    meth: largest_with_centroid_in_roi, largest_in_roi
    
    If debug: returns binframe, best_object, edge
    
    Returns: bottom edge, as sequence of (y, x) (or row, col) pairs
        If no acceptable object is found, returns None.
    """
    # Segment image
    binframe = frame < lum_threshold
    
    # Split apart the pipes and the shape
    opened_binframe = scipy.ndimage.morphology.binary_opening(
        binframe, iterations=split_iters)
    
    # Label them
    objects, nobjects = scipy.ndimage.label(opened_binframe)

    if meth == 'largest_with_centroid_in_roi':
        # Get size and centroid of each object
        szs, centroids = get_object_size_and_centroid(objects)

        # Find which objects are in the ROI
        mask_is_in_roi = np.asarray([is_centroid_in_roi(centroid,
            roi_x, roi_y) for centroid in centroids])

    # Get largest object that is anywhere in roi
    if meth == 'largest_in_roi':
        szs = np.array([np.sum(objects == nobject) for nobject in range(nobjects + 1)])
        subframe = objects[
            np.min(roi_y):np.max(roi_y),
            np.min(roi_x):np.max(roi_x)]
        is_in_roi = np.unique(subframe)
        mask_is_in_roi = np.zeros(nobjects + 1, dtype=np.bool)
        mask_is_in_roi[is_in_roi] = True

    # Choose the largest one in the ROI that is not background
    mask_is_in_roi[0] = 0 # do not allow background
    if np.sum(mask_is_in_roi) == 0:
        #raise ValueError("no objects found in ROI")
        if debug:
            return binframe, None, None
        else:
            return None
    best_id = np.where(mask_is_in_roi)[0][np.argmax(szs[mask_is_in_roi])]

    # Error if no object found above sz 10000 (100x100)
    if szs[best_id] < 10000:
        #raise ValueError("all objects in the ROI are too small")
        if debug:
            return binframe, None, None
        else:
            return None

    # Get the contour of the object
    best_object = objects == best_id
    edge = edge_getter(best_object)

    if debug:
        return binframe, best_object, edge
    else:
        return edge

def get_all_edges_from_video(video_file, n_frames=np.inf, verbose=True,
    lum_threshold=50, roi_x=(200, 500), roi_y=(0, 400),
    return_frames_instead=False, meth='largest_in_roi', split_iters=10,
    side='left', debug=False, debug_frametimes=None):
    """Function that captures video frames and calls find_edge_of_shape.
    
    The normal function is to call process_chunks_of_video on the whole
    video. Alternatively, the raw frames can be returned (to allow the user
    to set parameters). Or, it can be run in debug mode, which returns
    all intermediate computations (and uses get_frame instead of
    process_chunks_of_video).
    
    return_frames_instead : for debugging. If True, return the raw frames
        instead of the edges
    side : Must be either 'left' or 'top'
        If 'left', then uses get_bottom_edge
        If 'top', then uses get_left_edge
    debug : If True, then enter debug mode which is slower but allows
        debugging. Gets individual frames with my.video.get_frame instead
        of processing chunks using my.video.process_chunks_of_video.
        Passes debug=True to find_edge_of_shape in order to extract
        the intermediate results, like thresholded shapes and best objects.
    
    Returns: edge_a
    """
    # Set the edge_getter using the side
    if side == 'left':
        edge_getter = get_bottom_edge
    elif side == 'top':
        edge_getter = get_left_edge
    else:
        raise ValueError("side must be left or top, instead of %r" % side)
    
    if not debug:
        # Helper function to pass to process_chunks_of_video
        def mapfunc(frame):
            """Gets the edge from each frame"""
            edge = find_edge_of_shape(frame, lum_threshold=lum_threshold,
                roi_x=roi_x, roi_y=roi_y, edge_getter=edge_getter,
                meth=meth, split_iters=split_iters)
            if edge is None:
                return None
            else:
                return edge.astype(np.int16)

        if return_frames_instead:
            mapfunc = 'keep'

        # Get the edges
        edge_a = my.video.process_chunks_of_video(video_file, 
            n_frames=n_frames,
            func=mapfunc,
            verbose=verbose,
            finalize='listcomp')

        return edge_a
    
    else:
        if debug_frametimes is None:
            raise ValueError("must specify debug frametimes")
        
        # Value to return
        res = {'frames': [], 'binframes': [], 'best_objects': [], 'edges': []}
        
        # Iterate over frames
        for frametime in debug_frametimes:
            # Get the frame
            frame, stdout, stderr = my.video.get_frame(video_file, frametime)
            
            # Compute the edge and intermediate results
            binframe, best_object, edge = find_edge_of_shape(
                frame, lum_threshold=lum_threshold,
                roi_x=roi_x, roi_y=roi_y, edge_getter=edge_getter,
                meth=meth, split_iters=split_iters, debug=True)
            
            if edge is not None:
                edge = edge.astype(np.int16)
            
            # Store and return
            res['frames'].append(frame)
            res['binframes'].append(binframe)
            res['best_objects'].append(best_object)
            res['edges'].append(edge)
        
        return res

def plot_edge_subset(edge_a, stride=200, xlim=(0, 640), ylim=(480, 0)):
    """Overplot the edges to test whether they were detected"""
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    for edge in edge_a[::stride]:
        if edge is not None:
            ax.plot(edge[:, 1], edge[:, 0])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()



def edge_frames_manual_params_db(session, interactive=True, **kwargs):
    """Interactively set lum thresh and roi for edging
    
    This ROI will be used to identify which of the dark shapes in the object
    is the stimulus. Typically, we choose the largest shape that has any
    part of itself in the ROI. Thus, choose the ROI such that the face is never
    included, but some part of the shape is always included.
    
    Requires: row['vfile'] to exist
    Sets: edge_roi_x, edge_roi_y, edge_lumthresh
    """
    # Get metadata
    db = whiskvid.db.load_db()
    db_changed = False
    row = db.ix[session]
    
    # Get manual params
    if pandas.isnull(row['vfile']):
        raise ValueError("no vfile for", session)
    params = edge_frames_manual_params(row['vfile'], 
        interactive=interactive, **kwargs)
    
    # Save in db
    for key, value in params.items():
        if key in db:
            if not pandas.isnull(db.loc[session, key]):
                print "warning: overwriting %s in %s" % (key, session)
        else:
            print "warning: adding %s as a param" % key
        db.loc[session, key] = value
        db_changed = True
    
    # Save db
    if db_changed:
        whiskvid.db.save_db(db)     
    else:
        print "no changes made to edge in", session


def edge_frames_manual_params(video_file, interactive=True, **kwargs):
    width, height = my.video.get_video_aspect(video_file)
    
    # Try to find a frame with a good example of a shape
    def keep_roi(frame):
        height, width = frame.shape
        return frame[:int(0.5 * height), int(0.5 * width):]
    frames_a = my.video.process_chunks_of_video(video_file, n_frames=10000,
        func='keep', frame_chunk_sz=1000, verbose=True, finalize='listcomp')
    idxs = np.argsort([keep_roi(frame).min() for frame in frames_a])
    best_frame = frames_a[idxs[0]]

    # Plot it so we can set params
    f, axa = plt.subplots(3, 3)
    for good_frame, ax in zip(frames_a[idxs[::100]], axa.flatten()):
        im = my.plot.imshow(good_frame, axis_call='image', ax=ax)
        im.set_clim((0, 255))
        my.plot.colorbar(ax=ax)
        my.plot.rescue_tick(ax=ax, x=4, y=5)
    plt.show()

    # Get the shape roi
    res = my.video.choose_rectangular_ROI(video_file, interactive=interactive,
        **kwargs)
    #~ if len(res) == 0:
        #~ return res
    
    # Rename the keys
    res2 = {}
    for key in res:
        res2['edge_roi_' + key] = res[key]

    # Get the lum_threshold
    lumthresh_s = raw_input("Enter lum threshold (eg, 50): ")
    lumthresh_int = int(lumthresh_s)
    res2['edge_lumthresh'] = lumthresh_int

    # Get the face side
    side_s = raw_input(
        "Enter face side (eg, 'top', 'left' but without quotes): ")
    res2['side'] = side_s

    #~ ## replot figure with params
    #~ f, axa = plt.subplots(3, 3)
    #~ for good_frame, ax in zip(frames_a[idxs[::100]], axa.flatten()):
        #~ im = my.plot.imshow(good_frame > lum_threshold, axis_call='image', ax=ax)
        #~ ax.plot(ax.get_xlim(), [roi_y[1], roi_y[1]], 'w:')
        #~ ax.plot([roi_x[0], roi_x[0]], ax.get_ylim(), 'w:')
        #~ my.plot.colorbar(ax=ax)
        #~ my.plot.rescue_tick(ax=ax, x=4, y=5)
    #~ plt.show()

    return res2


def edge_frames(session, db=None, debug=False, **kwargs):
    """Edges the frames and updates db
    
    If debug: returns frames, edge_a and does not update db
    """
    if db is None:
        db = whiskvid.db.load_db()
    row = db.ix[session]
    
    # Generate output file name
    if pandas.isnull(db.loc[session, 'edge']):
        output_file = whiskvid.db.EdgesAll.generate_name(row['session_dir'])
        db.loc[session, 'edge'] = output_file
    
    # A better default for side
    if 'side' in kwargs:
        side = kwargs.pop('side')
    elif pandas.isnull(row['side']):
        print "warning: side is null, using left"
        side = 'left'
    else:
        side = row['side']
    
    # Form the params
    kwargs = kwargs.copy()
    for kwarg in ['edge_roi_x0', 'edge_roi_x1', 
        'edge_roi_y0', 'edge_roi_y1']:
        kwargs[kwarg] = row[kwarg]
    kwargs['lum_threshold'] = row['edge_lumthresh']
    
    # Depends on debug
    if debug:
        frames, edge_a = edge_frames_nodb(
            row['vfile'], db.loc[session, 'edge'], side=side, debug=True,
            **kwargs)
        
        return frames, edge_a
    else:
        edge_frames_nodb(
            row['vfile'], db.loc[session, 'edge'], side=side, debug=False,
            **kwargs)

        # Save
        whiskvid.db.save_db(db)  

def edge_frames_nodb(video_file, edge_file, 
    lum_threshold, edge_roi_x0, edge_roi_x1, edge_roi_y0, edge_roi_y1, 
    split_iters=13, n_frames=np.inf, 
    stride=100, side='left', meth='largest_in_roi', debug=False, **kwargs):
    """Edge all frames and save to edge_file. Also plot_edge_subset.
    
    This is a wrapper around get_all_edges_from_video_file which does the
    actual edging. This function parses the inputs for it, and handles the
    saving to disk and the plotting of the edge subset.
    
    debug : If True, then this will extract frames and edges from a subset
        of the frames and display / return them for debugging of parameters.
        In this case, returns frames, edge_a
    """
    # Get video aspect
    width, height = my.video.get_video_aspect(video_file)
    
    # Form the kwargs that we will use for the call
    kwargs = kwargs.copy()
    kwargs['n_frames'] = n_frames
    kwargs['lum_threshold'] = lum_threshold
    kwargs['roi_x'] = (edge_roi_x0, edge_roi_x1)
    kwargs['roi_y'] = (edge_roi_y0, edge_roi_y1)
    kwargs['meth'] = 'largest_in_roi'
    kwargs['split_iters'] = split_iters
    kwargs['side'] = side
    
    # Depends on debug
    if debug:
        # Set parameters for debugging
        if np.isinf(n_frames):
            kwargs['n_frames'] = 1000
            print "debug mode; lowering n_frames"
        
        # Get raw frames
        frames = whiskvid.get_all_edges_from_video(video_file,
            return_frames_instead=True, **kwargs)        
        
        # Get edges from those frames
        edge_a = whiskvid.get_all_edges_from_video(video_file,
            return_frames_instead=False, **kwargs)
        
        return frames, edge_a
    
    else:
        # Get edges
        edge_a = whiskvid.get_all_edges_from_video(video_file,
            return_frames_instead=False, **kwargs)

        # Save
        np.save(edge_file, edge_a)

        # Plot
        whiskvid.plot_edge_subset(edge_a, stride=stride,    
            xlim=(0, width), ylim=(height, 0))

def purge_edge_frames(session, db=None):
    """Delete the results of the edged frames.
    
    Probably you want to purge the edge summary as well.
    """
    # Get the filename
    if db is None:
        db = whiskvid.db.load_db()
    row = db.ix[session]
    edge_file = db.loc[session, 'edge']
    
    # Try to purge it
    if pandas.isnull(edge_file):
        print "no edge file to purge"
    elif not os.path.exists(edge_file):
        print "cannot find edge file to purge: %r" % edge_file
    else:
        os.remove(edge_file)
    
def edge_frames_debug_plot(session):
    """This is a helper function for debugging edging.
    
    For some subset of frames, plot the raw frame, the detected edge, maybe
    the thresholded frame and the detected objects as well.
    """
    # Extract raw frames with edges
    frames, edge_a = whiskvid.edge_frames(
        session, verbose=True, debug=True)    
    db = whiskvid.db.load_db()
    
    # Plot them
    f, axa = plt.subplots(5, 5)
    nax = 0
    stride = 10
    for nframe, (frame, edge) in enumerate(zip(frames[::stride], edge_a[::stride])):
        #~ # This is a good way to test
        #~ binframe, best_object, edge = whiskvid.find_edge_of_shape(frame, 
            #~ lum_threshold=50, roi_x=(150, 390), roi_y=(150, 390), 
            #~ edge_getter=whiskvid.base.get_left_edge, debug=True)
        
        if edge is None:
            continue
        ax = axa.flatten()[nax]
        nax = nax + 1
        if nax == len(axa.flatten()):
            break
        v_width, v_height = db.loc[session, 'v_width'], db.loc[session, 'v_height']
        im2 = my.plot.imshow(frame, ax=ax, 
            axis_call='image', cmap=plt.cm.gray, extent=(0, v_width, v_height, 0))
        ax.plot(edge[:, 1], edge[:, 0], 'g-')
        ax.set_title("frame %d" % nframe)
        plt.show()    

## End of functions for extracting objects from video


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


def get_whisker_ends_hdf5(hdf5_file=None, side=None, 
    also_calculate_length=True):
    """Reimplement get_whisker_ends on hdf5 file"""
    import tables
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


## cropping
def crop_manual_params_db(session, interactive=True, **kwargs):
    """Get crop size and save to db"""
    # Get metadata
    db = whiskvid.db.load_db()
    db_changed = False
    row = db.ix[session]
    
    # Get manual params
    if pandas.isnull(row['input_vfile']):
        raise ValueError("no input_vfile for", session)
    params = crop_manual_params(row['input_vfile'], 
        interactive=interactive, **kwargs)
    
    # Save in db
    for key, value in params.items():
        if not pandas.isnull(db.loc[session, key]):
            print "warning: overwriting %s in %s" % (key, session)
        db.loc[session, key] = value
        db_changed = True
    
    # Save db
    if db_changed:
        whiskvid.db.save_db(db)     
    else:
        print "no changes made to crop in", session

def crop_manual_params(vfile, interactive=True, **kwargs):
    """Use choose_rectangular_ROI to set cropping params"""
    res = my.video.choose_rectangular_ROI(vfile, interactive=interactive,
        **kwargs)
    
    if len(res) == 0:
        return res
    
    # Rename the keys
    res2 = {}
    for key in res:
        res2['crop_' + key] = res[key]
    return res2    

def crop_session(session, db=None, **kwargs):
    """Crops the input file into the output file, and updates db"""
    if db is None:
        db = whiskvid.db.load_db()
    row = db.ix[session]
    
    # Generate output file name
    if pandas.isnull(db.loc[session, 'vfile']):
        output_file = os.path.join(row['session_dir'], session + '_cropped.mp4')
        db.loc[session, 'vfile'] = output_file
    
    crop_session_nodb(row['input_vfile'], db.loc[session, 'vfile'],
        row['crop_x0'], row['crop_x1'], row['crop_y0'], row['crop_y1'],
        **kwargs)

    # Save
    whiskvid.db.save_db(db)  

def crop_session_nodb(input_file, output_file, crop_x0, crop_x1, 
    crop_y0, crop_y1, **kwargs):
    """Crops the input file into the output file"""
    my.video.crop(input_file, output_file, crop_x0, crop_x1, 
        crop_y0, crop_y1, **kwargs)

## end cropping

## tracing
def trace_session(session, db=None, **kwargs):
    """Crops the input file into the output file, and updates db"""
    if db is None:
        db = whiskvid.db.load_db()
    row = db.ix[session]
    
    # Generate output file name
    if pandas.isnull(db.loc[session, 'whiskers']):
        output_file = whiskvid.db.Whiskers.generate_name(row['session_dir'])
        db.loc[session, 'whiskers'] = output_file

    # Save right away, to avoid stale db
    whiskvid.db.save_db(db)  

    trace_session_nodb(row['vfile'], db.loc[session, 'whiskers'])
    

def trace_session_nodb(input_file, output_file, verbose=False):
    """Trace whiskers from input to output"""
    cmd = 'trace %s %s' % (input_file, output_file)
    run_dir = os.path.split(input_file)[0]
    run_dir2 = os.path.split(output_file)[0]
    if run_dir != run_dir2:
        raise ValueError("warning: trace I/O needs same dir")
    if verbose:
        print cmd
    
    orig_dir = os.getcwd()
    os.chdir(run_dir)
    try:        
        os.system(cmd)
    except:
        raise
    finally:
        os.chdir(orig_dir)

## Calculating contacts
def calculate_contacts_manual_params_db(session, **kwargs):
    """Gets manual params and saves to db"""
    # Get metadata
    db = whiskvid.db.load_db()
    row = db.ix[session]
    
    # Get manual params
    params = calculate_contacts_manual_params(row['vfile'], interactive=True, 
        **kwargs)
    for key, value in params.items():
        db.loc[session, key] = value
    
    # Save
    whiskvid.db.save_db(db)  

def calculate_contacts_manual_params(vfile, n_frames=4, interactive=False):
    """Display a subset of video frames to set fol_x and fol_y"""
    res = my.video.choose_rectangular_ROI(vfile, n_frames=n_frames, 
        interactive=interactive)
    
    if len(res) == 0:
        return res
    
    # Rename the keys
    res2 = {}
    for key in res:
        res2['fol_' + key] = res[key]
    return res2
    
def calculate_contacts_session(session, db=None, **kwargs):
    """Calls `calculate_contacts` on `session`"""
    if db is None:
        db = whiskvid.db.load_db()
    row = db.ix[session]
    
    if pandas.isnull(row['tac']):
        db.loc[session, 'tac'] = whiskvid.db.Contacts.generate_name(
            row['session_dir'])
    tac = calculate_contacts(row['wseg_h5'], row['edge'], row['side'], 
        tac_filename=db.loc[session, 'tac'], 
        fol_range_x=(db.loc[session, 'fol_x0'], db.loc[session, 'fol_x1']),
        fol_range_y=(db.loc[session, 'fol_y0'], db.loc[session, 'fol_y1']),
        **kwargs)
    
    whiskvid.db.save_db(db)

def calculate_contacts(h5_filename, edge_file, side, tac_filename=None,
    length_thresh=75, contact_dist_thresh=10,
    fol_range_x=(0, 70), fol_range_y=(250, 360),
    verbose=True):
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

    # Get the edges
    edge_a = np.load(edge_file)

    # Find the contacts
    # For every frame, iterate through whiskers and compare to shape
    contacts_l = []
    for frame, frame_tips in resdf.groupby('frame'):
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
    tips_and_contacts = resdf.join(contacts_df.set_index('index'))
    tips_and_contacts = tips_and_contacts[
        tips_and_contacts.closest_dist < contact_dist_thresh]
    if not pandas.isnull(tac_filename):
        tips_and_contacts.to_pickle(tac_filename)
    return tips_and_contacts

def purge_tac(session, db=None):
    """Delete the tac"""
    # Get the filename
    if db is None:
        db = whiskvid.db.load_db()
    row = db.ix[session]
    tac_file = db.loc[session, 'tac']
    
    # Try to purge it
    if pandas.isnull(tac_file):
        print "no tac file to purge"
    elif not os.path.exists(tac_file):
        print "cannot find tac file to purge: %r" % tac_file
    else:
        os.remove(tac_file)

## End calculating contacts



## Edge summary dumping
def dump_edge_summary(session, db=None, **kwargs):
    """Calls `dump_edge_summary_nodb` on `session`"""
    if db is None:
        db = whiskvid.db.load_db()
    
    # Get behavior df
    bfile_name = db.loc[session, 'bfile']
    if pandas.isnull(bfile_name) or not os.path.exists(bfile_name):
        raise IOError("cannot find bfile for %s" % session)
    trial_matrix = ArduFSM.TrialMatrix.make_trial_matrix_from_file(bfile_name)
    if 'choice_time' not in trial_matrix:
        trial_matrix['choice_time'] = BeWatch.misc.get_choice_times(bfile_name)

    # Get edges
    edge_a = whiskvid.db.EdgesAll.load(db.loc[session, 'edge'])
    b2v_fit = np.asarray(db.loc[session, ['fit_b2v0', 'fit_b2v1']])
    v_width, v_height = my.video.get_video_aspect(db.loc[session, 'vfile'])
    
    # Set up edge summary filename
    db_changed = False
    if pandas.isnull(db.loc[session, 'edge_summary']):
        db.loc[session, 'edge_summary'] = whiskvid.db.EdgesSummary.generate_name(
            db.loc[session, 'session_dir'])
        db_changed = True
    edge_summary_filename = db.loc[session, 'edge_summary']
    
    # Dump edge summary
    dump_edge_summary_nodb(trial_matrix, edge_a, b2v_fit, v_width, v_height,
        edge_summary_filename=edge_summary_filename,
        **kwargs)
    
    if db_changed:
        whiskvid.db.save_db(db)
    
def dump_edge_summary_nodb(trial_matrix, edge_a, b2v_fit, v_width, v_height,
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
## End edge summary dumping

## Frame dumping
def dump_frames(session, db=None):
    """Calls `dump_frames_nodb` on `session`"""
    if db is None:
        db = whiskvid.db.load_db()    

    # Get behavior df
    bfilename = db.loc[session, 'bfile']
    b2v_fit = np.asarray(db.loc[session, ['fit_b2v0', 'fit_b2v1']])
    video_file = db.loc[session, 'vfile']
    
    # Set up filename
    db_changed = False
    if pandas.isnull(db.loc[session, 'frames']):
        db.loc[session, 'frames'] = whiskvid.db.TrialFramesDir.generate_name(
            db.loc[session, 'session_dir'])
        db_changed = True
    frame_dir = db.loc[session, 'frames']
    
    # Dump frames   
    dump_frames_nodb(bfilename, b2v_fit, video_file, frame_dir)

    if db_changed:
        whiskvid.db.save_db(db)

def dump_frames_nodb(bfilename, b2v_fit, video_file, frame_dir):
    """Dump frames at servo retraction time
    
    Wrapper around BeWatch.overlays.dump_frames_at_retraction_time
    """
    # overlays
    duration = my.video.get_video_duration(video_file)
    metadata = {'filename': bfilename, 'fit0': b2v_fit[0], 'fit1': b2v_fit[1],
        'guess_vvsb_start': 0, 'filename_video': video_file, 
        'duration_video': duration * np.timedelta64(1, 's')}
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)
        BeWatch.overlays.dump_frames_at_retraction_time(metadata, frame_dir)
    else:
        print "not dumping frames, %s already exists" % frame_dir
## End frame dumping

## Overlays
def make_overlay_image(session, db=None, ax=None):
    """Generates trial_frames_by_type and trial_frames_all_types for session
    
    This is a wrapper around make_overlay_image_nodb that extracts metadata
    and works with the db.
    
    Returns: trial_frames_by_type, trial_frames_all_types
    """
    if db is None:
        db = whiskvid.db.load_db()    

    # Get behavior df
    bfile_name = db.loc[session, 'bfile']
    trial_matrix = ArduFSM.TrialMatrix.make_trial_matrix_from_file(bfile_name)
    frame_dir = db.loc[session, 'frames']

    # Error check
    # Should this be a cascading call to dump_frames?
    if pandas.isnull(frame_dir) or not os.path.exists(frame_dir):
        raise ValueError("frames must be dumped before making overlays")

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

    # Call make_overlay_image_nodb
    trial_frames_by_type, trial_frames_all_types = \
        make_overlay_image_nodb(trial_frames_by_type_filename,
        overlay_image_name, frame_dir, trial_matrix, ax=ax)
    
    # Save db
    if db_changed1 or db_changed2:
        whiskvid.db.save_db(db)     
    else:
        print "no changes made to overlay filenames in db for", session    
    
    return trial_frames_by_type, trial_frames_all_types
    
def make_overlay_image_nodb(trial_frames_by_type_filename,
    overlay_image_name, frame_dir, trial_matrix, ax=None):
    """Make overlays of shapes to show positioning.
    
    Wrapper over the methods in BeWatch.overlays
    
    trial_frames_by_type_filename : where to save the pandas dataframe
        containing the meaned image over all trials of each type
    overlay_image_name : where to save the 3d color array of the overlays
        This is the sum of all the types in trial_frames_by_type, colorized
        by rewarded side.
    frame_dir : where the frames are
    trial_matrix : the trial matrix
    
    In addition to saving them to disk, this also returns:
        trial_frames_by_type (DataFrame), trial_frames_all_types (array)
    
    """
    # Make the various overlays
    # Reload
    trialnum2frame = BeWatch.overlays.load_frames_by_trial(
        frame_dir, trial_matrix)

    # Keep only those trials that we found images for
    trial_matrix = trial_matrix.ix[sorted(trialnum2frame.keys())]

    # Generate the trial_frames_by_type data frame
    # Split on side, servo_pos, stim_number
    res = []
    gobj = trial_matrix.groupby(['rewside', 'servo_pos', 'stepper_pos'])
    for (rewside, servo_pos, stim_number), subti in gobj:
        meaned = np.mean([trialnum2frame[trialnum] for trialnum in subti.index],
            axis=0)
        res.append({'rewside': rewside, 'servo_pos': servo_pos, 
            'stim_number': stim_number, 'meaned': meaned})
    resdf = pandas.DataFrame.from_records(res)

    # Save trial_frames_by_type
    whiskvid.db.TrialFramesByType.save(trial_frames_by_type_filename, resdf)

    # Make figure window
    if ax is None:
        f, ax = plt.subplots(figsize=(6.4, 6.2))

    # Make the trial_frames_all_types and save it
    C = BeWatch.overlays.make_overlay(resdf, ax, meth='all')
    whiskvid.db.TrialFramesAllTypes.save(overlay_image_name, C)
    
    return resdf, C
## End overlays


## edge_summary + tac
def get_tac(session, min_t=None, max_t=None):
    """Wrapper to load tac, add behavioral information and trial times
    
    min_t, max_t : if not None, then drops rows that are not in this time
        range relative to the choice time on each trial
    """
    db = whiskvid.db.load_db()

    # Load stuff
    res = whiskvid.db.load_everything_from_session(session, db)
    tac = res['tac']
    trial_matrix = res['trial_matrix']
    v2b_fit = res['v2b_fit']

    # Get trial timings
    bfile = db.loc[session, 'bfile']
    trial_matrix['choice_time'] = BeWatch.misc.get_choice_times(bfile)

    # Add trials
    tac = whiskvid.db.add_trials_to_tac(tac, v2b_fit, trial_matrix, 
        drop_late_contacts=False)
    
    if min_t is not None:
        tac = tac[tac.t_wrt_choice >= min_t]
    if max_t is not None:
        tac = tac[tac.t_wrt_choice < max_t]
    
    return tac

def plot_tac(session, ax=None, versus='rewside', min_t=None, max_t=None,
    alpha=1):
    """Plot the contact locations based on which trial type or response.
    
    whiskvid.db.add_trials_to_tac is used to connect the contact times
    to the behavioral data.
    
    t_min and t_max are passed to tac
    """
    db = whiskvid.db.load_db()
    tac = get_tac(session, min_t=min_t, max_t=max_t)

    if ax is None:
        f, ax = plt.subplots()
    
    if versus == 'rewside':
        # Plot tac vs rewside
        rewside2color = {'left': 'b', 'right': 'r'}
        gobj = my.pick_rows(tac, 
            choice=['left', 'right'], outcome='hit', isrnd=True).groupby('rewside')
        for rewside, subtac in gobj:
            ax.plot(subtac['tip_x'], subtac['tip_y'], 'o',
                color=rewside2color[rewside], mec='none', alpha=alpha)
            ax.set_xlim((0, db.loc[session, 'v_width']))
            ax.set_ylim((db.loc[session, 'v_height'], 0))
        my.plot.rescue_tick(ax=ax, x=4, y=4)
    elif versus == 'choice':
        # Plot tac vs choice
        choice2color = {'left': 'b', 'right': 'r'}
        gobj = my.pick_rows(tac, 
            choice=['left', 'right'], isrnd=True).groupby('choice')
        for rewside, subtac in gobj:
            ax.plot(subtac['tip_x'], subtac['tip_y'], 'o',
                color=rewside2color[rewside], mec='none', alpha=alpha)
            ax.set_xlim((0, db.loc[session, 'v_width']))
            ax.set_ylim((db.loc[session, 'v_height'], 0))
        my.plot.rescue_tick(ax=ax, x=4, y=4)
    else:
        raise ValueError("bad versus: %s" % versus)

    plt.show()
    
    return ax

def plot_edge_summary(session, ax=None, **kwargs):
    """Plot the 2d histogram of edge locations
    
    kwargs are passed to imshow, eg clim or alpha
    The image is stretched to video width and height, regardless of
    histogram edges.
    
    Returns: typical_edges_hist2d, typical_edges_row, typical_edges_col
    """
    db = whiskvid.db.load_db()
    
    # Load overlay image and edge_a
    everything = whiskvid.db.load_everything_from_session(session, db)

    # Get behavior times
    trial_matrix = everything['trial_matrix']
    trial_matrix['choice_time'] = BeWatch.misc.get_choice_times(
        db.loc[session, 'bfile'])
    choice_btime = np.polyval(everything['b2v_fit'], trial_matrix['choice_time'])
    trial_matrix['choice_bframe'] = np.rint(choice_btime * 30)

    # Get hists
    typical_edges_hist2d = np.sum(everything['edge_summary']['H_l'], axis=0)
    typical_edges_row = everything['edge_summary']['row_edges']
    typical_edges_col = everything['edge_summary']['col_edges']

    # Plot H_l
    if ax is None:
        f, ax = plt.subplots()
    im = my.plot.imshow(typical_edges_hist2d, ax=ax,
        xd_range=(0, db.loc[session, 'v_width']),
        yd_range=(0, db.loc[session, 'v_height']),
        axis_call='image', cmap=plt.cm.gray_r, **kwargs)
    #~ f.savefig(os.path.join(row['root_dir'], session, 
        #~ session + '.edges.overlays.png'))
    
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