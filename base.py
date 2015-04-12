"""Functions for analyzing whiski data"""
import traj, trace
import numpy as np, pandas
import os
import scipy.ndimage
import my
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
    
    Currently, for each row, we take the left most nonzero value. We
    return (row, col) for each such pixel. However, this doesn't work for
    horizontal parts of the edge.
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
    meth='largest_in_roi', split_iters=10):
    """Find the left edge of the shape in frame.
    
    This is a wrapper around the other utility functions
    
    1. Thresholds image to find dark spots
    2. Segments the dark spots using scipy.ndimage.label
    3. Chooses the largest dark spot within the ROI
    4. Finds the left edge of this spot
    
    meth: largest_with_centroid_in_roi, largest_in_roi
    
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
        return None
    best_id = np.where(mask_is_in_roi)[0][np.argmax(szs[mask_is_in_roi])]

    # Error if no object found above sz 10000 (100x100)
    if szs[best_id] < 10000:
        #raise ValueError("all objects in the ROI are too small")
        return None

    # Get the contour of the object
    best_object = objects == best_id
    edge = edge_getter(best_object)
    
    return edge

def get_all_edges_from_video(video_file, n_frames=np.inf, verbose=True,
    lum_threshold=50, roi_x=(200, 500), roi_y=(0, 400),
    return_frames_instead=False, meth='largest_in_roi', split_iters=10):
    """Top-level function for extracting edges from video
    
    Uses process_chunks_of_video and find_edge_of_shape
    Typically you want to test on a small number of frames to make sure
    it's working.
    
    return_frames_instead : for debugging. If True, return the raw frames
        instead of the edges
    
    Returns: edge_a
    """
    
    # Helper function to pass to process_chunks_of_video
    def mapfunc(frame):
        """Gets the edge from each frame"""
        edge = find_edge_of_shape(frame, lum_threshold=50,
            roi_x=roi_x, roi_y=roi_y, edge_getter=get_bottom_edge,
            meth=meth, split_iters=split_iters)
        if edge is None:
            return None
        else:
            return edge.astype(np.int16)

    if return_frames_instead:
        mapfunc = 'keep'

    # Get the edges
    edge_a = my.misc.process_chunks_of_video(video_file, 
        n_frames=n_frames,
        func=mapfunc,
        verbose=verbose,
        finalize='listcomp')

    return edge_a

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

def put_whiskers_into_hdf5(whisk_filename, h5_filename, verbose=True,
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
        output_file = os.path.join(row['session_dir'], session + '.whiskers')
        db.loc[session, 'whiskers'] = output_file
    
    trace_session_nodb(row['vfile'], db.loc[session, 'whiskers'])
    
    # Save
    whiskvid.db.save_db(db)  

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
    1/0

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
    tac = calculate_contacts(row['wseg_h5'], row['edge'], row['side'], 
        tac_filename=row['tac'], **kwargs)

def calculate_contacts(h5_filename, edge_file, side, tac_filename=None,
    length_thresh=75, contact_dist_thresh=10,
    fol_range_x=(0, 70), fol_range_y=(250, 360),
    verbose=True):
    # Get the ends
    resdf = get_whisker_ends_hdf5(h5_filename, side=side)

    # Drop everything < thresh
    resdf = resdf[resdf['length'] >= length_thresh]

    # Follicle mask
    resdf = resdf[
        (resdf['fol_x'] > fol_range_x[0]) & (resdf['fol_x'] < fol_range_x[1]) &
        (resdf['fol_y'] > fol_range_y[0]) & (resdf['fol_y'] < fol_range_y[1])]

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

    # Join
    tips_and_contacts = resdf.join(contacts_df.set_index('index'))
    tips_and_contacts = tips_and_contacts[
        tips_and_contacts.closest_dist < contact_dist_thresh]
    if tac_filename is not None:
        tips_and_contacts.to_pickle(tac_filename)
    return tips_and_contacts

## End calculating contacts

