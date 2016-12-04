"""Module for handling edge detection and summarizing"""
from base import *
import numpy as np
import pandas
import scipy.ndimage
import matplotlib.pyplot as plt
import my


class AllEdgesHandler(CalculationHandler):
    """Handler for all_edges. 
    
    On every frame, we calculate and store the edge of the shape.
    
    load_data : Data is stored and loaded using numpy.load
    choose_manual_params : 
        The following manual params are required:
            'edge_roi_x0', 
            'edge_lumthresh', 'side'
            'crop_x0', ...
            'split_iters'
    calculate :
        Start the calculation. Requires manual params to be set first.
    """
    _db_field_path = 'all_edges_filename'
    _name = 'all_edges'
    _required_manual_param_fields = (
        'param_face_side', 
        'param_edge_x0', 'param_edge_x1', 'param_edge_y0', 'param_edge_y1', 
        'param_edge_lumthresh', 'param_edge_split_iters',
        'param_edge_crop_x0', 'param_edge_crop_x1', 
        'param_edge_crop_y0', 'param_edge_crop_y1', 
    )
    
    def load_data(self):
        filename = self.get_path
        try:
            data = np.load(filename)
        except IOError:
            raise IOError("no all_edges found at %s" % filename)
        return data

    def _check_if_manual_params_set(self):
        """Returns True if all required manual params are set in the db"""
        all_params_set = True
        for attr in self._required_manual_param_fields:
            if pandas.isnull(getattr(self.video_session._django_object, attr)): 
                all_params_set = False
        return all_params_set
        
    def choose_manual_params(self, force=False):
        """Interactively get the necessary manual params"""
        # Return if force=False and all params already set
        if not force and self._check_if_manual_params_set():
            return
        
        
        

    def calculate(self, force=False, save=True):
        """Calculate edges using calculate_all_edges_nodb
        
        See calculate_all_edges_nodb for algorithm. This loads
        data from disk and stores result.
        
        Returns : all_edges
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


def calculate_all_edges_nodb():
    ## edge them
    print "edge"
    db = whiskvid.db.load_db()
    for session in subdb.index:
        row = db.ix[session]
        if pandas.isnull(row['edge']) or not os.path.exists(row['edge']):
            print session
            
            # Determine whether we need to set manual params
            run_manual_params = False
            trigger_params = ['edge_roi_x0', 'edge_lumthresh', 'side']
            for trigger_param in trigger_params:
                if trigger_param not in db or pandas.isnull(row[trigger_param]):
                    run_manual_params = True
        
            # Get params if none exist
            if run_manual_params:
                whiskvid.edge_frames_manual_params_db(session)

            # Debugging
            # Can set these parameters manually to see what works
            # Otherwise will be taken from database
            #~ debug_frametimes = np.linspace(100., 7000., 16)
            #~ debug_res = whiskvid.edge_frames_debug_plot(session, 
                #~ frametimes=debug_frametimes,
                #~ split_iters=5,
                #~ crop_x0=0, crop_x1=525, crop_y0=200, crop_y1=600,
                #~ roi_x=None, roi_y=None,
                #~ lumthresh=None,
                #~ )
            #~ 1/0
            
            # Run the edging
            whiskvid.edge_frames(session, verbose=True, 
                split_iters=5,
                crop_x0=0, crop_x1=525, crop_y0=200, crop_y1=600,
                )


class EdgeSummaryHandler(CalculationHandler):
    _db_field_path = 'edge_summary_filename'
    _name = 'edge_summary'




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

def get_top_edge(object_mask):
    """Return the top edge of the object.
    
    Currently, for each column, we take the top most nonzero value. We
    return (row, col) for each such pixel. However, this doesn't work for
    vertical parts of the edge.
    """
    contour = []
    for ncol, col in enumerate(object_mask.T):
        true_rows = np.where(col)[0]
        if len(true_rows) == 0:
            continue
        else:
            contour.append((true_rows[0], ncol))
    return np.asarray(contour)


def plot_all_objects(objects, nobjects):
    f, axa = plt.subplots(1, nobjects)
    for object_id in range(nobjects):
        axa[object_id].imshow(objects == object_id)
    plt.show()

def find_edge_of_shape(frame, 
    crop_x0=None, crop_x1=None, crop_y0=None, crop_y1=None,
    lum_threshold=30, roi_x=(320, 640),
    roi_y=(0, 480), size_threshold=1000, edge_getter=get_bottom_edge,
    meth='largest_in_roi', split_iters=10, debug=False):
    """Find the left edge of the shape in frame.
    
    This is a wrapper around the other utility functions
    
    0. Crops image. The purpose of this is to remove dark spots that may
      be contiguous with the shape. For instance, dark border of frame, or
      mouse face, or pipes.
    1. Thresholds image to find dark spots
    2. Segments the dark spots using scipy.ndimage.label
    3. Chooses the largest dark spot within the ROI
    4. Finds the left edge of this spot
    
    crop_x0 : ignore all pixels to the left of this value
    crop_x1 : ignore all pixels to the right of this value
    crop_y0 : ignore all pixels above (less than) this value
    crop_y1 : ignore all pixels below (greater than) this value
    
    meth: largest_with_centroid_in_roi, largest_in_roi
    
    If debug: returns binframe, best_object, edge, status
        where status is 'none in ROI', 'all too small', or 'good'
        unless status is 'good', best_object and edge are None
    
    Returns: bottom edge, as sequence of (y, x) (or row, col) pairs
        If no acceptable object is found, returns None.
    """
    # Segment image
    binframe = frame < lum_threshold
    
    # Force the area that is outside the crop to be 0 (ignored)
    if crop_x0 is not None:
        binframe[:, :crop_x0] = 0
    if crop_x1 is not None:
        binframe[:, crop_x1:] = 0
    if crop_y0 is not None:
        binframe[:crop_y0, :] = 0
    if crop_y1 is not None:
        binframe[crop_y1:, :] = 0
    
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
            return binframe, None, None, 'none in ROI'
        else:
            return None
    best_id = np.where(mask_is_in_roi)[0][np.argmax(szs[mask_is_in_roi])]

    # Error if no object found above sz 10000 (100x100)
    if szs[best_id] < size_threshold:
        #raise ValueError("all objects in the ROI are too small")
        if debug:
            return binframe, None, None, 'all too small'
        else:
            return None

    # Get the contour of the object
    best_object = objects == best_id
    edge = edge_getter(best_object)

    if debug:
        return binframe, best_object, edge, 'good'
    else:
        return edge

def get_all_edges_from_video(video_file, n_frames=np.inf, verbose=True,
    crop_x0=None, crop_x1=None, crop_y0=None, crop_y1=None,
    lum_threshold=50, roi_x=(200, 500), roi_y=(0, 400),
    return_frames_instead=False, meth='largest_in_roi', split_iters=10,
    side='left', debug=False, debug_frametimes=None):
    """Function that captures video frames and calls find_edge_of_shape.
    
    The normal function is to call process_chunks_of_video on the whole
    video. Alternatively, the raw frames can be returned (to allow the user
    to set parameters). This is also better because the same exact frames
    are returned as would have been processed.
    
    Or, it can be run in debug mode, which returns
    all intermediate computations (and uses get_frame instead of
    process_chunks_of_video). Not clear that this method returns exactly
    the same frames.
    
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
    elif side == 'right':
        edge_getter = get_top_edge
    else:
        raise ValueError("side must be left or top, instead of %r" % side)
    
    if not debug:
        # Helper function to pass to process_chunks_of_video
        def mapfunc(frame):
            """Gets the edge from each frame"""
            edge = find_edge_of_shape(frame, 
                crop_x0=crop_x0, crop_x1=crop_x1, 
                crop_y0=crop_y0, crop_y1=crop_y1,
                lum_threshold=lum_threshold,
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
        res = {'frames': [], 'binframes': [], 'best_objects': [], 'edges': [],
            'statuses': []}
        
        # Iterate over frames
        for frametime in debug_frametimes:
            # Get the frame
            frame, stdout, stderr = my.video.get_frame(video_file, frametime)
            
            # Compute the edge and intermediate results
            binframe, best_object, edge, status = find_edge_of_shape(
                frame, lum_threshold=lum_threshold,
                crop_x0=crop_x0, crop_x1=crop_x1, 
                crop_y0=crop_y0, crop_y1=crop_y1,                
                roi_x=roi_x, roi_y=roi_y, edge_getter=edge_getter,
                meth=meth, split_iters=split_iters, debug=True)
            
            if edge is not None:
                edge = edge.astype(np.int16)
            
            # Store and return
            res['frames'].append(frame)
            res['binframes'].append(binframe)
            res['best_objects'].append(best_object)
            res['edges'].append(edge)
            res['statuses'].append(status)
        
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
    """Interactively set the parameters for edging.
    
    Takes the first 10000 frames of the video. Sorts the frames by those
    that have minimal intensity in the upper right corner. Plots a subset
    of those. (This all assumes dark shapes coming in from the upper right.)
    
    This heatmap view allows the user to visualize the typical luminance of
    the shape, to set lumthresh.
    
    Then calls choose_rectangular_ROI so that the user can interactively set
    the ROI that always includes some part of the shape and never includes
    the face.
    
    Finally the user inputs the face side.
    """
    width, height = my.video.get_video_aspect(video_file)
    
    # Try to find a frame with a good example of a shape
    def keep_roi(frame):
        height, width = frame.shape
        return frame[:int(0.5 * height), int(0.5 * width):]
    frames_a = my.video.process_chunks_of_video(video_file, n_frames=3000,
        func='keep', frame_chunk_sz=1000, verbose=True, finalize='listcomp')
    idxs = np.argsort([keep_roi(frame).min() for frame in frames_a])

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
    else:
        print "already edged, returning"
        return
    
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
            row['vfile'], output_file, side=side, debug=True,
            **kwargs)
        
        return frames, edge_a
    else:
        edge_frames_nodb(
            row['vfile'], output_file, side=side, debug=False,
            **kwargs)
    
    # Update the db
    db = whiskvid.db.load_db()
    db.loc[session, 'edge'] = output_file
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
    
def edge_frames_debug_plot(session, frametimes, split_iters=7,
    crop_x0=None, crop_x1=None, crop_y0=None, crop_y1=None,
    roi_x=None, roi_y=None, lumthresh=None, side=None,
    ):
    """This is a helper function for debugging edging.
    
    For some subset of frames, plot the raw frame, the detected edge,
    the thresholded frame and (TODO) the detected objects.
    
    Uses whiskvid.get_all_edges_from_video to get the intermediate results,
    and then plots them. Also returns all intermediate results.
    
    frametimes : which frames to analyze as a test
    
    roi_x, roi_y, lumthresh, side : can be provided, or else will be taken
        from db
    """
    import my.plot 
    
    if len(frametimes) > 64:
        raise ValueError("too many frametimes")
    
    # Get raw frames, binframes, edges, on a subset
    db = whiskvid.db.load_db()
    v_width, v_height = db.loc[session, 'v_width'], db.loc[session, 'v_height']
    video_file = db.loc[session, 'vfile']
    
    # Get params from db if necessary
    if side is None:
        side = db.loc[session, 'side']
    if roi_x is None:
        roi_x = (db.loc[session, 'edge_roi_x0'], db.loc[session, 'edge_roi_x1'])
    if roi_y is None:
        roi_y = (db.loc[session, 'edge_roi_y0'], db.loc[session, 'edge_roi_y1'])
    if lumthresh is None:
        lumthresh = db.loc[session, 'edge_lumthresh']
    
    # Gets the edges from subset of debug frames using provided parameters
    debug_res = whiskvid.get_all_edges_from_video(video_file, 
        crop_x0=crop_x0, crop_x1=crop_x1, crop_y0=crop_y0, crop_y1=crop_y1,
        roi_x=roi_x, roi_y=roi_y, split_iters=split_iters, side=side,
        lum_threshold=lumthresh,
        debug=True, debug_frametimes=frametimes)    
    
    # Plot them
    f, axa = my.plot.auto_subplot(len(frametimes), return_fig=True, figsize=(12, 12))
    f2, axa2 = my.plot.auto_subplot(len(frametimes), return_fig=True, figsize=(12, 12))
    nax = 0
    for nax, ax in enumerate(axa.flatten()):
        # Get results for this frame
        try:
            frame = debug_res['frames'][nax]
        except IndexError:
            break
        binframe = debug_res['binframes'][nax]
        best_object = debug_res['best_objects'][nax]
        edge = debug_res['edges'][nax]
        frametime = frametimes[nax]
        
        # Plot the frame
        im = my.plot.imshow(frame, ax=ax, axis_call='image', 
            cmap=plt.cm.gray)#, extent=(0, v_width, v_height, 0))
        im.set_clim((0, 255))

        # Plot the edge
        if edge is not None:
            ax.plot(edge[:, 1], edge[:, 0], 'g-', lw=5)
        
        # Plot the binframe
        ax2 = axa2.flatten()[nax]
        im2 = my.plot.imshow(binframe, ax=ax2, axis_call='image',
            cmap=plt.cm.gray)#, extent=(0, v_width, v_height, 0))
        
        # Plot the best object
        ax.set_title("t=%0.1f %s" % (
            frametime, 'NO EDGE' if edge is None else ''), size='small')
    f.suptitle('Frames')
    f2.suptitle('Binarized frames')
    my.plot.rescue_tick(f=f)
    my.plot.rescue_tick(f=f2)
    f.tight_layout()
    f2.tight_layout()
    plt.show()    
    return debug_res

## End of functions for extracting objects from video


