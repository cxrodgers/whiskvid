"""Module for handling edge detection and summarizing"""
from base import *
import numpy as np
import pandas
import scipy.ndimage
import matplotlib.pyplot as plt
import my
import BeWatch

def get_int_input_with_default(name, value):
    """Get an integer input with a default value"""
    if value is None:
        new_value_s = raw_input("Enter %s [default=None]:" % name)
    else:
        new_value_s = raw_input("Enter %s [default=%d]:" % (name, value))
        
    try:
        new_value = int(new_value_s)
    except (TypeError, ValueError):
        new_value = value
    
    return new_value

def get_string_input_with_default(name, value):
    """Get an integer input with a default value"""
    new_value_s = raw_input("Enter %s [default=%s]:" % (name, value))
    new_value = new_value_s.strip()
    
    if new_value == '':
        new_value = value

    return new_value

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
    
    # The params this handler helps set
    _manual_param_fields = (
        'param_face_side', 
        'param_edge_x0', 'param_edge_x1', 'param_edge_y0', 'param_edge_y1', 
        'param_edge_lumthresh', 'param_edge_split_iters',
        'param_edge_crop_x0', 'param_edge_crop_x1', 
        'param_edge_crop_y0', 'param_edge_crop_y1', 
    )
    
    # The fields that are required before calculate can run
    _required_fields_for_calculate = (
        'monitor_video',
    )

    # Override because it's not just _name
    @property
    def new_path(self):
        return 'all_edges.npy'
    
    def load_data(self):
        """Override load_data because edges_all uses numpy.load"""
        filename = self.get_path
        try:
            data = np.load(filename)
        except IOError:
            raise IOError("no all_edges found at %s" % filename)
        return data   
    
    def save_data(self, edge_a):
        """Save data to disk and set the database path.
        
        This override uses numpy.save. See the base class for doc.
        """
        filename = self.new_path_full
        
        # Save
        try:
            np.save(filename, edge_a)
        except IOError:
            raise IOError("cannot numpy.save to %s" % filename)
        
        # Set path
        self.set_path()
    
        # This should now work
        return self.get_path
    
    def choose_manual_params(self, force=False, crop_start=100.,
        crop_stop=7000., crop_n_frames=25):
        """Interactively get the necessary manual params
        
        This is a two-stage process. First we simply plot a subset of the
        frames and request the edge ROI, lumthresh, and face side.
        
        Secondly we extract edges from a subset of frames (using the
        params from the first stage) and request the crop params. The user
        may also want to change the params from the first part based on this
        result.
        """
        # Return if force=False and all params already set
        if not force and self._check_if_manual_params_set():
            return
        
        # Shortcut
        vs_obj = self.video_session._django_object
        
        # Get the edge roi, lumthresh, and face side
        monitor_video_filename = self.video_session.data.monitor_video.get_path
        manual_params = choose_manual_params_nodb(
            monitor_video_filename, interactive=True,
            side=vs_obj.get_param_face_side_display(),
            edge_x0=vs_obj.param_edge_x0, edge_x1=vs_obj.param_edge_x1, 
            edge_y0=vs_obj.param_edge_y0, edge_y1=vs_obj.param_edge_y1, 
            lumthresh=vs_obj.param_edge_lumthresh,         
        )
        
        # Set them in the database
        side2int = dict([(v, k) for k, v in vs_obj.param_face_choices])
        try:
            vs_obj.param_face_side = side2int[manual_params['side']]
        except KeyError:
            print "warning: invalid face side, leaving unchanged"
        vs_obj.param_edge_lumthresh = manual_params['edge_lumthresh']
        vs_obj.param_edge_x0 = manual_params['edge_roi_x0']        
        vs_obj.param_edge_x1 = manual_params['edge_roi_x1']        
        vs_obj.param_edge_y0 = manual_params['edge_roi_y0']        
        vs_obj.param_edge_y1 = manual_params['edge_roi_y1']        
        vs_obj.save()
        
        ## Second stage
        # Get the crop manual params
        debug_frametimes = np.linspace(crop_start, crop_stop, crop_n_frames)
        crop_params = choose_crop_params_nodb(
            video_file=monitor_video_filename,
            frametimes=debug_frametimes, 
            side=vs_obj.get_param_face_side_display(),
            edge_x0=vs_obj.param_edge_x0, edge_x1=vs_obj.param_edge_x1, 
            edge_y0=vs_obj.param_edge_y0, edge_y1=vs_obj.param_edge_y1, 
            lumthresh=vs_obj.param_edge_lumthresh, 
            split_iters=vs_obj.param_edge_split_iters,
            crop_params_init=None,
        )
        
        # Set the crop manual params
        vs_obj.param_edge_crop_x0 = crop_params[0]
        vs_obj.param_edge_crop_x1 = crop_params[1]
        vs_obj.param_edge_crop_y0 = crop_params[2]
        vs_obj.param_edge_crop_y1 = crop_params[3]
        vs_obj.save()

    def calculate(self, force=False):
        """Calculate edges using calculate_all_edges_nodb
        
        See calculate_all_edges_nodb for algorithm. This loads
        data from disk and stores result.
        
        force : if False and the data exists on disk, returns immediately
            If the data does exist (checked by calling get_path and
            seeing if exception results) then this function returns
            immediately.
        
        Returns : nothing
            We avoid the overhead of loading from disk unless load_data
            is specifically called
        """
        # Return if force=False and the data exists
        if not force:
            # Check if data available
            data_available = True
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
        
        ## Begin handler-specific stuff
        # Shortcut
        vs_obj = self.video_session._django_object
        
        video_file = self.video_session.data.monitor_video.get_path
        
        # Calculate
        edge_a = get_all_edges_from_video(video_file,
            return_frames_instead=False,
            n_frames=np.inf,
            lum_threshold=vs_obj.param_edge_lumthresh, 
            roi_x=(vs_obj.param_edge_x0, vs_obj.param_edge_x1),
            roi_y=(vs_obj.param_edge_y0, vs_obj.param_edge_y1),
            split_iters=vs_obj.param_edge_split_iters,
            side=vs_obj.get_param_face_side_display(),
            crop_x0=vs_obj.param_edge_crop_x0,
            crop_x1=vs_obj.param_edge_crop_x1,
            crop_y0=vs_obj.param_edge_crop_y0,
            crop_y1=vs_obj.param_edge_crop_y1,
        )
        ## End handler-specific stuff
        
        # Store
        self.save_data(edge_a)

class EdgeSummaryHandler(CalculationHandler):
    _db_field_path = 'edge_summary_filename'
    _name = 'edge_summary'

    # The fields that are required before calculate can run
    _required_fields_for_calculate = (
        'all_edges_filename',
    )

    def load_data(self):
        """Override load_data to use my.misc.pickle_load"""
        filename = self.get_path
        try:
            data = my.misc.pickle_load(filename)
        except IOError:
            raise IOError("no edge_summary found at %s" % filename)
        return data   
    
    def save_data(self, edge_summary):
        """Save data to disk and set the database path.
        
        This override uses my.misc.pickle_dump. See the base class for doc.
        """
        filename = self.new_path_full
        
        # Save
        try:
            my.misc.pickle_dump(edge_summary, filename)
        except IOError:
            raise IOError("cannot my.misc.pickle_dump to %s" % filename)
        
        # Set path
        self.set_path()
    
        # This should now work
        return self.get_path

    def calculate(self, force=False, **kwargs):
        """Summarize edges
        
        See calculate_edge_summary_nodb for doc
        
        force : if False and the data exists on disk, returns immediately
            If the data does exist (checked by calling get_path and
            seeing if exception results) then this function returns
            immediately.
        
        Returns : nothing
            We avoid the overhead of loading from disk unless load_data
            is specifically called
        """
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
        
        ## Begin handler-specific stuff
        # Get trial matrix
        trial_matrix = BeWatch.db.get_trial_matrix(
            self.video_session.bsession_name, True)

        # Get edges
        edge_a = self.video_session.data.all_edges.load_data()
        
        # Get fit
        b2v_fit = self.video_session.fit_b2v
        
        # Get aspect
        v_width = self.video_session.frame_width
        v_height = self.video_session.frame_height
        
        # Calculate edge summary
        edge_summary = calculate_edge_summary_nodb(
            trial_matrix, edge_a, b2v_fit, 
            v_width, v_height, **kwargs)
        
        ## End handler-specific stuff
        # Store
        self.save_data(edge_a)


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

def choose_manual_params_nodb(video_file, interactive=True,
    side=None, edge_x0=None, edge_x1=None, edge_y0=None, edge_y1=None,
    lumthresh=None):
    """Interactively set the parameters for edging.

    This ROI will be used to identify which of the dark shapes in the object
    is the stimulus. Typically, we choose the largest shape that has any
    part of itself in the ROI. Thus, choose the ROI such that the face is never
    included, but some part of the shape is always included.
    
    Requires monitor video to exist.
    
    Takes the first 10000 frames of the video. Sorts the frames by those
    that have minimal intensity in the upper right corner. Plots a subset
    of those. (This all assumes dark shapes coming in from the upper right.)
    
    This heatmap view allows the user to visualize the typical luminance of
    the shape, to set lumthresh.
    
    Then calls choose_rectangular_ROI so that the user can interactively set
    the ROI that always includes some part of the shape and never includes
    the face.
    
    Finally the user inputs the face side.
    
    Params:
        video_file : filename of monitor video
        interactive : passed to my.video.choose_rectangular_ROI
        all other kwargs : the current or default values of the params,
            or None. These are simply displayed as a hint for the user
            while respecifying.
    
    Returns: dict with keys edge_roi_*, edge_lumthresh, side
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
        hints={'x0': edge_x0, 'x1': edge_x1, 'y0': edge_y0, 'y1': edge_y1})
    
    # Rename the keys
    res2 = {}
    for key in res:
        res2['edge_roi_' + key] = res[key]

    # Get the lum_threshold
    res2['edge_lumthresh'] = get_int_input_with_default('lumthresh', lumthresh)

    # Get the face side
    res2['side'] = get_string_input_with_default('face side', side)

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

def choose_crop_params_nodb(video_file, frametimes, 
    side, edge_x0, edge_x1, edge_y0, edge_y1, lumthresh, split_iters,
    crop_params_init=None,
    ):
    """Manual param selection for edge cropping.
    
    For some subset of frames, plot the raw frame, the detected edge,
    the thresholded frame and the detected objects. The user inputs
    cropping parameters to crop the detected edges, so that they can't
    be joined onto the lick pipes for instance.
    
    Uses get_all_edges_from_video to get the intermediate results,
    and then plots them using plot_effect_of_crop_params.
    
    video_file : monitor video filename
    frametimes : which frames to analyze as a test
    side, edge_x*, edge_y*, lumthresh, split_iters : These parameters
        are sent to get_all_edges_from_video. These should be the ones
        that are actually going to be used, but were set in a different
        stage.
    crop_params_init : None, or a 4-tuple (x0, x1, y0, y1)
        This is the initial guess for the cropping params. If None, uses
        the edge_* params. This is sent to get_all_edges_from_video and
        the cropping is applied.
    
    Returns: tuple
        crop_x0, crop_x1, crop_y0, crop_y1
    """
    import my.plot 
    
    if len(frametimes) > 64:
        raise ValueError("too many frametimes")
    
    # Initialize the crop params with the edge params
    if crop_params_init is None:
        crop_x0, crop_x1, crop_y0, crop_y1 = (
            edge_x0, edge_x1, edge_y0, edge_y1)
    else:
        crop_x0, crop_x1, crop_y0, crop_y1 = crop_params_init
    
    # Iteratively select the crop params
    while True:
        # Gets the edges from subset of debug frames using provided parameters
        debug_res = get_all_edges_from_video(video_file, 
            crop_x0=crop_x0, crop_x1=crop_x1, crop_y0=crop_y0, crop_y1=crop_y1,
            roi_x=(edge_x0, edge_x1), roi_y=(edge_y0, edge_y1),
            split_iters=split_iters, side=side,
            lum_threshold=lumthresh,
            debug=True, debug_frametimes=frametimes)    
        
        # Plot result
        plot_effect_of_crop_params(frametimes, debug_res)
        
        confirm_input = raw_input("Confirm? [y/N]:")
        if confirm_input.lower().strip() == 'y':
            break
        
        # Get input
        crop_x0 = get_int_input_with_default('crop_x0', crop_x0)
        crop_x1 = get_int_input_with_default('crop_x1', crop_x1)
        crop_y0 = get_int_input_with_default('crop_y0', crop_y0)
        crop_y1 = get_int_input_with_default('crop_y1', crop_y1)
    
    return crop_x0, crop_x1, crop_y0, crop_y1

def plot_effect_of_crop_params(frametimes, debug_res):
    """Plots the detected edges to help choose crop params
    
    Makes two figure window, each with len(frametimes) subplots.
    The first one shows the frame with the identified edge overlaid.
    The second one just shows the binarized frame.
    
    frametimes : the time of each frame that was analyzed
    debug_res : result of whiskvid.get_all_edges_from_video
    """
    # Plot them
    f, axa = my.plot.auto_subplot(len(frametimes), return_fig=True, 
        figsize=(12, 12))
    f2, axa2 = my.plot.auto_subplot(len(frametimes), return_fig=True, 
        figsize=(12, 12))
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
    
def calculate_edge_summary_nodb(trial_matrix, edge_a, b2v_fit, v_width, v_height,
    hist_pix_w=2, hist_pix_h=2, vid_fps=30, offset=-.5):
    """Extract edges at choice times for each trial type
    
    2d-histograms at choice times and saves the resulting histogram
    
    trial_matrix : must have choice time added in already
    edge_a : array of edge at every frame
    offset : time relative to choice time at which frame is dumped
    
    Check if there is a bug here when the edge is in the last row and is
    not in the histogram.
    
    Returns: {
        'row_edges': row_edges, 'col_edges': col_edges, 
        'H_l': H_l, 'rewside_l': rwsd_l, 'srvpos_l': srvpos_l}    
    """
    # Convert choice time to frames using b2v_fit
    choice_btime = np.polyval(b2v_fit, trial_matrix['choice_time'])
    choice_btime = choice_btime + offset
    
    # Convert frame to int, though it will be float if it contains NaN
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
        for frame in subtm['choice_bframe'].dropna().astype(np.int).values:
            # Skip ones outside the video
            if frame < 0 or frame >= len(edge_a):
                continue
            
            # Count the ones for which no edge was detected
            elif edge_a[frame] is None:
                n_bad_edges = n_bad_edges + 1
                continue
            
            else:
                sub_edge_a.append(edge_a[frame])

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
    return res
## End edge summary dumping
