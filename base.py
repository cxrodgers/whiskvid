"""Functions for analyzing whiski data"""
import traj, trace
import numpy as np, pandas

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