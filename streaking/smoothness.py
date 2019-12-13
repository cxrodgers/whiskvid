from __future__ import absolute_import
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
import pandas
from . import base


def measure_smoothness_costs(mwe, next_frame_streaks, alignments, next_frame,
    loss_slope = .119, loss_icpt = -5.79, disappearance_cost=100,
    key='object'):
    """Measure smoothness costs, based only on center motion
    
    The "smoothness" is how close each streak is to its matched object
    in the previous frame, as assessed by the distance between the centers.
    If the matched object doesn't exist in the previous frame (i.e., it
    just appeared), then the smoothness has a fixed penalty
    `disappearance_cost`.
    
    Based on preliminary modeling, the smoothness cost is a log loss
    with probability 0.5 around 50px.
    
    The smoothness is calculated for all possible alignments and returned.
    
    loss_slope, loss_icpt : parameters for converting a distance between
        centers, to a probability
    disappreance_cost : penalty for a disappearing object (really an
        appearing object), expressed in equivalent distance units. So 50
        corresponds to about p=0.5.
    
    Returns: smoothness_costs, smoothness_costs_by_alignment, all_smoothness_dists
        smoothness_costs : Series, total cost of each alignment
        smoothness_costs_by_alignment : DataFrame, with each assignment
            on the columns. The sum within each row is `smoothness_costs`
        all_smoothness_costs : the distance between all pairs of streaks
            and objects.
    """
    # For this we only need center
    geometry_model_columns = ['center_x', 'center_y']
    
    # Data for all streaks in next_frame (even ones that are already assigned)
    streak_data = mwe.loc[
        (mwe['frame'] == next_frame)
    ]
    
    # Data for each object, in previous frame only
    # Or could take the most recent example of each object in the past N frames
    object_data = mwe.loc[
        mwe['frame'] == (next_frame - 1)
    ]
    assert not np.any(object_data[key].isnull())
    
    # Note the objects in previous frame
    # We will penalize disappearance of these objects or appearance of new ones
    objects_in_previous_frame = object_data[key].astype(np.int).values
    
    # Take a subset of columns and copy
    streak_data = streak_data[geometry_model_columns + 
        ['frame', 'streak']].copy()
    object_data = object_data[geometry_model_columns + 
        [key, 'frame']].copy()
    
    # Outer-join on key column
    streak_data['key'] = 1
    object_data['key'] = 1
    merged = object_data.merge(streak_data, on='key', 
        suffixes=('0', '1')).drop('key', 1)
    
    # Calculate the distance for each streak*object pair
    merged['dist'] = base.calculate_center2center_distance_on_merged(merged)
    
    # Index by object and streak
    merged[key] = merged[key].astype(np.int)
    merged = merged.set_index([key, 'streak']).sort_index()
    
    
    
    ## Calculated total cost for each alignment
    method = 'fast' # 'old'
    if method == 'fast':
        # Get all idxs
        all_idxs = []
        n_alignment_l = []
        n_assignment_l = []
        for n_alignment, alignment in enumerate(alignments):
            for n_assignment, (obj, streak) in enumerate(alignment):
                all_idxs.append((obj, streak))
                n_alignment_l.append(n_alignment)
                n_assignment_l.append(n_assignment)

        # One lookup
        if len(merged) > 0:
            # Special case this one
            vals = merged.loc[all_idxs, 'dist']
        else:
            vals = pandas.Series([np.nan] * len(all_idxs))

        # If not found, then object was not in the previous frame, penalize
        vals = vals.fillna(disappearance_cost)
        
        # DataFrame it
        smoothness_dists_by_alignment = pandas.DataFrame.from_dict({
            'dist': vals.values,
            'alignment': n_alignment_l,
            'assignment': n_assignment_l,
        }).set_index(['alignment', 'assignment'])['dist'].unstack()
    
    elif method == 'old':
        smoothness_costs_by_alignment_l = []
        for alignment in alignments:
            smoothness_costs_this_alignment = []
            for obj, streak in alignment:
                if obj not in objects_in_previous_frame:
                    # It just appeared, penalize
                    cost = disappearance_cost
                else:
                    cost = np.abs(merged.loc[(obj, streak), 'dist'])
                smoothness_costs_this_alignment.append(cost)
            smoothness_costs_by_alignment_l.append(smoothness_costs_this_alignment)
        
        # DataFrame
        smoothness_dists_by_alignment = pandas.DataFrame(
            smoothness_costs_by_alignment_l, index=list(range(len(alignments)))
        )

    # Convert distance to log probability
    smoothness_costs_by_alignment = np.log10(1 - (old_div(1, 
        (1 + np.exp(-(loss_slope * smoothness_dists_by_alignment + loss_icpt))))))

    # Sum over each alignment
    smoothness_costs = smoothness_costs_by_alignment.sum(axis=1)
    
    # For debugging
    all_smoothness_dists = merged['dist'].unstack()
    
    return smoothness_costs, smoothness_costs_by_alignment, all_smoothness_dists
