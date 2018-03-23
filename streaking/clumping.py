import pandas
import numpy as np
from base import calculate_center2center_distance_on_merged
import scipy.optimize

def clump_segments_into_streaks(mwe):
    ## Clump segments into streaks
    mwe = mwe.copy()
    
    # Copy into new intermediate dataframe
    geo_cols = ['center_x', 'center_y']
    mwe2 = mwe.loc[:, geo_cols + ['frame']].reset_index()
    mwe2['next_frame'] = mwe2['frame'] + 1
    
    # merge over next_frame
    merged = pandas.merge(mwe2, mwe2, left_on='next_frame', right_on='frame', 
        suffixes=('0', '1'))
    
    # find distance between each pair
    dist = calculate_center2center_distance_on_merged(merged)
    dist.name = 'cost' # for compatibility with below
    
    # For each frame, for each segment choose the segment in the next
    # frame that is closest distance
    # Can do this greedily or hungarian
    # It ends up being basically the same but hungarian takes longer
    # (mostly due to pandas operations)
    method = 'first' # or 'hungarian'
    if method == 'first':
        # Get sorted costs
        sorted_costs = pandas.concat([merged, dist], axis=1).loc[:, 
            ['frame0', 'index0', 'index1', 'cost']].sort_values('cost')
        
        # Within each frame, take the first hit in index1 for each index0
        # This ensures only one choice per index0
        assignments = sorted_costs.drop_duplicates(['frame0', 'index0'])
        
        # Index by frame and index
        assignments = assignments.set_index(['frame0', 'index0']).sort_index()
        
    elif method == 'hungarian':
        # Index by frame and each possible segment, take 'cost'
        cost_by_frame = pandas.concat([merged, dist], axis=1).set_index(
            ['frame0', 'index0', 'index1'])['cost']
        
        # For each frame, use hungarian to assign index0 to index1
        frame_res_l = []
        choices_l = []
        
        # This loop is intensive, not even the hungarian that is the hard part,
        # just the grouping and unstacking
        for frame, cost_matrix_this_frame in cost_by_frame.groupby(level=0):
            # Cost matrix for this frame, with columns corresponding to segments
            # in the next frame
            cost_matrix_this_frame = cost_matrix_this_frame.unstack()
            
            # Optimize
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                cost_matrix_this_frame)
            
            # The optimized selections as a series
            choices = pandas.Series(cost_matrix_this_frame.columns[col_ind], 
                index=cost_matrix_this_frame.index[row_ind])

            # Store choices
            choices_l.append(choices)
        
        # Concat choices
        all_choices = pandas.concat(choices_l)
        
        # Extract costs
        all_choice_costs = pandas.concat([merged, dist], axis=1)[
            ['frame0', 'index0', 'index1', 'cost']].set_index(
            ['index0', 'index1'])['cost'].loc[
            pandas.MultiIndex.from_arrays([
            all_choices.index.get_level_values(1), all_choices.values])].values

        # Store both in assignments
        assignments = pandas.concat([
            all_choices, pandas.Series(
            all_choice_costs, index=all_choices.index, name='cost')],
            axis=1)

    # Nullify assignments above criterion
    assignments = assignments.loc[assignments['cost'] < 45]

    
    ## Propagate assignments
    # This is now the heaviest part of the algorithm
    # method 1: more legible
    method = 'fast'
    if method == 'legible':
        mwe['streak'] = range(len(mwe))
        # Propagate
        for frame, subass in assignments.groupby(level=0):
            mwe.loc[subass['index1'].values, 'streak'] = (
                mwe.loc[subass.index.get_level_values(1), 'streak'].values)

        # int
        mwe['streak'] = mwe['streak'].rank(method='dense').astype(np.int)
    
    # method 2: faster
    elif method == 'fast':
        # Get streaks, frames, and indexes as arrays
        streakass = np.arange(len(mwe)).astype(np.int)
        frameidx = assignments.index.get_level_values(level='frame0')
        idx0 = assignments.index.get_level_values(level='index0')
        idx1 = assignments['index1'].values
        
        # ilocify the indexes into streakass
        ser = pandas.Series(range(len(mwe)), index=mwe.index)
        iidx0 = ser.loc[idx0]
        iidx1 = ser.loc[idx1]
        
        # propagate
        for frame in assignments.index.levels[0]:
            frame_mask = frameidx == frame
            streakass[iidx1[frame_mask]] = streakass[iidx0[frame_mask]]
        
        # densify streakass
        mwe['streak'] = streakass
        mwe['streak'] = mwe['streak'].rank(method='dense').astype(np.int)

    return mwe
