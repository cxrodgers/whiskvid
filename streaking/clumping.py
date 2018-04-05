import pandas
import numpy as np
from base import calculate_center2center_distance_on_merged
import scipy.optimize

def clump_segments_into_streaks(mwe, threshold=32.0):
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
        
        # Within each frame, there should be only one index0 and one index1
        # The ordering matters here, that's just a downside to this greedy
        # algorithm versus the hungarian
        assignments = sorted_costs.drop_duplicates(
            ['frame0', 'index0']).drop_duplicates(
            ['frame0', 'index1'])
        
        # Index by frame and index
        assignments = assignments.set_index(['frame0', 'index0']).sort_index()
        
        # Error check: it should be a chain of assignments
        assert not assignments['index1'].duplicated().any()
        assert not assignments.index.duplicated().any()

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
    assignments = assignments.loc[assignments['cost'] < threshold]

    
    ## Propagate assignments
    # This is now the heaviest part of the algorithm
    # method 1: more legible
    method = 'faster'
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
        frameidx = assignments.index.get_level_values(level='frame0').values
        idx0 = assignments.index.get_level_values(level='index0')
        idx1 = assignments['index1'].values
        
        # ilocify the indexes into streakass
        ser = pandas.Series(range(len(mwe)), index=mwe.index)
        iidx0 = ser.loc[idx0].values
        iidx1 = ser.loc[idx1].values
        
        # propagate
        for frame in assignments.index.levels[0]:
            frame_mask = frameidx == frame
            streakass[iidx1[frame_mask]] = streakass[iidx0[frame_mask]]
        
        # densify streakass
        mwe['streak'] = streakass
        mwe['streak'] = mwe['streak'].rank(method='dense').astype(np.int)

    elif method == 'faster':
        # Get streaks, frames, and indexes as arrays
        streakass = np.arange(len(mwe)).astype(np.int)
        frameidx = assignments.index.get_level_values(level='frame0').values
        idx0 = assignments.index.get_level_values(level='index0')
        idx1 = assignments['index1'].values
        
        # ilocify the indexes into streakass
        ser = pandas.Series(range(len(mwe)), index=mwe.index)
        iidx0 = ser.loc[idx0].values
        iidx1 = ser.loc[idx1].values

        # This propagation series defines which ii (on index) is linked to
        # which following ii (on values)
        prop_ser = pandas.Series(iidx1, index=iidx0)

        # Include only the nodes that start a chain
        starts_only = prop_ser.drop(prop_ser.values, errors='ignore')

        # Continually map the propagation series onto itself until we've
        # reached the end of all chains
        # Begin with the start of each chain
        rec_l = [
            pandas.Series(starts_only.index.values, index=starts_only.index)]
        
        # Propagate
        res = starts_only.copy()
        while True:
            rec_l.append(res)
            
            # dropna and intify are important here for speed
            res = res.map(prop_ser).dropna().astype(np.int)
            
            # break if all chains are done
            if len(res) == 0:
                break

        # Concatenate all chains    
        chains = pandas.concat(rec_l, axis=0, keys=range(len(rec_l)), 
            verify_integrity=True).swaplevel().sort_index()
        chains.index.names = ['start', 'link']

        # Error check no duplicates
        assert not chains.duplicated().any()

        # Error check that the length of all chains plus the number of
        # singletons singletons equals the data length
        n_singletons = (~np.in1d(streakass, chains.values)).sum()
        assert n_singletons + len(chains) == len(streakass)

        # Assign
        for start, this_chain in chains.groupby(level=0):
            streakass[this_chain.values] = start

        # densify streakass
        mwe['streak'] = streakass
        mwe['streak'] = mwe['streak'].rank(method='dense').astype(np.int)

    # method 3: this doesn't converge
    elif method == 'even_odd':
        # Get streaks, frames, and indexes as arrays
        streakass = np.arange(len(mwe)).astype(np.int)
        frameidx = assignments.index.get_level_values(level='frame0').values
        idx0 = assignments.index.get_level_values(level='index0')
        idx1 = assignments['index1'].values
        
        # ilocify the indexes into streakass
        ser = pandas.Series(range(len(mwe)), index=mwe.index)
        iidx0 = ser.loc[idx0].values
        iidx1 = ser.loc[idx1].values
        
        # propagate
        even_frames = assignments.index.levels[0][::2].values
        odd_frames = assignments.index.levels[0][1::2].values
        
        while True:
            # do even
            frame_mask = np.in1d(frameidx, even_frames)
            n_to_change_even = (streakass[iidx1[frame_mask]] != streakass[iidx0[frame_mask]]).sum()
            streakass[iidx1[frame_mask]] = streakass[iidx0[frame_mask]]
            
            # do odd
            frame_mask = np.in1d(frameidx, odd_frames)
            n_to_change_odd = (streakass[iidx1[frame_mask]] != streakass[iidx0[frame_mask]]).sum()
            streakass[iidx1[frame_mask]] = streakass[iidx0[frame_mask]]            
        
            if n_to_change_even + n_to_change_odd == 0:
                break
            else:
                print n_to_change_even, n_to_change_odd
        
        # densify streakass
        mwe['streak'] = streakass
        mwe['streak'] = mwe['streak'].rank(method='dense').astype(np.int)

    return mwe
