import pandas
import numpy as np
from base import calculate_center2center_distance_on_merged
import scipy.optimize

def hungarian_assign(merged, dist):
    """Hungarian assignment for clumping
    
    This is called by clump_segments_into_streaks and implements
    the Hungarian algorithm for clumping.
    
    Returns: assignments
        Indexed by frame0 and index0, columns are index1 and cost
    """
    # Get cost by frame
    cost_by_frame = pandas.concat([merged, dist], axis=1).set_index(
        ['frame0', 'index0', 'index1'])['cost'].sort_index()    
    
    # Extract as arrays
    frame0_array = cost_by_frame.index.get_level_values('frame0').values
    index0_array = cost_by_frame.index.get_level_values('index0').values
    index1_array = cost_by_frame.index.get_level_values('index1').values
    cost_array = cost_by_frame.values
    
    # Get the number of index0 and index1 for each frame
    len_index0_by_frame = cost_by_frame.groupby(
        level=[0, 2]).size().groupby(level=0).first()
    len_index1_by_frame = cost_by_frame.groupby(
        level=[0, 1]).size().groupby(level=0).first()
    assert (len_index0_by_frame.index.values == 
        len_index1_by_frame.index.values).all()
    
    # Group by frame and extract the indexes into cost_array for each frame
    gobj = cost_by_frame.reset_index().groupby('frame0')
    frame_idxs_l = gobj.groups.values()
    
    # Check that the frames are sorted the same for each array
    unique_frames = np.sort(np.unique(frame0_array))
    assert (unique_frames == gobj.groups.keys()).all()
    assert (unique_frames == len_index0_by_frame.index.values).all()
    
    
    # Iterate over frames and store raveled indexes of choices
    raveled_idxs_l = []
    for n_frame, frame_idxs in enumerate(frame_idxs_l):
        # Reshape
        cost_matrix_this_frame = cost_array[frame_idxs.values].reshape((
            len_index0_by_frame.values[n_frame],
            len_index1_by_frame.values[n_frame],
        ))

        # Optimize
        # This step takes about 50% of the total time in this function
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            cost_matrix_this_frame)        
        
        # Ravel to match the flat arrays
        raveled_idxs = np.ravel_multi_index((row_ind, col_ind), 
            cost_matrix_this_frame.shape)

        # Offset by the first index of this frame and store
        offset = frame_idxs.values[0]
        raveled_idxs_l.append(offset + raveled_idxs)
    
    # Concatenate raveled indexes and slice accordingly
    all_raveled = np.concatenate(raveled_idxs_l)
    choices_frames = frame0_array[all_raveled]
    choices_index0 = index0_array[all_raveled]
    choices_index1 = index1_array[all_raveled]
    choices_costs = cost_array[all_raveled]

    # DataFrame
    assignments = pandas.DataFrame.from_items([
        ('index0', choices_index0), ('index1', choices_index1),
        ('frame0', choices_frames), ('cost', choices_costs),]
        ).set_index(['frame0', 'index0']).sort_index()
    
    return assignments

def clump_segments_into_streaks(mwe, threshold=32.0, method='hungarian',
    maximum_streak_length=200):
    """Clump segments into streaks
    
    Each segment will be evaluated versus every segment in the following
    frame. The pairwise distance is currently the Cartesian distance
    between the center of each segment. Segments are matched to segments
    in the following frame if the distance is less than `threshold`, using
    either a greedy or Hungarian algoirthm. These assignments are propagated
    throughout and the result is a series of "streaks".
    
    mwe : data
    threshold : threshold for clumping
    method : 'greedy' or 'hungarian'
        'greedy' is much faster
        'hungarian' is optimal
    maximum_streak_length : maximum length of a streak
        If it is longer than this, a new one will be started at this interval
        If None, then they can be any length
    
    Returns : clumped_data
        This is `mwe` with a 'streak' column inserted
    """
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
    if method == 'greedy':
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
        assignments = hungarian_assign(merged, dist)

    # Nullify assignments above criterion
    assignments = assignments.loc[assignments['cost'] < threshold]

    
    ## Propagate assignments
    # This step takes longer than 'greedy' but is trivial compared to
    # 'hungarian'.
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
    chains.name = 'mwe_index'
    
    
    ## Break chains
    if maximum_streak_length is not None:
        # Identify chains that are too long
        start2size = chains.groupby(level=0).size()
        too_long_chains = start2size[start2size > maximum_streak_length].index

        # Iterate over too long chains and break them
        chains_ri = chains.reset_index()
        for start in too_long_chains:
            mask = chains_ri['start'] == start
            sub_chains_ri = chains_ri[mask]
            
            # The new link number will be mod `maximum_streak_length` of 
            # the old link number
            new_link = np.mod(sub_chains_ri['link'], maximum_streak_length)
            
            # The new start is every maximum_streak_length-th link in 
            # the old chain
            new_start = np.concatenate([[this_start] * maximum_streak_length
                for this_start in sub_chains_ri['mwe_index'].values[
                ::maximum_streak_length]]
                )[:len(sub_chains_ri)]
            
            # Store
            chains_ri.loc[mask, 'link'] = new_link
            chains_ri.loc[mask, 'start'] = new_start
        
        # Set index again
        broken_chains = chains_ri.set_index(['start', 'link'])['mwe_index']
        chains = broken_chains
    
    
    ## Error checks
    # Error check no duplicates
    assert not chains.duplicated().any()

    # Error check that the length of all chains plus the number of
    # singletons singletons equals the data length
    n_singletons = (~np.in1d(streakass, chains.values)).sum()
    assert n_singletons + len(chains) == len(streakass)

    
    ## Assign
    for start, this_chain in chains.groupby(level=0):
        streakass[this_chain.values] = start

    # densify streakass
    mwe['streak'] = streakass
    mwe['streak'] = mwe['streak'].rank(method='dense').astype(np.int)

    return mwe
