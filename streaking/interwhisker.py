from __future__ import absolute_import
from __future__ import division
from builtins import range
from past.utils import old_div
import itertools
import numpy as np
import pandas
import scipy.special
import scipy.stats
from . import base

def update_relationships2(mwe, drop_same=False, drop_nan=True, key='object'):
    """Within each frame, merge on key and apply vectorized_ls_dist

    drop_same : if True, drop rows where the same object is keyed on itself
    drop_nan : if True, drop rows where 'dist' is nan
        This occurs if they share no overlap in x
    
    Returns: DataFrame with columns 'frame', key0, key1, and 'dist'
        'dist' is the distance between each pair of rows in mwe
    """
    ## More efficient ls_dist
    key0 = key + '0'
    key1 = key + '1'
    
    # Keep only the non-null objects, and the relevant geometry
    mwe2 = mwe.loc[~mwe[key].isnull(), 
        ['tip_x', 'tip_y', 'fol_x', 'fol_y', 'frame', key]]
    
    # Merge with itself, suffixing with 0 and 1
    merged = mwe2.merge(mwe2, on='frame', suffixes=('0', '1'))

    # Optionally drop same-object
    if drop_same:
        merged = merged.loc[merged[key0] != merged[key1]]

    # Vectorize
    merged['dist'] = base.vectorized_ls_dist(merged)
    
    # Drop NaN dists, this seems to be rare
    if drop_nan:
        merged = merged.loc[~merged['dist'].isnull()]
    
    # Keep only some columns
    return merged[['frame', key0, key1, 'dist']]

def test_all_alignments_for_ordering(mwe, next_frame_streaks, alignments,
    distrs, streak2object_ser,
    min_data_count=50, cost_floor=-6, clamp_std=10,
    verbose=True, key='object',
    ):
    """Test all possible alignments for streaks to known objects for ordering.
    
    mwe : mwe
    next_frame_streaks : list of streaks to test
    alignments : alignments to test
    distrs : stored information about inter-object distance distributions
    streak2object_ser : Series, index known streaks, values labeled objects
    min_data_count : if there are fewer than this many pairs, assumed
        standard deviation is `clamp_std`
    clamp_std : lower limit on standard deviation of each pair
    cost_floor : floor on cost for each possible assignment
    key : how to identify the object within each pair
    
    We define `streaks_to_assign` as the values in `next_frame_streaks`
    which are not in the index of `streak2object_ser`. We define
    `available_objects` as the unique values of streak2object_ser for which
    
    
    We consider all possible alignments between `next_frame_streaks` and
    `available_objects`. Each alignment is a set of assignments between
    streaks and objects. The cost of each alignment is the sum over all
    its assignment pairs, of the log(p-value) of observing that inter-streak
    distance assuming they came from that inter-object distance distribution.
    
    Returns: llik_ser, alignment_llik_df, alignment_costs
        llik_ser : The cost of each possible assignment
        alignment_llik_df : The cost of each assignment in each alignment
        alignment_costs : Sum cost of each alignment        
    """
    # Take all data from each streak in `next_frame_streaks`
    streak_data = mwe.loc[mwe['streak'].isin(next_frame_streaks)]
    
    method = 'fast'
    
    if method == 'old':
        # Iterate over all such frames and construct probe_distrs
        rec_l = []
        for frame, frame_data in streak_data.groupby('frame'):
            # All pairs of streaks within this frame
            for idx0 in range(len(frame_data)):
                for idx1 in range(len(frame_data)):
                    # Skip same guy
                    if idx0 == idx1:
                        continue
                    
                    # Calculate distance between this pair
                    dist = base.ls_dist(
                        frame_data['tip_x'].iloc[idx0],
                        frame_data['tip_y'].iloc[idx0],
                        frame_data['fol_x'].iloc[idx0],
                        frame_data['fol_y'].iloc[idx0],
                        frame_data['tip_x'].iloc[idx1],
                        frame_data['tip_y'].iloc[idx1],
                        frame_data['fol_x'].iloc[idx1],
                        frame_data['fol_y'].iloc[idx1],
                    )
                    
                    # Append distance between each pair
                    rec_l.append((frame, 
                        frame_data['streak'].iloc[idx0], 
                        frame_data['streak'].iloc[idx1], 
                        dist))
        probe_distrs = pandas.DataFrame.from_records(rec_l, 
            columns=['frame', 'streak0', 'streak1', 'dist'])

        # mean over all frame for each object
        # this is the mean relative distance between all pairs of streaks
        # should probably take all actual sample points, because some objects may
        # have many more data than others
        pd_mean = probe_distrs.groupby(['streak0', 'streak1'])['dist'].mean()
    
    elif method == 'fast':
        # More efficient
        probe_distrs = update_relationships2(
            streak_data, drop_same=True, drop_nan=True, key='streak')
        pd_mean = probe_distrs.groupby(['streak0', 'streak1'])['dist'].mean()

    
    ## Now compare every pair of streaks, to every pair of known objects in distrs
    key0 = key + '0'
    key1 = key + '1'
    
    # We have streaks A,B,C,... to assign to objects 1,2,3,...
    # For every pairwise inter-streak distance A-B, compare to every pairwise
    # inter-object distance distribution 1-2.
    method = 'fast'
    if method == 'fast':
        # Mean and std of every object*object distr, excluding same*same
        gobj = distrs.query('%s != %s' % (key0, key1)).groupby([
            key0, key1])['dist']    
        mdist = gobj.mean()
        sdist = gobj.std()
        ndist = gobj.count()

        # Clamp sdist
        sdist[ndist < min_data_count] = clamp_std
        sdist[sdist < clamp_std] = clamp_std

        # Evaluate every streak*streak probe dist under these models
        # Normalize every probe value
        normprobe = pandas.DataFrame(np.transpose(len(mdist) * [pd_mean.values]),
            index=pd_mean.index, columns=mdist.index)
        normprobe = normprobe.subtract(mdist)
        normprobe = normprobe.divide(sdist)
        
        # CDF it
        logp_probe = np.log10(2 * (1 - scipy.stats.norm.cdf(
            np.abs(normprobe.values))))
        logp_probe = pandas.DataFrame(logp_probe, index=normprobe.index,
            columns=normprobe.columns)
        llik_ser = logp_probe.stack(key0).stack(key1)
    
    elif method == 'old':
        rec_l = []
        gobj = distrs.groupby([key0, key1])['dist']    
        
        # Iterate over pairs of streaks
        for (streak0, streak1), streak_dist in pd_mean.items():
            # Iterate over pairs of objects
            for (object0, object1), object_dist_distr in gobj:
                # Skip same object (this shouldn't even be in the data anyway)
                if object0 == object1:
                    continue
                
                # Parameterize distr
                mdist = object_dist_distr.mean()
                
                # If too little data, just assume clamp_std
                if len(object_dist_distr) < min_data_count:
                    sdist = clamp_std
                else:
                    sdist = object_dist_distr.std()
                
                # Floor sdist
                if sdist < clamp_std:
                    sdist = clamp_std
                
                # Calculate p-value of data
                normalized_value = old_div((streak_dist - mdist), sdist)
                llik = np.log10(2 * (1 - scipy.stats.norm.cdf(
                    np.abs(normalized_value))))
                
                # Store
                rec_l.append((streak0, streak1, object0, object1, llik))
        
        # DataFrame it
        llik_df = pandas.DataFrame.from_records(rec_l, 
            columns=('streak0', 'streak1', 'object0', 'object1', 'llik'))
        
        # Index
        llik_ser = llik_df.set_index(
            ['streak0', 'streak1', 'object0', 'object1'])['llik'].sort_index()


    ## Cost of each alignment
    # Floor llik_ser
    llik_ser2 = llik_ser.copy()
    llik_ser2[llik_ser2 < cost_floor] = cost_floor
    
    method = 'fast' # 'old'
    if method == 'fast':
        # For efficiency, first get idxs, then do one lookup
        all_idxs = []
        n_alignment_l = []
        for n_alignment, alignment in enumerate(alignments):
            iobj = itertools.combinations(alignment, 2)
            
            # Iterate over all combinations of assignments for this alignment
            for (object0, streak0), (object1, streak1) in iobj:
                # Match the ordering of indexing into llik_ser
                all_idxs.append((streak0, streak1, object0, object1))
                n_alignment_l.append(n_alignment)
        
        # Error check
        n_combinations = scipy.special.comb(len(alignments[0]), 2, exact=True)
        assert len(all_idxs) == n_combinations * len(alignments)
        
        # One lookup
        # This line dominates the running time, at least when there are
        # many realignments
        vals = llik_ser2.loc[all_idxs].values
        
        # DataFrame it
        alignment_llik_df = pandas.DataFrame.from_records(all_idxs,
            columns=['object0', 'streak0', 'object1', 'streak1']
        )
        alignment_llik_df['n_alignment'] = n_alignment_l
        alignment_llik_df['llik'] = vals
    elif method == 'old':
        alignment_llik_l = []
        for n_alignment, alignment in enumerate(alignments):
            # Iterate over all pairs of nodes (object=streak assignments)
            all_node_pairs = itertools.combinations(alignment, 2)
            alignment_sum = 0
            for (object0, streak0), (object1, streak1) in all_node_pairs:
                llik = llik_ser2.loc[streak0, streak1, object0, object1]
                alignment_llik_l.append((streak0, streak1, object0, object1,
                    n_alignment, llik))
        alignment_llik_df = pandas.DataFrame.from_records(alignment_llik_l,
            columns=['streak0', 'streak1', 'object0', 'object1', 
                'n_alignment', 'llik'],
        )
    
    # Sum by alignment
    alignment_costs = alignment_llik_df.groupby('n_alignment')['llik'].sum()

    return llik_ser, alignment_llik_df, alignment_costs

def test_relationships(mwe, stf_data, distrs, min_data_count=50, clamp_std=10):
    # Apply the line dist to each overlapping known streak
    rec_l = []
    
    # Get all frames containing the streak of interest
    matching_frames = stf_data.loc[:, 'frame']
    
    # Get all rows of mwe for those frames
    matching_rows = mwe.loc[mwe.frame.isin(matching_frames)]
    
    # Iterate over all such frames
    for frame, frame_data in matching_rows.groupby('frame'):
        # The test row is the one in stf_data
        test_idx = stf_data[stf_data['frame'] == frame].index[0]

        # Compare against all remaining rows
        remaining_frame_data = frame_data[~frame_data['object'].isnull()]
        for idx0, object0 in remaining_frame_data['object'].items():
            dist = ls_dist(
                frame_data.loc[idx0, 'tip_x'],
                frame_data.loc[idx0, 'tip_y'],
                frame_data.loc[idx0, 'fol_x'],
                frame_data.loc[idx0, 'fol_y'],
                frame_data.loc[test_idx, 'tip_x'],
                frame_data.loc[test_idx, 'tip_y'],
                frame_data.loc[test_idx, 'fol_x'],
                frame_data.loc[test_idx, 'fol_y'],
            )
            rec_l.append((frame, object0, dist))
    distrs_vs_test = pandas.DataFrame.from_records(rec_l, 
        columns=['frame', 'object0', 'dist'])

    # mean over all frame for each object
    # should probably take all actual sample points, because some objects may
    # have many more data than others
    dvt_object_mean = distrs_vs_test.groupby('object0')['dist'].mean()

    # Now for every available object and every comparison value in distrs_vs_test,
    # find the probability the comparison value came from that distr
    # Not sure how to do this! Is it 1-pvalue of the data?
    gdvt = distrs.groupby(['object0', 'object1'])
    rec_l = []
    for available_object in available_objects:
        for comparison_object in dvt_object_mean.index:
            value = dvt_object_mean.loc[comparison_object]
            dist = gdvt.get_group((comparison_object, available_object))['dist']
            mdist = dist.mean()
            
            # Get the standard deviation of the distribution
            if len(dist) < min_data_count:
                sdist = clamp_std
            else:
                sdist = dist.std()
            
            # This doesn't work because the pdf is always really low
            # llik = scipy.stats.norm(mdist, sdist).logpdf(value)
            
            normalized_value = old_div((value - mdist), sdist)
            llik = np.log10(2 * (1 - scipy.stats.norm.cdf(np.abs(normalized_value))))
            rec_l.append((available_object, comparison_object, llik))
    llik_df = pandas.DataFrame.from_records(rec_l, 
        columns=('available', 'comparison', 'llik')).set_index(
        ['available', 'comparison'])['llik'].unstack()
    
    return llik_df

def update_relationships(mwe):
    """Model the relationships between known objects in mwe
    
    # distrs vs object1,for visualizing
    #~ object2dist = dict(list(
        #~ distrs[distrs.object0 == 1].groupby('object1')['dist']))
    #~ hist([dist.values for dist in object2dist.values()], bins=100, histtype='step')    

    Returns: distrs
    """
    ## Calculate statistics on typical relationships between objects
    # calculate line seg dist between every pair of segments
    rec_l = []
    for frame, frame_data in mwe[~mwe.object.isnull()].groupby('frame'):
        for idx0, object0 in frame_data['object'].items():
            for idx1, object1 in frame_data['object'].items():
                dist = ls_dist(
                    frame_data.loc[idx0, 'tip_x'],
                    frame_data.loc[idx0, 'tip_y'],
                    frame_data.loc[idx0, 'fol_x'],
                    frame_data.loc[idx0, 'fol_y'],
                    frame_data.loc[idx1, 'tip_x'],
                    frame_data.loc[idx1, 'tip_y'],
                    frame_data.loc[idx1, 'fol_x'],
                    frame_data.loc[idx1, 'fol_y'],
                )
                rec_l.append((frame, object0, object1, dist))
    distrs = pandas.DataFrame.from_records(rec_l, 
        columns=['frame', 'object0', 'object1', 'dist'])    
    
    return distrs

def update_relationships_iterative(distrs, new_data):
    """Concat to existing model (for efficiency)"""
    # calculate line seg dist between every pair of segments
    rec_l = []
    for frame, frame_data in new_data[~new_data.object.isnull()].groupby('frame'):
        for idx0, object0 in frame_data['object'].items():
            for idx1, object1 in frame_data['object'].items():
                dist = ls_dist(
                    frame_data.loc[idx0, 'tip_x'],
                    frame_data.loc[idx0, 'tip_y'],
                    frame_data.loc[idx0, 'fol_x'],
                    frame_data.loc[idx0, 'fol_y'],
                    frame_data.loc[idx1, 'tip_x'],
                    frame_data.loc[idx1, 'tip_y'],
                    frame_data.loc[idx1, 'fol_x'],
                    frame_data.loc[idx1, 'fol_y'],
                )
                rec_l.append((frame, object0, object1, dist))
    new_distrs = pandas.DataFrame.from_records(rec_l, 
        columns=['frame', 'object0', 'object1', 'dist'])    
    
    return pandas.concat([distrs, new_distrs], axis=0, ignore_index=True)
