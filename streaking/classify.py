import numpy as np
import pandas
import clumping
import geometry
import interwhisker
import animation
import smoothness

def find_frame_with_most_simultaneous_streaks(mwe):
    """Returns frame with most simultaneous streaks"""
    # insert length of streaks
    streak_length = mwe['streak'].value_counts().sort_index()
    mwe['streak_length'] = streak_length.loc[mwe['streak']].values

    # Crazy memory problems
    #~ mwe['streak_length'] = mwe['streak'].replace(
    #~ to_replace=streak_length.index, value=streak_length.values)

    # count number of streaks by frame
    n_streaks_by_frame = mwe.groupby('frame')['streak'].count()

    # select frames with max streaks
    max_streaks = n_streaks_by_frame.max()
    frames_with_max_streaks = n_streaks_by_frame.index[
        n_streaks_by_frame == max_streaks]

    # of those, sort by mean streak length
    mean_streak_length = mwe.loc[
        mwe.frame.isin(frames_with_max_streaks)].groupby('frame')[
        'streak_length'].mean().sort_values()
    
    frame_start = int(mean_streak_length.index[-1])
    
    return frame_start

def determine_initial_ordering(mwe, frame_start):
    """Initalize `object` column in `mwe` based on `frame_start`
    
    Identifies streaks at frame `frame_start`. Each streak is given an
    object label, ordered by their relative distances.
    
    Returns: 
        streak2object_ser
        mwe, now with `object` column
    """
    ## Define the objects
    # streaks in frame_start
    streaks_in_frame_start = list(mwe[mwe.frame == frame_start][
        'streak'].values)

    # object labels, for now assume always integer strating with 0
    known_object_labels = list(range(len(streaks_in_frame_start)))

    # create 'object' column and label these streaks
    mwe['object'] = mwe['streak'].replace(
        to_replace=streaks_in_frame_start, 
        value=known_object_labels,
    )

    # unknown objects are labeled with np.nan
    mwe.loc[~mwe['streak'].isin(streaks_in_frame_start), 'object'] = np.nan


    ## Reorder object by distance
    # Estimate distances
    distrs_temp = interwhisker.update_relationships2(mwe)

    # mean distance between each object
    mean_dist_between_objects = distrs_temp.pivot_table(
        index='object0', columns='object1', values='dist')
        
    # infer the ordering of the whiskers
    ordered_whiskers = mean_dist_between_objects.mean().argsort().values

    # Re-order everything
    # Take `streaks_in_frame_start` in the order specified by `ordered_whiskers`
    ordered_sifs = np.array(streaks_in_frame_start)[ordered_whiskers]

    # Repeat the code above for setting object column
    mwe['object'] = mwe['streak'].replace(
        to_replace=ordered_sifs,
        value=known_object_labels,
    )
    mwe.loc[~mwe['streak'].isin(ordered_sifs), 'object'] = np.nan

    # This is the mapping between streaks and objects
    streak2object_ser = pandas.Series(known_object_labels, ordered_sifs)

    return streak2object_ser, mwe

def classify(mwe, DO_ANIMATION=False, ANIMATION_START=0):
    """Run the streaking algorithm on a dataset
    
    """
    mwe = mwe.copy()
    

    ## Clump segments into streaks
    # Parameterize center
    mwe['center_x'] = mwe[['fol_x', 'tip_x']].mean(1)
    mwe['center_y'] = mwe[['fol_y', 'tip_y']].mean(1)

    # Clump
    print "clumping"
    mwe = clumping.clump_segments_into_streaks(mwe)
    print "done"


    ## Choose starting point and create `object` columns
    frame_start = find_frame_with_most_simultaneous_streaks(mwe)
    streak2object_ser, mwe = determine_initial_ordering(mwe, frame_start)


    ## initialize models
    print "updating geometry"
    geometry_model_columns = ['tip_x', 'tip_y', 'fol_x', 'fol_y']
    model, geometry_scaler = geometry.update_geometry(mwe, geometry_model_columns)

    print "updating relationships"
    distrs = interwhisker.update_relationships2(mwe)


    ## init animation
    if DO_ANIMATION:
        unclassified_lines = []
        object2line = {}
        f, ax = plt.subplots()
        animation.init_animation(mwe, object2line, f, ax)


    ## Iterate through frames
    next_frame = frame_start
    perf_rec_l = []
    while True:
        ## Animate
        if DO_ANIMATION:
            if next_frame > ANIMATION_START:
                update_animation(mwe, next_frame, object2line, f, ax, unclassified_lines)    
        
        
        ## Get data
        # Find streaks from next_frame
        next_frame_streaks = mwe.loc[mwe.frame == next_frame, 'streak'].values

        # These streaks have already been assigned
        pre_assigned_streaks = streak2object_ser.reindex(
            next_frame_streaks).dropna().astype(np.int)

        # Identify which streaks need to be assigned
        streaks_to_assign = [streak for streak in next_frame_streaks
            if streak not in streak2object_ser.index]
        
        # Identify which objects are available
        available_objects = [obj for obj in streak2object_ser.unique()
            if obj not in pre_assigned_streaks.values]
        
        # determine if we have any work to do
        n_streaks_to_assign = len(streaks_to_assign)
        
        # continue if no work
        if n_streaks_to_assign == 0:
            print "skipping frame", next_frame
            next_frame += 1
            continue
        
        # status message
        print "need to assign %d streaks in frame %d" % (
            n_streaks_to_assign, next_frame)
        

        ## Test ordering
        print "measuring alignment costs"
        alignments, llik_ser, alignment_llik_df, alignment_costs = (
            interwhisker.test_all_alignments_for_ordering(
            mwe, next_frame_streaks, distrs, streak2object_ser)
        )
        alignment_costs.name = 'alignment'
        
        
        ## Test geometry
        print "measuring geometry costs"
        geometry_costs, geometry_costs_by_alignment = geometry.measure_geometry_costs(
            mwe, model, next_frame_streaks, alignments,
            geometry_model_columns, geometry_scaler)
        

        ## Test smoothness
        print "measuring smoothness costs"
        (smoothness_costs, smoothness_costs_by_alignment, 
            all_smoothness_dists) = smoothness.measure_smoothness_costs(
            mwe, next_frame_streaks, alignments,
            next_frame)
        smoothness_costs.name = 'smoothness'
        
        
        ## Combine geometry and alignment metrics
        print "choosing"
        # Combine metrics
        metrics = pandas.DataFrame([
            geometry_costs_by_alignment,
            alignment_costs,
            smoothness_costs,
            ]).T
        
        # Weighted sum
        overall_metrics = (
            .35 * metrics['smoothness'] +
            .35 * metrics['alignment'] +
            .3 * metrics['geometry']
        )
        
        # Choose the best alignment
        best_choice_idx = np.argmax(overall_metrics)
        best_alignment = alignments[best_choice_idx]
        best_choice_llik = overall_metrics[best_choice_idx]

        # assign
        print "assigning %r, llik %g" % (best_alignment, best_choice_llik)
        
        # Actually assign
        for obj, streak in best_alignment:
            # This is a pre-assigned assignment in the alignment
            if streak in streak2object_ser.index:
                assert np.all(mwe.loc[mwe['streak'] == streak, 'object'] == obj)
            else:
                # Actually assign
                mwe.loc[mwe['streak'] == streak, 'object'] = obj
                
                # Add to db
                streak2object_ser.loc[streak] = obj


        ## Store some metrics
        perf_rec_l.append({'frame': next_frame, 'cost': best_choice_llik,
            'n_streaks_to_assign': n_streaks_to_assign,
            'n_alignments': len(alignments)})


        ## Update models
        print "updating geometry"
        model, geometry_scaler = geometry.update_geometry(mwe, geometry_model_columns)
        
        print "updating relationships"
        distrs = interwhisker.update_relationships2(mwe)
        print "done"
        

        ## Animate the decision
        # pause here if animating to point out the issue
        if DO_ANIMATION:
            if next_frame > ANIMATION_START:
                #~ plt.pause(.1) # long enough to see the result
                update_animation(mwe, next_frame, object2line, f, ax, unclassified_lines)    
                #~ plt.pause(.1)

        
        ## next frame
        # Stop if no more data
        if mwe.loc[mwe['frame'] >= next_frame, 'object'].isnull().any():
            # Move forward to next frame
            next_frame = mwe.loc[
                (mwe['frame'] >= next_frame) &
                (mwe['object'].isnull())
            ]['frame'].iloc[0]
        
        elif mwe.loc[:, 'object'].isnull().any():
            # No more after this
            # Go back to the first unassigned frame
            next_frame = mwe.loc[(mwe['object'].isnull())]['frame'].iloc[0]            
        
        else:
            # No more data to process, so stop
            break

    res = {
        'mwe': mwe, 
        'distrs': distrs, 
        'streak2object_ser': streak2object_ser,
        'model': model,
        'geometry_model_columns': geometry_model_columns,
        'geometry_scaler': geometry_scaler,
    }

    return res
    
