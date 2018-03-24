import numpy as np
import pandas
import clumping
import geometry
import interwhisker
import animation
import smoothness

def pick_streaks_and_objects_for_current_frame(mwe, next_frame,
    streak2object_ser, key='object'):
    """Pick out streaks to assign and available objects
    
    Slices `mwe` at frame `next_frame`.
    
    Returns: dict
        'streaks_in_frame' : all streaks in frame `next_frame`
        'assigned_streaks' : streaks in frame `next_frame` that are already
            assigned
        'unassigned_streaks' : streaks in frame `next_frame` that aren't
            assigned yet
        'available_objects' : objects that are not yet assigned in `next_frame`
        'objects_in_previous_frame' : objects in previous frame
    """
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

    # Identify objects in previous frame (for smoothness)
    objects_in_previous_frame = mwe.loc[mwe['frame'] == (next_frame - 1),
        key].astype(np.int).values

    res = {
        'unassigned_streaks': streaks_to_assign,
        'assigned_streaks': pre_assigned_streaks,
        'streaks_in_frame': next_frame_streaks,
        'available_objects': available_objects,
        'objects_in_previous_frame': objects_in_previous_frame,
    }
    
    return res

def choose_next_frame(mwe, current_frame):
    """Choose next frame containing unassigned objects with wraparound, or None
    
    """
    # Stop if no more data
    if mwe.loc[mwe['frame'] >= current_frame, 'object'].isnull().any():
        # Move forward to next frame
        next_frame = mwe.loc[
            (mwe['frame'] >= current_frame) &
            (mwe['object'].isnull())
        ]['frame'].iloc[0]
    
    elif mwe.loc[:, 'object'].isnull().any():
        # No more after this
        # Go back to the first unassigned frame
        next_frame = mwe.loc[(mwe['object'].isnull())]['frame'].iloc[0]            
    
    else:
        # No more data to process
        next_frame = None
    
    return next_frame
    

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
    mwe = mwe.copy()
    
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


class Classifier(object):
    """Classifies whiskers"""
    def __init__(self, data, animate=False, animation_start_frame=0,
        verbosity=2):
        """Initialize a new Classifier
        
        data : data to use. Will be copied.
        animate : whether to animate
        animation_start_frame : start animating after this frame
        """
        # Take init kwargs
        self.data = data.copy()
        self.animate = animate
        self.animation_start_frame = animation_start_frame
        self.verbosity = verbosity
        
        # Some other parameters
        self.geometry_model_columns = [
            'tip_x', 'tip_y', 'fol_x', 'fol_y', 'length']
        
        # Parameterization that is useful for some things
        self.data['center_x'] = self.data[['fol_x', 'tip_x']].mean(1)
        self.data['center_y'] = self.data[['fol_y', 'tip_y']].mean(1)        
    
    def clump(self):
        """Clump rows into streaks"""
        if self.verbosity >= 2:
            print "clumping"
        
        self.clumped_data = clumping.clump_segments_into_streaks(self.data)
        
        if self.verbosity >= 2:
            print "done clumping"    
    
    def update_geometry_model(self, oracular=False):
        """Set geometry_model and geometry_scaler from classified_data
        
        oracular: bool
            if False, use the 'object' key and save to 
                self.geometry_model and self.geometry_scaler
            if True, use the 'color_group' key and save to 
                self.oracular_geometry_model and self.oracular_geometry_scaler
        """
        if self.verbosity >= 2:
            print "updating geometry"
        
        if oracular:
            self.oracular_geometry_model, self.oracular_geometry_scaler = (
                geometry.update_geometry(
                    self.classified_data,
                    self.geometry_model_columns, 
                    model_typ='nb',
                    key='color_group',
                )
            )
        else:
            self.geometry_model, self.geometry_scaler = (
                geometry.update_geometry(
                    self.classified_data,
                    self.geometry_model_columns, 
                    model_typ='nb',
                )
            )

        if self.verbosity >= 2:
            print "done updating geometry"

    def update_interwhisker_model(self, oracular=False):
        """Set self.interwhisker_distrs from self.classified_data

        oracular: bool
            if False, use the 'object' key and save to 
                self.interwhisker_distrs
            if True, use the 'color_group' key and save to 
                self.oracular_interwhisker_distrs
        """
        if self.verbosity >= 2:
            print "updating interwhisker"
        
        if oracular:
            self.oracular_interwhisker_distrs = (
                interwhisker.update_relationships2(
                    self.classified_data,
                    key='color_group',
                )
            )
        else:
            self.interwhisker_distrs = (
                interwhisker.update_relationships2(
                    self.classified_data,
                )
            )
        
        if self.verbosity >= 2:
            print "done updating interwhisker"
    
    def setup_animation(self):
        """Set up animation handles"""
        # Init handles
        self.unclassified_lines = []
        self.object2line = {}
        self.f, self.ax = plt.subplots()
        
        # Init animation
        animation.init_animation(self.classified_data, 
            self.object2line, self.f, self.ax,
        )

    def update_animation(self):
        """Update animation if current frame >= self.animation_start_frame"""
        if self.verbosity >= 2:
            print "updating animation"
            
        if self.current_frame >= self.animation_start_frame:
            animation.update_animation(
                self.classified_data, self.current_frame, 
                self.object2line, self.f, self.ax, self.unclassified_lines,
            )

        if self.verbosity >= 2:
            print "done updating animation"

    def test_all_constraints(self, alignments, streaks_in_frame, 
        oracular=False):
        """Test all constraints on all possible alignments
        
        alignments : alignments to test
            Currently overwritten
        
        Returns: dict
            'alignments':
            'interwhisker_costs_by_alignment': 
            'interwhisker_costs_by_assignment':
            'interwhisker_costs_lookup_series':
            'geometry_costs_by_alignment':
            'geometry_costs_by_assignment':
            'smoothness_costs_by_alignment':
            'smoothness_costs_by_assignment':
            'smoothness_costs_lookup_dist_df':
        """
        ## Test ordering
        if self.verbosity >= 2:
            print "measuring alignment costs"
        
        # TODO: accept alignments rather than overwriting
        if oracular:
            alignments, llik_ser, alignment_llik_df, alignment_costs = (
                interwhisker.test_all_alignments_for_ordering(
                    self.classified_data, 
                    streaks_in_frame, 
                    self.oracular_interwhisker_distrs, 
                    self.streak2object_ser,
                )
            )
            alignment_costs.name = 'alignment'
        
        else:
            alignments, llik_ser, alignment_llik_df, alignment_costs = (
                interwhisker.test_all_alignments_for_ordering(
                    self.classified_data, 
                    streaks_in_frame, 
                    self.interwhisker_distrs, 
                    self.streak2object_ser,
                )
            )
            alignment_costs.name = 'alignment'            
        
        
        ## Test geometry
        if self.verbosity >= 2:
            print "measuring geometry costs"
        
        if oracular:
            geometry_costs, geometry_costs_by_alignment = (
                geometry.measure_geometry_costs(
                    self.classified_data, 
                    self.oracular_geometry_model, 
                    streaks_in_frame,
                    alignments,
                    self.geometry_model_columns, 
                    self.oracular_geometry_scaler,
                )
            )
        
        else:
            geometry_costs, geometry_costs_by_alignment = (
                geometry.measure_geometry_costs(
                    self.classified_data, 
                    self.geometry_model, 
                    streaks_in_frame,
                    alignments,
                    self.geometry_model_columns, 
                    self.geometry_scaler,
                )
            )            
        

        ## Test smoothness
        if self.verbosity >= 2:
            print "measuring smoothness costs"
        
        # Actually calculate smoothness
        (smoothness_costs, smoothness_costs_by_alignment, 
            all_smoothness_dists) = smoothness.measure_smoothness_costs(
                self.classified_data,
                streaks_in_frame,
                alignments,
                self.current_frame,
            )
        smoothness_costs.name = 'smoothness'
        

        return {
            'alignments': alignments,
            'interwhisker_costs_by_alignment': alignment_costs,
            'interwhisker_costs_by_assignment': alignment_llik_df,
            'interwhisker_costs_lookup_series': llik_ser,
            'geometry_costs_by_alignment': geometry_costs_by_alignment,
            'geometry_costs_by_assignment': geometry_costs,
            'smoothness_costs_by_alignment': smoothness_costs,
            'smoothness_costs_by_assignment': smoothness_costs_by_alignment,
            'smoothness_costs_lookup_dist_df': all_smoothness_dists,
        }
    
    def do_assignment(self, best_alignment, best_choice_llik):
        """Actually assign the best alignment"""
        # assign
        if self.verbosity >= 1:
            print "assigning %r, llik %g" % (best_alignment, best_choice_llik)
        
        # Actually assign
        for obj, streak in best_alignment:
            # This is a pre-assigned assignment in the alignment
            if streak in self.streak2object_ser.index:
                assert np.all(self.classified_data.loc[
                    self.classified_data['streak'] == streak, 
                    'object'] == obj
                )
            else:
                # Actually assign
                self.classified_data.loc[
                    self.classified_data['streak'] == streak, 'object'] = obj
                
                # Add to db
                self.streak2object_ser.loc[streak] = obj        

    def get_votes_df(self):
        """Return the results of all votes from all constraints thus far"""
        # Concat the votes
        votes_df = pandas.concat(self.votes_l, axis=0, keys=self.vote_keys_l, 
            verify_integrity=True, names=['frame', 'metric']).unstack(
            'metric').sort_index()
        
        return votes_df
    
    def get_state(self):
        return {
            'mwe': self.classified_data, 
            'distrs': self.interwhisker_distrs, 
            'streak2object_ser': self.streak2object_ser,
            'model': self.geometry_model,
            'geometry_model_columns': self.geometry_model_columns,
            'geometry_scaler': self.geometry_scaler,
            'votes_df': self.get_votes_df(),
        }

    def run(self, use_oracular=True):
        """Run on all frames
        
        use_oracular : if True, then also build models based on the
            curated correct answers. This is useful for comparing performance
            given a curated (best-case) model and an actual model.
        """
        ## Clump
        self.clump()


        ## Choose starting point 
        self.frame_start = find_frame_with_most_simultaneous_streaks(
            self.clumped_data)
        
        
        ## Create initial ordering and self.classified_data
        self.streak2object_ser, self.classified_data = (
            determine_initial_ordering(self.clumped_data, self.frame_start)
        )


        ## initialize models
        self.update_geometry_model()
        self.update_interwhisker_model()

        # Oracular
        if use_oracular:
            self.update_geometry_model(oracular=True)
            self.update_interwhisker_model(oracular=True)


        ## init animation
        if self.animate:
            self.setup_animation()


        ## Iterate through frames
        # Which frame we're on
        self.current_frame = self.frame_start
        
        # Various stuff to keep track of
        self.perf_rec_l = []
        self.votes_l = []
        self.vote_keys_l = []

        # Oracular
        if use_oracular:
            self.oracular_perf_rec_l = []
            self.oracular_votes_l = []
            self.oracular_vote_keys_l = []
        
        # Loop till break
        while True:
            ## Animate
            if self.animate:
                self.update_animation()
            
            ## Get data
            streaks_and_objects = pick_streaks_and_objects_for_current_frame(
                self.classified_data, self.current_frame,
                self.streak2object_ser,
            )
            
            if use_oracular:
                oracular_streaks_and_objects = (
                    pick_streaks_and_objects_for_current_frame(
                        self.classified_data, self.current_frame,
                        self.streak2object_ser, key='color_group',
                    )
                )                
            
            # Skip if no work
            if len(streaks_and_objects['unassigned_streaks']) == 0:
                if self.verbosity >= 2:
                    print "skipping frame,", self.current_frame
                self.current_frame += 1
                continue
            
            # Print status
            if self.verbosity >= 2:
                print "goal: assign streaks %r to objects %r" % (
                    streaks_and_objects['unassigned_streaks'],
                    streaks_and_objects['available_objects'],
                )
                print "info: streaks %r already assigned" % (
                    streaks_and_objects['assigned_streaks'],
                )
            
            
            ## Define possible alignments
            alignments = interwhisker.define_alignments(
                streaks_and_objects['streaks_in_frame'],
                self.streak2object_ser,
            )
            
            
            ## Test all constraints
            # Determine whether it's worth calculating smoothness
            # Smoothness can only be calculated for objects that are both in
            # `available_objects` and `objects_in_previous_frame`, in other words,
            # objects corresponding to streaks that just ended.
            # Otherwise it's just going to return the disappearing object penalty
            # for all unfixed assignments, and the same penalty for all fixed
            # assignments
            worth_measuring_smoothness = np.in1d(
                streaks_and_objects['objects_in_previous_frame'], 
                streaks_and_objects['available_objects'],
            ).any()
            
            # Test the constraints
            constraint_tests = self.test_all_constraints(alignments, 
                streaks_and_objects['streaks_in_frame'],
            )

            # Debugging check
            ctscba = constraint_tests['smoothness_costs_by_alignment'].values
            all_smoothness_same = np.allclose(
                ctscba - ctscba[0],
                np.zeros_like(ctscba)
            )
            assert worth_measuring_smoothness == (not all_smoothness_same)
            
            # Repeat for oracular
            if use_oracular:
                oracular_worth_measuring_smoothness = np.in1d(
                    oracular_streaks_and_objects['objects_in_previous_frame'], 
                    oracular_streaks_and_objects['available_objects'],
                ).any()
                
                # Test the constraints
                oracular_constraint_tests = self.test_all_constraints(
                    alignments, 
                    oracular_streaks_and_objects['streaks_in_frame'],
                    oracular=True,
                )

                # Debugging check
                ctscba = oracular_constraint_tests[
                    'smoothness_costs_by_alignment'].values
                oracular_all_smoothness_same = np.allclose(
                    ctscba - ctscba[0],
                    np.zeros_like(ctscba)
                )
                assert oracular_worth_measuring_smoothness == (
                    not oracular_all_smoothness_same)            
            
            
            ## Combine geometry and alignment metrics
            # Combine metrics
            metrics = pandas.DataFrame([
                constraint_tests['geometry_costs_by_alignment'],
                constraint_tests['interwhisker_costs_by_alignment'],
                constraint_tests['smoothness_costs_by_alignment'],
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

            # Repeat for oracular
            if use_oracular:
                # Combine metrics
                oracular_metrics = pandas.DataFrame([
                    oracular_constraint_tests['geometry_costs_by_alignment'],
                    oracular_constraint_tests['interwhisker_costs_by_alignment'],
                    oracular_constraint_tests['smoothness_costs_by_alignment'],
                    ]).T
                
                # Weighted sum
                oracular_overall_metrics = (
                    .35 * oracular_metrics['smoothness'] +
                    .35 * oracular_metrics['alignment'] +
                    .3 * oracular_metrics['geometry']
                )
                
                # Choose the best alignment
                oracular_best_choice_idx = np.argmax(oracular_overall_metrics)
                oracular_best_alignment = alignments[oracular_best_choice_idx]
                oracular_best_choice_llik = oracular_overall_metrics[
                    oracular_best_choice_idx]                


            ## Actually assign
            self.do_assignment(best_alignment, best_choice_llik)


            ## Store some metrics
            self.perf_rec_l.append({
                'frame': self.current_frame, 
                'cost': best_choice_llik,
                'n_alignments': len(alignments),
            })
            
            # The vote of each metric
            for metric in metrics.columns:
                # Skip worthless metric
                if metric == 'smoothness' and not worth_measuring_smoothness:
                    continue
                
                metric_vote = alignments[metrics[metric].idxmax()]
                metric_vote2 = pandas.Series(*np.transpose(metric_vote), 
                    name='object').loc[
                    streaks_and_objects['unassigned_streaks']
                ]
                metric_vote2.index.name = 'streak'
                self.votes_l.append(metric_vote2)
                self.vote_keys_l.append((self.current_frame, metric))

            # Repeat for oracular
            if use_oracular:
                self.oracular_perf_rec_l.append({
                    'frame': self.current_frame, 
                    'cost': oracular_best_choice_llik,
                    'n_alignments': len(alignments),
                })
                
                # The vote of each metric
                for metric in oracular_metrics.columns:
                    # Skip worthless metric
                    if metric == 'smoothness' and not oracular_worth_measuring_smoothness:
                        continue
                    
                    oracular_metric_vote = alignments[
                        oracular_metrics[metric].idxmax()]
                    oracular_metric_vote2 = pandas.Series(
                        *np.transpose(oracular_metric_vote), 
                        name='object').loc[
                        oracular_streaks_and_objects['unassigned_streaks']
                    ]
                    oracular_metric_vote2.index.name = 'streak'
                    self.oracular_votes_l.append(oracular_metric_vote2)
                    self.oracular_vote_keys_l.append(
                        (self.current_frame, metric))                


            ## Update models
            self.update_geometry_model()
            self.update_interwhisker_model()            

            # Oracular
            if use_oracular:
                self.update_geometry_model(oracular=True)
                self.update_interwhisker_model(oracular=True)            


            ## Animate the decision
            if self.animate:
                self.update_animation()

            
            ## next frame
            next_frame = choose_next_frame(self.classified_data, 
                self.current_frame)
            
            if next_frame is None:
                # We're done
                break
            else:
                self.current_frame = next_frame
        
        return self.classified_data
