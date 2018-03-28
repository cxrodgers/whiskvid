import numpy as np
import pandas
import clumping
import geometry
import interwhisker
import itertools
import animation

def define_alignments(next_frame_streaks, streak2object_ser, verbose=True):
    """Define all possible alignments from objects to streaks
    
    next_frame_streaks : streaks to consider
    streak2object_ser : already defined relationships
    
    First we figure out which streaks need to be assigned versus are
    already assigned. Then we figure out which objects are available. Then
    we figure out all alignments, which consists of a series of assignments.
    
    Returns: alignments, a list of length N
        N = all possible combinations between available objects and streaks
        to assign.
    """
    
    ## Which streaks need to be assigned
    # These streaks have already been assigned
    pre_assigned_streaks = streak2object_ser.reindex(
        next_frame_streaks).dropna().astype(np.int)

    # Identify which streaks need to be assigned
    streaks_to_assign = [streak for streak in next_frame_streaks
        if streak not in streak2object_ser.index]
    
    
    ## Which objects are available for assignment
    # Identify which objects are available
    available_objects = [obj for obj in streak2object_ser.unique()
        if obj not in pre_assigned_streaks.values]
    
    print "need to assign %r to %r;\n%r already assigned to %r" % (
        streaks_to_assign, available_objects, 
        pre_assigned_streaks.index, pre_assigned_streaks.values,
    )
    
    if len(streaks_to_assign) > len(available_objects):
        1/0
    
    ## Define alignments from objects to streaks
    # Pre-assigned streaks
    pre_assigned_assignments = zip(
        pre_assigned_streaks.values,
        pre_assigned_streaks.index.values,
    )
    
    if len(available_objects) < len(streaks_to_assign):
        # More streaks than objects, need to add a new object
        1/0
        
    else:
        # More objects than streaks, or the same number
        # All streaks_to_assign-length permutations of available_objects
        permuted_objects_l = itertools.permutations(
            available_objects, len(streaks_to_assign))
        
        # Zip them up with streaks_to_assign, appending fixed assignments
        alignments = [
            zip(permuted_objects, streaks_to_assign) + pre_assigned_assignments
            for permuted_objects in permuted_objects_l
        ]
    
    return alignments

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
        key].values

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
    """Returns frame with most simultaneous streaks
    
    Also inserts streak_length into mwe
    """
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
        """
        ## Test ordering
        if self.verbosity >= 2:
            print "measuring alignment costs"
        
        if oracular:
            llik_ser, alignment_llik_df, alignment_costs = (
                interwhisker.test_all_alignments_for_ordering(
                    self.classified_data, 
                    streaks_in_frame, 
                    alignments,
                    self.oracular_interwhisker_distrs, 
                    self.streak2object_ser,
                    key='color_group',
                )
            )
            alignment_costs.name = 'alignment'
        
        else:
            llik_ser, alignment_llik_df, alignment_costs = (
                interwhisker.test_all_alignments_for_ordering(
                    self.classified_data, 
                    streaks_in_frame, 
                    alignments,
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
        
        return {
            'alignments': alignments,
            'interwhisker_costs_by_alignment': alignment_costs,
            'interwhisker_costs_by_assignment': alignment_llik_df,
            'interwhisker_costs_lookup_series': llik_ser,
            'geometry_costs_by_alignment': geometry_costs_by_alignment,
            'geometry_costs_by_assignment': geometry_costs,
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

    def get_oracular_votes_df(self):
        """Return the results of all votes from all constraints thus far"""
        # Concat the votes
        votes_df = pandas.concat(self.oracular_votes_l, axis=0, 
            keys=self.oracular_vote_keys_l, 
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
        if not hasattr(self, 'clumped_data'):
            self.clump()


        ## Choose starting point 
        self.frame_start = find_frame_with_most_simultaneous_streaks(
            self.clumped_data)
        

        ## Create initial ordering and self.classified_data
        if not hasattr(self, 'classified_data'):
            self.streak2object_ser, self.classified_data = (
                determine_initial_ordering(self.clumped_data, self.frame_start)
            )
        elif not hasattr(self, 'streak2object_ser'):
            # regenerate streak2object_ser
            self.streak2object_ser = self.classified_data[
                ['object', 'streak']].dropna().drop_duplicates(
                'streak').astype(np.int).set_index('streak')[
                'object'].sort_index()
        
        if use_oracular:
            # The most common color_group assigned to each streak
            # Some streaks may erroneously combine across color groups
            oracular_streak2object_ser = self.classified_data.groupby(
                'streak')['color_group'].apply(
                lambda ser: ser.value_counts().idxmax()
            )            
        
        # Identify where each streak first appears
        self.first_frame_of_each_streak = self.classified_data[
            ['streak', 'frame']].dropna().drop_duplicates('streak')


        ## initialize models
        self.update_geometry_model()
        self.update_interwhisker_model()
        n_geo_rows_last_update = np.sum(~self.classified_data['object'].isnull())

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
            self.oracular_final_selection_keys_l = []
            self.oracular_final_selection_l = []
        
        # Loop till break
        while True:
            if self.verbosity >= 1:
                print "info: starting frame %d" % self.current_frame
            
            ## Animate
            if self.animate:
                self.update_animation()
            
            ## Get data
            streaks_and_objects = pick_streaks_and_objects_for_current_frame(
                self.classified_data, self.current_frame,
                self.streak2object_ser,
            )
            
            if use_oracular:
                # Create a test column of color_group
                self.classified_data['color_group_test'] = (
                    self.classified_data['color_group'].copy()
                )
                
                # test_streaks are all streaks that begin in this frame
                test_streaks = self.first_frame_of_each_streak.loc[
                    self.first_frame_of_each_streak['frame'] == 
                    self.current_frame, 'streak'].values                
                
                # Temporarily blank them out
                self.classified_data.loc[
                    self.classified_data['streak'].isin(test_streaks), 
                    'color_group_test'] = np.nan
                
                # Also blank out oracular streak assignments
                test_oracular_streak2object_ser = (
                    oracular_streak2object_ser.drop(test_streaks))
                
                # Pick streaks and objects
                oracular_streaks_and_objects = (
                    pick_streaks_and_objects_for_current_frame(
                        self.classified_data, self.current_frame,
                        test_oracular_streak2object_ser, 
                        key='color_group_test',
                    )
                )                
            
            # Skip if no work
            if len(streaks_and_objects['unassigned_streaks']) == 0:
                next_frame = choose_next_frame(self.classified_data, 
                    self.current_frame)                
                if self.verbosity >= 2:
                    print "skipping frame,", self.current_frame
                self.current_frame = next_frame
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
            alignments = define_alignments(
                streaks_and_objects['streaks_in_frame'],
                self.streak2object_ser,
            )
            
            if use_oracular:
                oracular_alignments = define_alignments(
                    oracular_streaks_and_objects['streaks_in_frame'],
                    test_oracular_streak2object_ser,
                )                
            
            
            ## Test all constraints
            # Test the constraints
            constraint_tests = self.test_all_constraints(alignments, 
                streaks_and_objects['streaks_in_frame'],
            )

            # Repeat for oracular
            if use_oracular:
                # Test the constraints
                oracular_constraint_tests = self.test_all_constraints(
                    oracular_alignments, 
                    oracular_streaks_and_objects['streaks_in_frame'],
                    oracular=True,
                )
            
            
            ## Combine geometry and alignment metrics
            # Combine metrics
            metrics = pandas.DataFrame([
                constraint_tests['geometry_costs_by_alignment'],
                constraint_tests['interwhisker_costs_by_alignment'],
                ]).T
            
            # Account for the differing dynamic ranges of each metric
            # Simply standardize each to [-1, 0]
            standardized_metrics = metrics.sub(metrics.min()).divide(
                metrics.max() - metrics.min())
            standardized_metrics[standardized_metrics.isnull()] = 1.0
            
            # Weighted sum
            overall_metrics = (
                .5 * standardized_metrics['alignment'] +
                .5 * standardized_metrics['geometry']
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
                    ]).T
                
                oracular_standardized_metrics = oracular_metrics.sub(
                    oracular_metrics.min()).divide(
                    oracular_metrics.max() - oracular_metrics.min())
                oracular_standardized_metrics[
                    oracular_standardized_metrics.isnull()] = 1.0

                # Weighted sum
                oracular_overall_metrics = (
                    .5 * oracular_standardized_metrics['alignment'] +
                    .5 * oracular_standardized_metrics['geometry']
                )
                
                # Choose the best alignment
                oracular_best_choice_idx = np.argmax(oracular_overall_metrics)
                oracular_best_alignment = oracular_alignments[
                    oracular_best_choice_idx]
                oracular_best_choice_llik = oracular_overall_metrics[
                    oracular_best_choice_idx]                


            ## Actually assign
            self.do_assignment(best_alignment, best_choice_llik)


            ## Store some metrics
            self.perf_rec_l.append({
                'frame': self.current_frame, 
                'cost': best_choice_llik,
                'n_alignments': len(alignments),
                'metrics_of_best': metrics.iloc[best_choice_idx],
                'std_metrics_of_best': standardized_metrics.iloc[
                    best_choice_idx],
            })
            
            # The vote of each metric
            for metric in metrics.columns:
                best_row = metrics[metric].idxmax()
                if np.isnan(best_row):
                    # This happens when all alignments are NaN
                    # For instance, if there is only one assignment per
                    # alignment
                    continue
                metric_vote = alignments[best_row]
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
                    'n_alignments': len(oracular_alignments),
                    'metrics_of_best': oracular_metrics.iloc[
                        oracular_best_choice_idx],
                    'std_metrics_of_best': oracular_standardized_metrics.iloc[
                        oracular_best_choice_idx],                        
                })
                
                # The vote of each metric
                for metric in oracular_metrics.columns:
                    # Skip worthless metric
                    oracular_metric_vote = oracular_alignments[
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
                
                # The overall vote
                self.oracular_final_selection_l.append(pandas.Series(
                    *np.transpose(oracular_best_alignment), 
                    name='color_group').loc[
                    oracular_streaks_and_objects['unassigned_streaks']
                ])
                self.oracular_final_selection_keys_l.append(self.current_frame)


            ## Update models
            # Only update if we have enough new data
            n_geo_rows = np.sum(~self.classified_data['object'].isnull())
            if n_geo_rows_last_update is None or (
                n_geo_rows >= 1.01 * n_geo_rows_last_update):
                # Update the model
                self.update_geometry_model()
                n_geo_rows_last_update = n_geo_rows
            
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
