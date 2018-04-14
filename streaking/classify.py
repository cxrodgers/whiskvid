import numpy as np
import pandas
import clumping
import geometry
import interwhisker
import itertools
import animation
import scipy.optimize
import datetime

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
    
    if verbose:
        print "info: need to assign %r to %r" % (
            streaks_to_assign, available_objects)
        print "info: %r already assigned to %r" % (
            list(pre_assigned_streaks.index.values), 
            list(pre_assigned_streaks.values),
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
    streak2object_ser, all_known_objects=None, key='object'):
    """Pick out streaks to assign and available objects
    
    Slices `mwe` at frame `next_frame`.
    
    mwe : data
    next_frame : frame to process
    streak2object_ser : known mapping between streaks and objects
    all_known_objects : list of all possible object labels
        if None, then we take streak2object_ser.unique(), but this requires
        that each available object has already occurred at least once
    key : used to find objects in previous frame
    
    Returns: dict
        'streaks_in_frame' : all streaks in frame `next_frame`
        'assigned_streaks' : streaks in frame `next_frame` that are already
            assigned
        'unassigned_streaks' : streaks in frame `next_frame` that aren't
            assigned yet
        'available_objects' : objects that are not yet assigned in `next_frame`
        'objects_in_previous_frame' : objects in previous frame
    """
    if all_known_objects is None:
        all_known_objects = streak2object_ser.unique()
    
    # Find streaks from next_frame
    next_frame_streaks = mwe.loc[mwe.frame == next_frame, 'streak'].values

    # These streaks have already been assigned
    pre_assigned_streaks = streak2object_ser.reindex(
        next_frame_streaks).dropna().astype(np.int)

    # Identify which streaks need to be assigned
    streaks_to_assign = [streak for streak in next_frame_streaks
        if streak not in streak2object_ser.index]
    
    # Identify which objects are available
    available_objects = [obj for obj in all_known_objects
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
    

def find_frame_with_most_simultaneous_streaks(mwe, take=-1):
    """Returns frame with most simultaneous streaks
    
    take : if -1 take the last one, if -2 take the second to last, etc
        All will have the same number of streaks, so this is a good way to
        get a "better" keystone frame.
    
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
    
    frame_start = int(mean_streak_length.index[take])
    
    return frame_start

def initialize_classified_data(clumped_data, keystone_frame):
    """Initialize `object` column in `clumped_data` based on `keystone_frame`
    
    Identifies streaks at frame `keystone_frame`. Each streak is given an
    object label, ordered by their relative distances.
    
    Returns: `classified_data`
        This is a copy of `clumped_data`, now with `object` column
    """
    classified_data = clumped_data.copy()
    
    ## Define the objects
    if keystone_frame not in classified_data['frame'].values:
        raise ValueError("clumped data must include keystone frame")
    
    # streaks in frame_start
    streaks_in_keystone_frame = list(
        classified_data[classified_data.frame == keystone_frame][
        'streak'].values)

    # object labels, for now assume always integer strating with 0
    known_object_labels = list(range(len(streaks_in_keystone_frame)))

    # create 'object' column and label these streaks
    classified_data['object'] = classified_data['streak'].replace(
        to_replace=streaks_in_keystone_frame, 
        value=known_object_labels,
    )

    # unknown objects are labeled with np.nan
    classified_data.loc[
        ~classified_data['streak'].isin(streaks_in_keystone_frame), 
        'object'] = np.nan


    ## Reorder object by distance
    # Estimate distances
    distrs_temp = interwhisker.update_relationships2(classified_data)

    # mean distance between each object
    mean_dist_between_objects = distrs_temp.pivot_table(
        index='object0', columns='object1', values='dist')
        
    # infer the ordering of the whiskers
    ordered_whiskers = mean_dist_between_objects.mean().argsort().values

    # Re-order everything
    # Take `streaks_in_keystone_frame` in the order specified by `ordered_whiskers`
    ordered_sifs = np.array(streaks_in_keystone_frame)[ordered_whiskers]

    # Repeat the code above for setting object column
    classified_data['object'] = classified_data['streak'].replace(
        to_replace=ordered_sifs,
        value=known_object_labels,
    )
    classified_data.loc[
        ~classified_data['streak'].isin(ordered_sifs), 'object'] = np.nan

    #~ # This is the mapping between streaks and objects
    #~ streak2object_ser = pandas.Series(known_object_labels, ordered_sifs)

    return classified_data


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
        self.multi_angle_bins = 11
        
        # Parameterization that is useful for some things
        self.data['center_x'] = self.data[['fol_x', 'tip_x']].mean(1)
        self.data['center_y'] = self.data[['fol_y', 'tip_y']].mean(1)        
        
        # announcements
        self.frac_complete_announce = 0
        self.frac_complete_announce_interval = .05
    
    def clump(self):
        """Clump rows into streaks"""
        if self.verbosity >= 2:
            print "clumping"
        
        self.clumped_data = clumping.clump_segments_into_streaks(self.data)
        
        if self.verbosity >= 2:
            print "done clumping"    
    
    def update_geometry_model(self):
        """Set geometry_model and geometry_scaler from classified_data
        
        """
        if self.verbosity >= 2:
            print "updating geometry"
        
        # Add an angle column
        if 'frangle' not in self.classified_data.columns:
            frame2angle = self.classified_data.groupby('frame')[
                'angle'].mean()
            self.classified_data['frangle'] = self.classified_data[
                'frame'].map(frame2angle)
        
        # Update
        (self.geometry_angle_bins, self.geometry_fab2model, 
            self.geometry_fab2scaler) = (
            geometry.update_geometry(
                self.classified_data,
                self.geometry_model_columns, 
                model_typ='nb',
                multi_angle_bins=self.multi_angle_bins,
            )
        )

        if self.verbosity >= 2:
            print "done updating geometry"

    def update_interwhisker_model(self):
        """Set self.interwhisker_distrs from self.classified_data

        """
        if self.verbosity >= 2:
            print "updating interwhisker"
        
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
  
    def do_assignment(self, best_alignment):
        """Actually assign the best alignment
        
        best_alignment : seequence of (object, streak) pairs
        
        For existing assignments (those already in self.streak2object_ser),
        this will error-check that they have been done correctly.

        For new assignments (those not in self.streak2object_ser), they
        will be done to `self.classified_data` and `self.streak2object_ser`.
        """
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

    def set_initial_ordering(self, keystone_frame):
        """Initialize the object identities from a keystone frame
        
        `keystone_frame` should be a frame that has the most number of
        simultaneous streaks as any frame in the video. The objects in this
        frame will be ordered and this will define the object id for the
        rest of the classification.
        
        This will overwite the attributes `classified_data` and 
        `streak2object_ser` if they exist!
        """
        # Initialize classified data based on the keystone frame
        self.classified_data = initialize_classified_data(self.clumped_data, 
            keystone_frame)
        
        # Set `streak2object_ser` based on `self.classified_data`
        self.generate_streak2object_ser()

    def generate_streak2object_ser(self):
        """Generate streak2object_ser from classified_data"""
        self.streak2object_ser = self.classified_data[
            ['object', 'streak']].dropna().drop_duplicates(
            'streak').astype(np.int).set_index('streak')[
            'object'].sort_index()        

    def run(self, frame_start=None, warm_start=None, keystone_frame=None):
        """Run on all frames
        
        * If the attribute `clumped_data` is not set, self.clump() is called.
        * If the attribute `classified_data` is set, then it's a warm start.
          Otherwise it's a cold start. If it's a cold start, keystone_frame
          must have been provided and this is use to initialize
          `classified data`.
        * If `streak2object_ser` is not set, then it is set by
          self.generate_streak2object_ser()
        
        frame_start : which frame to start on
        
        """
        self.date_time_start = datetime.datetime.now()
        
        ## Clump
        if not hasattr(self, 'clumped_data'):
            self.clump()


        ## Choose starting point 
        if frame_start is not None:
            self.frame_start = frame_start
        else:
            raise ValueError("must specify frame_start")


        ## Create initial ordering and self.classified_data
        if not hasattr(self, 'classified_data'):
            # Cold start
            # Use the keystone frame to set the initial ordering
            if keystone_frame is None:
                raise ValueError("must provide keystone frame for cold start")
            self.set_initial_ordering(keystone_frame)
        
        else:
            # Warm start
            # See if we need to regenerate streak2object_ser
            if not hasattr(self, 'streak2object_ser'):
                self.generate_streak2object_ser()

        # Identify where each streak first appears
        self.first_frame_of_each_streak = self.classified_data[
            ['streak', 'frame']].dropna().drop_duplicates('streak')


        ## initialize models
        self.update_geometry_model()
        n_geo_rows_last_update = np.sum(~self.classified_data['object'].isnull())

        # Needed by measure_geometry_costs
        all_known_objects = np.sort(np.unique(
            self.classified_data['object'].dropna().astype(np.int).values))
        

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

        # Loop till break
        while True:
            ## Announce frame
            if self.verbosity >= 2:
                print ("%07d info: starting frame" % self.current_frame)
            elif self.verbosity >= 1:
                # Fraction complete
                frac_complete = (np.sum(
                    ~self.classified_data['object'].isnull()) / float(
                    len(self.classified_data)))
                
                if frac_complete > self.frac_complete_announce:
                    print ("%07d info: %0.1f%% complete" % (
                        self.current_frame, 100 * frac_complete))

                    # Update frac_complete_announce
                    self.frac_complete_announce += (
                        self.frac_complete_announce_interval)

            
            ## Animate
            if self.animate:
                self.update_animation()
            
            ## Get data
            streaks_and_objects = pick_streaks_and_objects_for_current_frame(
                self.classified_data, self.current_frame,
                self.streak2object_ser,
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
                    list(streaks_and_objects['assigned_streaks'].values),
                )
            
            
            ## Test all constraints
            # Test geometry costs
            geometry_costs = (
                geometry.measure_geometry_costs(
                    self.classified_data, 
                    self.geometry_angle_bins,
                    self.geometry_fab2model,
                    self.geometry_fab2scaler,
                    streaks_and_objects['streaks_in_frame'],
                    self.geometry_model_columns, 
                    all_known_objects,
                )
            )              
            
            # Extract out only the available objects and unassigned streaks
            sub_geometry_costs = geometry_costs.loc[
                streaks_and_objects['unassigned_streaks'],
                streaks_and_objects['available_objects']
            ]
            assert not sub_geometry_costs.isnull().any().any()
            
            # Hungarian on unassigned stuff
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                -sub_geometry_costs.values)
            
            # These are the new assignments
            best_alignment = [(obj, streak) for obj, streak in zip(
                sub_geometry_costs.columns[col_ind].values, 
                sub_geometry_costs.index[row_ind].values
            )]
            
            # A diferent format of best_alignment
            best_alignment_ser = pandas.Series(*np.transpose(best_alignment),
                name='object')
            best_alignment_ser.index.name = 'streak'
            

            ## Actually assign
            # This line takes about 12% of the total running time and
            # should be optimized
            self.do_assignment(best_alignment)


            ## Store some metrics
            self.perf_rec_l.append({
                'frame': self.current_frame, 
                'assignments': best_alignment_ser,
            })
            

            ## Update models
            # Only update if we have enough new data
            n_geo_rows = np.sum(~self.classified_data['object'].isnull())
            if n_geo_rows_last_update is None or (
                n_geo_rows >= 1.01 * n_geo_rows_last_update):
                
                if self.verbosity >= 1:
                    print ("%07d info: updating geometry model" % 
                        self.current_frame)
                
                # Update the model
                self.update_geometry_model()
                n_geo_rows_last_update = n_geo_rows

                if self.verbosity >= 1:
                    print ("%07d info: done updating geometry model" % 
                        self.current_frame)
            

            ## Animate the decision
            if self.animate:
                self.update_animation()

            
            ## next frame
            # This line takes about 10% of the total running time and
            # should be optimized
            next_frame = choose_next_frame(self.classified_data, 
                self.current_frame)
            assert next_frame != self.current_frame
            
            if next_frame is None:
                # We're done
                break
            else:
                self.current_frame = next_frame
        
        self.date_time_stop = datetime.datetime.now()
        
        return self.classified_data


class Assigner(object):
    """Assigns objects but doesn't update its own model"""
    def __init__(self, classified_data, geometry_fab2model=None,
        geometry_angle_bins=None, geometry_fab2scaler=None,
        geometry_model_columns=None, verbosity=1, 
        frac_complete_announce_interval=.05,
        streak2object_ser=None, all_known_objects=None):
        """Initialize a new Assigner
        
        classified_data : data to use. Will be copied.
            Should already have 'streak' and 'object' columns
        
        geometry_fab2model, geometry_angle_bins, geometry_fab2scaler,
        geometry_model_columns : the model to use
        streak2object_ser : used to pick out streaks and objects so
            it should be up to date
        all_known_objects : used in measure geometry costs
        """
        # Take init kwargs
        self.classified_data = classified_data.copy()
        self.verbosity = verbosity
        
        # Store the model if provided
        self.geometry_model_columns = geometry_model_columns
        self.geometry_angle_bins = geometry_angle_bins
        self.geometry_fab2scaler = geometry_fab2scaler
        self.geometry_fab2model = geometry_fab2model
        
        # Known objects
        self.streak2object_ser = streak2object_ser
        self.all_known_objects = all_known_objects
        
        # announcements
        self.frac_complete_announce = 0
        self.frac_complete_announce_interval = frac_complete_announce_interval
    
    def announce_frame(self):
        """Announce frame and update announcement interval"""
        if self.verbosity >= 2:
            print ("%07d info: starting frame" % self.current_frame)
        
        elif self.verbosity >= 1:
            # Fraction complete
            frac_complete = (np.sum(
                ~self.classified_data['object'].isnull()) / float(
                len(self.classified_data)))
            
            # Announce and update interval
            if frac_complete > self.frac_complete_announce:
                print ("%07d info: %0.1f%% complete" % (
                    self.current_frame, 100 * frac_complete))

                # Update frac_complete_announce
                self.frac_complete_announce += (
                    self.frac_complete_announce_interval)        
    
    def announce_frame_status(self, streaks_and_objects):
        # Print status
        if self.verbosity >= 2:
            print "goal: assign streaks %r to objects %r" % (
                streaks_and_objects['unassigned_streaks'],
                streaks_and_objects['available_objects'],
            )
            print "info: streaks %r already assigned" % (
                list(streaks_and_objects['assigned_streaks'].values),
            )

    def do_assignment(self, best_alignment):
        """Actually assign the best alignment
        
        best_alignment : sequence of (object, streak) pairs
        
        For existing assignments (those already in self.streak2object_ser),
        this will error-check that they have been done correctly.

        For new assignments (those not in self.streak2object_ser), they
        will be done to `self.classified_data` and `self.streak2object_ser`.
        """
        # Error check
        streaks_to_assign, objects_to_assign = [], []
        for obj, streak in best_alignment:
            if streak in self.streak2object_ser.index:
                # This streak has already been assigned
                # Error check for consistency
                assert np.all(self.classified_data.loc[
                    self.classified_data['streak'] == streak, 
                    'object'] == obj
                )
            else:
                # Not yet assigned
                streaks_to_assign.append(streak)
                objects_to_assign.append(obj)
        
        # Add to db
        self.streak2object_ser = self.streak2object_ser.append(
            pandas.Series(objects_to_assign, index=streaks_to_assign),
            verify_integrity=True)
        
        # Apply to data
        self.classified_data['object'] = self.classified_data['streak'].map(
            self.streak2object_ser)

    def run_on_frame(self, frame):
        """Measure costs and do assignment for individual frame
        
        The availables treaks and objects are identified for this frame.
        The geometry costs are measured. The Hungarian algorithm is used
        to choose an assignment. The assignment is made using 
        `self.do_assignment`.
        """
        ## Announce frame
        self.current_frame = frame
        self.announce_frame()


        ## Get data
        streaks_and_objects = pick_streaks_and_objects_for_current_frame(
            self.classified_data, self.current_frame,
            self.streak2object_ser, self.all_known_objects,
        )
        
        # Skip if no work
        if len(streaks_and_objects['unassigned_streaks']) == 0:
            if self.verbosity >= 2:
                print "skipping frame,", self.current_frame
            return
        
        # announce status
        self.announce_frame_status(streaks_and_objects)
        
        
        ## Test geometry costs
        geometry_costs = (
            geometry.measure_geometry_costs(
                self.classified_data, 
                self.geometry_angle_bins,
                self.geometry_fab2model,
                self.geometry_fab2scaler,
                streaks_and_objects['streaks_in_frame'],
                self.geometry_model_columns, 
                self.all_known_objects,
            )
        )              
        
        # Extract out only the available objects and unassigned streaks
        sub_geometry_costs = geometry_costs.loc[
            streaks_and_objects['unassigned_streaks'],
            streaks_and_objects['available_objects']
        ]
        assert not sub_geometry_costs.isnull().any().any()
        
        # Hungarian on unassigned stuff
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            -sub_geometry_costs.values)
        
        # These are the new assignments
        best_alignment = [(obj, streak) for obj, streak in zip(
            sub_geometry_costs.columns[col_ind].values, 
            sub_geometry_costs.index[row_ind].values
        )]


        ## Actually assign
        # This line takes about 12% of the total running time and
        # should be optimized
        self.do_assignment(best_alignment)        

    def run(self, frame_start=None):
        """Run starting from `frame_start` and continuing until done.
        
        Each individual frame, beginning with `frame_start`, is processed
        with `self.run_on_frame`. The next frame is chosen using
        `choose_next_frame`.
        
        The attributes `date_time_start`, `date_time_stop`, and
        `current_frame` are used for debugging information.
        
        frame_start : frame to start on
        
        Returns: self.classified_data
        """
        # Store start time
        self.date_time_start = datetime.datetime.now()

        # Which frame we're on
        self.current_frame = frame_start

        # Loop till break
        while True:
            # Run on frame
            # Will return immediately if no work to do on this frame
            self.run_on_frame(self.current_frame)

            # Choose next frame
            # This line takes about 10% of the total running time and
            # should be optimized
            next_frame = choose_next_frame(self.classified_data, 
                self.current_frame)
            
            # Error check
            # This can happen if assignments aren't being made properly
            # resulting in infinite loop
            assert next_frame != self.current_frame
            
            # Update current frame
            if next_frame is None:
                # We're done
                break
            else:
                self.current_frame = next_frame
        
        # Store stop time
        self.date_time_stop = datetime.datetime.now()
        
        return self.classified_data


## Repair
def evaluate_geometry_costs_by_seg(classifier):
    ## Re-evaluate the geometry model on all rows
    # Insert frangle column
    classifier.classified_data['frangle_bin'] = pandas.cut(
        classifier.classified_data['frangle'], 
        classifier.geometry_angle_bins, 
        labels=False).astype(np.int)        

    # Iterate over frangles
    geometry_costs_l = []
    for fab, sub_mwe in classifier.classified_data.groupby('frangle_bin'):
        # Get corresponding model
        model = classifier.geometry_fab2model[fab]
        scaler = classifier.geometry_fab2scaler[fab]
        
        # Fit
        scaled_stf_data = scaler.transform(
            sub_mwe[classifier.geometry_model_columns].values)
        proba = model.predict_proba(scaled_stf_data)

        # Store
        geometry_costs_l.append(pandas.DataFrame(proba, 
            index=sub_mwe.index, columns=model.classes_))
    
    # Concat
    geometry_costs_by_seg = pandas.concat(geometry_costs_l, axis=0).sort_index()

    # Reindex by all classes in case it never came up
    all_known_objects = np.sort(classifier.classified_data[
        'object'].unique().astype(np.int))
    geometry_costs_by_seg = geometry_costs_by_seg.reindex(all_known_objects, 
        axis=1) 
    geometry_costs_by_seg.columns.name = 'object'
    geometry_costs_by_seg.index.name = 'mwe_index'
    
    # log10
    geometry_costs_by_seg = np.log10(1e-6 + geometry_costs_by_seg)

    return geometry_costs_by_seg

def evaluate_streak_likelihood(geometry_costs_by_seg, classifier):
    """Calculate the likelihood of every streak under its classification
    
    Also calculates the max possible likelihood if every segment in this
    streak could be arbitrarily assigned.
    """
    # Extract the max likelihood of each seg if it could be any object
    max_lk = geometry_costs_by_seg.max(1)
    max_lk.name = 'max_lik'
    argmax_lk = geometry_costs_by_seg.idxmax(1)
    argmax_lk.name = 'max_obj'
    
    # Extract the likelihood of each seg under the assigned object
    assigned_lk_idxs = np.ravel_multi_index((
        np.arange(len(geometry_costs_by_seg), dtype=np.int), 
        classifier.classified_data['object'].values.astype(np.int)), 
        geometry_costs_by_seg.shape)
    assigned_lk = geometry_costs_by_seg.values.flatten()[
        assigned_lk_idxs]
    
    # Concat
    lks = pandas.concat([
        max_lk, argmax_lk,
        pandas.Series(assigned_lk, index=geometry_costs_by_seg.index,
            name='assigned_lik'),
        pandas.Series(classifier.classified_data['object'].values, 
            index=classifier.classified_data.index,
            name='assigned_obj'),
        classifier.classified_data['streak'],
        classifier.classified_data['frame'],
        ], axis=1, verify_integrity=True)
    
    # Sum within streak
    streak_sum_lks = lks.groupby('streak')[['assigned_lik', 'max_lik']].sum()
    
    # Ratio between optimal and assigned
    streak_sum_lks['llr'] = (streak_sum_lks['assigned_lik'] - 
        streak_sum_lks['max_lik'])

    return streak_sum_lks

def repair_classifier(classifier, fold_heldout_results=None, verbosity=1,
    repair_llr_threshold=-3):
    """Run the repair by atomization algorithm.
    
    """
    # Evaluate geometry costs of each segment
    geometry_costs_by_seg = evaluate_geometry_costs_by_seg(classifier)
    
    # Evaluate likelihood of each streak
    streak_sum_lks = evaluate_streak_likelihood(geometry_costs_by_seg, 
        classifier)

    # Identify streaks to fix
    conflict_streaks = streak_sum_lks.loc[
        streak_sum_lks['llr'] < repair_llr_threshold, :].sort_values(
        'llr').index
    
    # We'll repair the data in this copy
    cdata = classifier.classified_data.copy()
    all_known_objects = np.sort(cdata['object'].astype(np.int).unique())
    
    # This one will only accept repairs that are better than original
    cdata2 = classifier.classified_data.copy()

    # Iterate over the conflict streaks, from worst to best
    repair_l = []
    for conflict_streak in conflict_streaks:
        # Identify affected frames
        conflict_streak_frames = cdata.loc[cdata['streak'] == conflict_streak,
            'frame'].unique()
        
        if len(conflict_streak_frames) == 0:
            # Probably already atomized by a previous conflict
            continue

    
        ## Nullify object labels in affected frames
        # Maybe nullify labels in an extra frame on either side to take care
        # of dependencies?
        atomize_mask = cdata['frame'].isin(conflict_streak_frames)
        
        # Copy a slice for efficiency
        cccdata = cdata.loc[atomize_mask, :].copy()
        cccdata.loc[:, 'object'] = np.nan
        
        # Break streaks in affected frames
        new_streak_start = cdata['streak'].max() + 1
        cccdata.loc[:, 'streak'] = np.arange(
            new_streak_start, new_streak_start + atomize_mask.sum())

        # Calculate streak2object_ser
        streak2object_ser = cccdata[
            ['object', 'streak']].dropna().drop_duplicates(
            'streak').astype(np.int).set_index('streak')[
            'object'].sort_index()        

    
        ## Set up Assigner
        assigner = Assigner(
            classified_data=cccdata, 
            geometry_fab2model=classifier.geometry_fab2model,
            geometry_angle_bins=classifier.geometry_angle_bins, 
            geometry_fab2scaler=classifier.geometry_fab2scaler,
            geometry_model_columns=classifier.geometry_model_columns, 
            verbosity=0,
            streak2object_ser=streak2object_ser, 
            all_known_objects=all_known_objects,
        )
        
        # Run on conflict_streak_frames
        assigner.run(frame_start=conflict_streak_frames[0])
        assigner_result = assigner.classified_data
        
        # Extract the likelihood of each seg under the newly assigned object
        repaired_lik_idxs = np.ravel_multi_index((
            np.where(atomize_mask)[0],
            assigner_result.loc[atomize_mask, 'object'].values.astype(np.int)), 
            geometry_costs_by_seg.shape
        )
        repaired_lik = geometry_costs_by_seg.values.flatten()[
            repaired_lik_idxs]
        repaired_lik_sum = repaired_lik.sum()
        
        # Apply result
        cdata.loc[atomize_mask, 'object'] = assigner_result.loc[
            atomize_mask, 'object'].astype(np.int)

        # Only apply to cdata2 if better
        if repaired_lik_sum >= streak_sum_lks.loc[conflict_streak, 'assigned_lik']:
            better_than_assigned = True
            cdata2.loc[atomize_mask, 'object'] = assigner_result.loc[
                atomize_mask, 'object'].astype(np.int)            
        else:
            better_than_assigned = False
        
        # Measure error rate
        if fold_heldout_results is not None:
            error_rate = (fold_heldout_results != 
                cdata.loc[fold_heldout_results.index, 'object']).mean()
            error_rate2 = (fold_heldout_results != 
                cdata2.loc[fold_heldout_results.index, 'object']).mean()
        else:
            error_rate = None
            error_rate2 = None

        # Store results
        repair_l.append({
            'conflict_streak': conflict_streak,
            'repaired_lik_sum': repaired_lik_sum,
            'repaired_lik_mean': repaired_lik.mean(),
            'error_rate': error_rate,
            'error_rate2': error_rate2,
            'better_than_assigned': better_than_assigned,
        })
        if verbosity >= 1:
            if fold_heldout_results is not None:
                print "%d/%d %f %f" % (len(repair_l), len(conflict_streaks), 
                    error_rate, error_rate2)
            else:
                print "%d/%d" % (len(repair_l), len(conflict_streaks))

    
    ## Store back in classifier
    classifier.repaired_data = cdata
    classifier.repaired_data2 = cdata2
    
    # DataFrame results
    repaired_df = pandas.DataFrame.from_records(repair_l)

    return classifier, repaired_df, streak_sum_lks