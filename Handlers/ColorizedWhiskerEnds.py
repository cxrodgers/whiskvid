import pandas
import numpy as np
from base import CalculationHandler
import BeWatch

class ColorizedWhiskerEndsHandler(CalculationHandler):
    """Whisker ends after colorizing"""
    _db_field_path = 'colorized_whisker_ends_filename'
    _name = 'colorized_whisker_ends'
    
    def triggered_on_rwin(self, relative_time_bins=None):
        """Return angle of each whisker triggered on rwin"""
        # Load colorized whisker ends
        cwe = self.load_data()

        # Which whisker is which color
        whisker_colors = (
            self.video_session._django_object.whisker_colors.split())
        whisker_colors_df = pandas.Series(whisker_colors, 
            index=range(1, len(whisker_colors) + 1))

        # Get sync
        v2b_fit = self.video_session.fit_v2b

        # Get trial matrix
        bsession = self.video_session.bsession_name
        tm = BeWatch.db.get_trial_matrix(bsession, True)

        relative_time_bins = None
        if relative_time_bins is None:
            relative_time_bins = np.arange(-3.5, 5, .05)

        # Get the trigger times
        rwin_open_times_by_trial = tm['rwin_time']
        trigger_times = rwin_open_times_by_trial.dropna()

        # Iterate over colors
        color2twa = {}
        for color, whisker_name in whisker_colors_df.iteritems():
            ## Extract data just for this whisker
            angle_by_frame = cwe.loc[cwe.color_group == color, 
                ['frame', 'angle']].set_index('frame')['angle']

            # time of each row
            angle_vtime = angle_by_frame.index.values / 30.
            angle_btime = np.polyval(v2b_fit, angle_vtime)

            # Index the angle based on the btime
            angle_by_btime = pandas.Series(index=angle_btime, 
                data=angle_by_frame.values)
            angle_by_btime.index.name = 'btime'

            ## Interpolate angle_by_btime at the new time bins that we want
            # Get absolute time bins
            absolute_time_bins_l = []
            for trial, trigger_time in trigger_times.iteritems():
                # Get time bins relative to trigger
                absolute_time_bins = relative_time_bins + trigger_time
                absolute_time_bins_l.append(absolute_time_bins)

            # Drop the ones before and after data
            # By default pandas interpolate fills forward but not backward
            absolute_time_bins_a = np.concatenate(absolute_time_bins_l)
            absolute_time_bins_a_in_range = absolute_time_bins_a[
                (absolute_time_bins_a < angle_by_btime.index.values.max()) &
                (absolute_time_bins_a > angle_by_btime.index.values.min())
            ].copy()

            # Make bigger index with positions for each of the desired time bins
            # Ensure it doesn't contain duplicates
            new_index = (angle_by_btime.index | 
                pandas.Index(absolute_time_bins_a_in_range))
            new_index = new_index.drop_duplicates()

            # Interpolate
            resampled_session = angle_by_btime.reindex(new_index).interpolate(
                'index')
            assert not np.any(resampled_session.isnull())

            ## Extract interpolated times for each trial
            # Take interpolated values at each of the absolute time bins
            # Will be NaN before and after the data
            interpolated = resampled_session.ix[absolute_time_bins_a]
            assert interpolated.shape == absolute_time_bins_a.shape

            ## Reshape
            # One column per trial
            twa = pandas.DataFrame(
                interpolated.values.reshape(
                    (len(trigger_times), len(relative_time_bins))).T,
                index=relative_time_bins, columns=trigger_times.index
            )
            twa.index.name = 'time'
            twa.columns.name = 'trial'

            ## Store
            color2twa[whisker_name] = twa

        twa_by_whisker = pandas.concat(color2twa, axis=1, names=('whisker',))
        twa_by_whisker = twa_by_whisker.swaplevel(axis=1).sort_index(
            axis=1).dropna(axis=1)        
        
        return twa_by_whisker