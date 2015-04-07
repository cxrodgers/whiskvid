"""For dealing with db of high-speed video stuff"""
import numpy as np
import pandas
import ArduFSM
import my

def load_everything_from_session(session, db):
    """Load all data from specified row of db
    
    TODO: just specify session, and automatically read db here
    """
    row = db.ix[session]

    # Fit from previous file
    b2v_fit = np.loadtxt(os.path.join(row['root_dir'], session, row['fit']))
    v2b_fit = my.misc.invert_linear_poly(b2v_fit)

    # Get behavior df
    trial_matrix = ArduFSM.TrialMatrix.make_trial_matrix_from_file(
        os.path.join(row['root_dir'], session, row['bfile']))

    # Get tips and contacts
    tac = pandas.read_pickle(os.path.join(row['root_dir'], session, row['tac']))
    
    return {'b2v_fit': b2v_fit, 'v2b_fit': v2b_fit, 
        'trial_matrix': trial_matrix, 'tac': tac}

def add_trials_to_tac(tac, v2b_fit, trial_matrix, drop_late_contacts=False):
    """Add the trial numbers to tac and return it
    
    trial_matrix should already have "choice_time" column from BeWatch.misc
    
    Also adds "vtime" (in the spurious 30fps timebase)
    and "btime" (using fit)
    and "t_wrt_choice" (using choice_time)
    
    If `drop_late_contacts`: drops every contact that occurred after choice
    """
    # "vtime" is in the spurious 30fps timebase
    # the fits take this into account
    tac['vtime'] = tac['frame'] / 30.
    tac['btime'] = np.polyval(v2b_fit, tac['vtime'].values)

    # Add trial info to tac
    tac['trial'] = trial_matrix.index[
        np.searchsorted(trial_matrix['start_time'].values, 
            tac['btime'].values) - 1]    

    # Add rewside and outcome to tac
    tac = tac.join(trial_matrix[['rewside', 'outcome', 'choice_time']], 
        on='trial')
    tac['t_wrt_choice'] = tac['btime'] - tac['choice_time']
    
    if drop_late_contacts:
        tac = tac[tac.t_wrt_choice < 0]

    return tac
    