import numpy as np
import pandas
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.calibration
import sklearn.model_selection


def update_geometry(mwe, geometry_model_columns, key='object', model_typ='nb',
    calibrate=True, multi_angle_bins=11):
    """Model the geometry of known objects in mwe
    
    mwe : mwe
    geometry_model_columns : list of columns in mwe to incorporate into
        the geometric model
    key : the columns containing the known labels
    model_typ : 'nb' for Naive Bayes or 'sgd' for SGDClassifier
    
    Returns: model, scaler
        This is an SGD model that takes geometric inputs, and predicts
        object identity.
    """
    
    #~ # Build SVM model of the geometry
    #~ model = sklearn.svm.SVC(kernel='linear', class_weight='balanced', 
        #~ probability=True, decision_function_shape='ovr')

    if model_typ == 'sgd':
        # Build SGD model of the geometry
        # This permits partial_fit, but seems to be enough faster anyway not to
        # need it
        model = sklearn.linear_model.SGDClassifier(
            fit_intercept=False, # data will be scaled
            max_iter=1000, tol=1e-3, # to avoid warnings
            loss='log', # to permit probability estimate
            class_weight='balanced',
        )
    elif model_typ == 'nb':
        model = sklearn.naive_bayes.GaussianNB()
    else:
        1/0


    ## Only fit known objects
    model_mask = ~mwe[key].isnull()
    mwe2 = mwe.loc[model_mask].copy()
    

    ## Determine angle bins
    frame2angle = mwe2.groupby('frame')['frangle'].mean()
    angle_bins = frame2angle.quantile(np.linspace(0, 1, multi_angle_bins)).values
    angle_bins[0] = -np.inf
    angle_bins[-1] = np.inf

    ## Bin data by angle
    mwe2['frangle_bin'] = pandas.cut(mwe2['frangle'], angle_bins, 
        labels=False).astype(np.int)


    ## Fit separate models for each angle bin
    fab2model = {}
    fab2scaler = {}
    for fab, sub_mwe in mwe2.groupby('frangle_bin'):
        # Get input and output data
        input_data = sub_mwe.loc[:, geometry_model_columns].values
        output_data = sub_mwe.loc[:, key].values.astype(np.int)

        # Scale to avoid hanging
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(input_data)
        scaled_data = scaler.transform(input_data)
        
        if calibrate:
            # Reserve some data for calibration
            cv = sklearn.model_selection.StratifiedKFold(n_splits=3, shuffle=True)
            
            # Fit with calibration
            model = sklearn.calibration.CalibratedClassifierCV(
                sklearn.naive_bayes.GaussianNB(), cv=cv, method='isotonic')
            model.fit(scaled_data, output_data)

        else:
            # Uncalibrated
            model.fit(scaled_data, output_data)
        
        # Store
        fab2model[fab] = model
        fab2scaler[fab] = scaler
    
    return angle_bins, fab2model, fab2scaler

def measure_geometry_costs(mwe, angle_bins, fab2model, fab2scaler, 
    next_frame_streaks, 
    geometry_model_columns, cost_floor=-6):
    """Measure geometry costs
    
    mwe : the data
    model, geometry_scaler : from update_geometry
    next_frame_streaks : streaks in next frame to consider
    geometry_model_columns : columns from mwe to feed to model
    
    This computes the cost of assigning each streak to each possible
    object. Cost is assessed using model.predict_log_proba. The cost of
    each alignment is the sum of all its contained assignments, after
    flooring the cost of every alignment at `cost_floor`.
    
    Returns: geometry_costs, geometry_costs_by_alignment
        geometry_costs : DataFrame
            The index is every streak in `next_frame_streaks`, and the
            columns are every object in the classifier. The values are
            the costs of making that assignment.
    """
    # Insert frangle column
    mwe['frangle_bin'] = pandas.cut(mwe['frangle'], angle_bins, 
        labels=False).astype(np.int)    
    
    # Process data of corresponding streaks
    mwe2 = mwe.loc[mwe['streak'].isin(next_frame_streaks)]
    
    # Extract all objects, in case they don't come up here
    all_classes = np.sort(mwe['object'].dropna().astype(np.int).unique())
    
    # Iterate over frangles
    geometry_costs_l = []
    for fab, sub_mwe in mwe2.groupby('frangle_bin'):
        # Get corresponding model
        model = fab2model[fab]
        scaler = fab2scaler[fab]
        
        # Fit
        scaled_stf_data = scaler.transform(
            sub_mwe[geometry_model_columns].values)
        proba = model.predict_proba(scaled_stf_data)
        
        # Store
        geometry_costs_l.append(pandas.DataFrame(proba, 
            index=sub_mwe.index, columns=model.classes_))
    
    # Concat
    geometry_costs_by_frame = pandas.concat(geometry_costs_l, axis=0)
    geometry_costs_by_frame['streak'] = mwe2.loc[geometry_costs_by_frame.index, 'streak'].values
    geometry_costs_by_frame = geometry_costs_by_frame.reset_index().set_index(
        ['streak', 'index'])
    geometry_costs_by_frame = np.log10(1e-300 + geometry_costs_by_frame)
    geometry_costs_by_frame.columns.name = 'object'

    # Reindex by all classes in case it never came up
    geometry_costs_by_frame = geometry_costs_by_frame.reindex(all_classes, 
        axis=1)
    
    # Fillna in case no data
    geometry_costs_by_frame = geometry_costs_by_frame.fillna(-300)

    # Mean within streak
    geometry_costs = geometry_costs_by_frame.mean(level=0)

    # Floor the cost
    geometry_costs2 = geometry_costs.copy()
    if cost_floor is not None:
        geometry_costs2[geometry_costs2 < cost_floor] = cost_floor
    
    return geometry_costs

