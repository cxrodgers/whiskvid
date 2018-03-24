import numpy as np
import pandas
import sklearn.linear_model
import sklearn.naive_bayes


def update_geometry(mwe, geometry_model_columns, key='object'):
    """Model the geometry of known objects in mwe
    
    mwe : mwe
    geometry_model_columns : list of columns in mwe to incorporate into
        the geometric model
    key : the columns containing the known labels
    
    # calculate distrs over fol_x, fol_y, tip_x, tip_y for each object
    f, ax = plt.subplots()
    objects = map(int, list(mwe.object.dropna().unique()))
    colorbar = my.plot.generate_colorbar(len(objects))
    for n_object, (object, subdata) in enumerate(mwe.groupby('object')):
        color = colorbar[n_object]
        ax.plot(subdata['tip_x'], subdata['tip_y'], ',', color=color, ms=3)
        ax.plot(subdata['fol_x'], subdata['fol_y'], ',', color=color, ms=3)
    whiskvid.plotting.set_axis_for_image(ax, 800, 800)
    for n_color, color in enumerate(colorbar):
        f.text(.9, .9 - n_color * .05, str(n_color), color=color)
    plt.show()
    
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

    # Data
    model_mask = ~mwe[key].isnull()
    input_data = mwe.loc[model_mask, geometry_model_columns].values
    output_data = mwe.loc[model_mask, key].values.astype(np.int)

    # Scale to avoid hanging
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(input_data)
    scaled_data = scaler.transform(input_data)

    # Fit from geometry to object label
    model.fit(scaled_data, output_data)
    
    return model, scaler

def measure_geometry_costs(mwe, model, next_frame_streaks, alignments,
    geometry_model_columns, geometry_scaler):
    """Measure geometry costs
    
    Returns: geometry_costs, geometry_costs_by_alignment
    """
    geometry_costs_l = []
    for streak_to_fit in next_frame_streaks:
        # Data for this streak
        stf_data = mwe.loc[mwe['streak'] == streak_to_fit, 
            geometry_model_columns].values
        
        # Scale
        scaled_stf_data = geometry_scaler.transform(stf_data)
        
        # Predict the probability of the streak data
        # This has shape (len(stf_data), len(model.classes_))
        stf_log_proba = model.predict_log_proba(scaled_stf_data)
        
        # Mean over every frame in the observed streak
        mean_stf_log_proba = stf_log_proba.mean(0)

        # Store
        geometry_costs_l.append(mean_stf_log_proba)
    
    # DataFrame it
    geometry_costs = pandas.DataFrame(np.array(geometry_costs_l),
        index=next_frame_streaks, columns=model.classes_)
    geometry_costs.index.name = 'streak'
    geometry_costs.columns.name = 'object'

    # Calculate a geometric cost for every alignment
    # This is just the sum of the costs of all included assignments
    geometry_costs_by_alignment = pandas.Series([
        np.sum([geometry_costs.loc[streak, obj] for obj, streak in alignment]) 
        for alignment in alignments],
        index=range(len(alignments)))
    geometry_costs_by_alignment.name = 'geometry'
    
    return geometry_costs, geometry_costs_by_alignment
