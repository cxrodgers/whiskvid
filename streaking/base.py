import numpy as np
import pandas
import numpy as np
import scipy.optimize

def ls_dist(xA_0, yA_0, xB_0, yB_0, xA_1, yA_1, xB_1, yB_1):
    """Distance between two line segments
    
    We identify the range [x0, x1] over which both are defined. Then integrate
    (yB - yA) over that range, and divide by (x1 - x0).
    
    A and B have to be the left side and right side of the line segments
    
    This works out to be: 
    0.5 * (delta_m) * (x1 + x0) + (delta_intercept)
    where delta_m is the difference in slopes and delta_intercept is 
    the difference in intercepts
    """
    m0 = (yB_0 - yA_0) / float(xB_0 - xA_0)
    m1 = (yB_1 - yA_1) / float(xB_1 - xA_1)
    b0 = yB_0 - m0 * xB_0
    b1 = yB_1 - m1 * xB_1

    assert xB_0 > xA_0
    assert xB_1 > xA_1
    x0 = np.max([xA_0, xA_1])
    x1 = np.min([xB_0, xB_1])
    
    if x1 <= x0:
        # no region of overlap
        return np.nan
    
    res = 0.5 * (m1 - m0) * (x1 + x0) + (b1 - b0)
    #~ print "m0=%.2f, m1=%.2f, b0=%.2f, b1=%.2f, x0=%.2f, x1=%.2f, res=%.2f" % (
        #~ m0, m1, b0, b1, x0, x1, res)
    
    return res

def vectorized_ls_dist(df):
    """Distance between two line segments, each a row in `df`.
    
    Returns np.nan if there is no overlap in x between the two
    segments.    
    """
    m0 = (df['fol_y0'] - df['tip_y0']) / (df['fol_x0'] - df['tip_x0'])
    m1 = (df['fol_y1'] - df['tip_y1']) / (df['fol_x1'] - df['tip_x1'])
    b0 = df['fol_y0'] - m0 * df['fol_x0']
    b1 = df['fol_y1'] - m1 * df['fol_x1']

    assert (df['fol_x0'] > df['tip_x0']).all()
    assert (df['fol_x1'] > df['tip_x1']).all()
    
    # The right-most tip is the left side of the integration window
    x0 = df[['tip_x0', 'tip_x1']].max(axis=1) #np.max([df['tip_x0'], df['tip_x1']])
    
    # The left-most follicle is the right side of the integration window
    x1 = df[['fol_x0', 'fol_x1']].min(axis=1) #np.min([df['fol_x0'], df['fol_x1']])
    
    # Mask the rows where there is no overlap at all
    badmask = x1 <= x0
    
    # Compute result
    res = 0.5 * (m1 - m0) * (x1 + x0) + (b1 - b0)
    
    # Apply badmask
    res[badmask] = np.nan
    
    return res

def calculate_center2center_distance_on_merged(merged):
    """Inter-frame distance distribution within- and across-whiskers.
    
    Must have already been merged as desired

    Returns: Series
        distance of each row in `merged`
    """
    # Calculate differences in geometric columns
    d_center_x = merged['center_x1'] - merged['center_x0']
    d_center_y = merged['center_y1'] - merged['center_y0']

    # Norm
    res = np.sqrt(d_center_x ** 2 + d_center_y ** 2)
    res.name = 'd_center'
    
    return res


def score(jres, cres, curated_num2name):
    ## Join jason's results onto curated results
    ares = cres.join(jres, rsuffix='_jason', how='inner')

    # For right now drop the unlabeled ones in the curated dataset
    ares = ares[ares.color_group != -1].copy()


    ## Print general statistics of each dataset
    zobj = zip(['Test', 'Curated', 'Joint'], [jres, cres, ares])
    for setname, label_set in zobj:
        print "%s dataset:\n----" % setname
        unique_labels = label_set['color_group'].value_counts().index.values
        print "%d groups: %s (in order of prevalence)" % (
            len(unique_labels), ','.join(map(str, unique_labels)))
        print "%d rows, of which %d unlabeled (-1)" % (
            len(label_set), np.sum(label_set['color_group'] == -1))
        print


    ## Figure out the mapping between j_labels and c_labels
    # Count confusion matrix
    confusion_matrix = ares.reset_index().pivot_table(
        index='color_group', columns='color_group_jason', values='frame', 
        aggfunc='count').fillna(0).astype(np.int)

    # Assign
    # j_labels will be Jason's labels for curated classes in sorted order
    # j_labels_idx is an index into the actual labels, which 
    # are on the pandas columns
    c_labels_idx, j_labels_idx = scipy.optimize.linear_sum_assignment(
        -confusion_matrix.values)
    j_labels = confusion_matrix.columns.values[j_labels_idx]
    c_labels = confusion_matrix.index.values[c_labels_idx]
    unused_j_labels = np.array([j_label for j_label in confusion_matrix.columns 
        if j_label not in j_labels])

    # Sort the columns of the confusion matrix to match c_labels
    new_column_order = list(j_labels) + list(unused_j_labels)
    confusion_matrix = confusion_matrix.loc[:, new_column_order]

    # Print results
    print "Assignments (C>J):\n%s" % ('\n'.join([
        '%s (%s) > %s' % (c_label, curated_num2name[c_label], j_label) 
        for c_label, j_label in zip(c_labels, j_labels)]))
    print "Unassigned labels: %s" % (' '.join(map(str, unused_j_labels)))
    print


    ## Calculate performance
    print "Confusion matrix: "
    relabeled_confusion_matrix = confusion_matrix.join(curated_num2name).set_index(
        'whisker')
    print relabeled_confusion_matrix
    print

    return relabeled_confusion_matrix
    
    
def parse_confusion_into_sensitivity_and_specificity(relabeled_confusion_matrix):
    ## Metrics
    sensitivity = (relabeled_confusion_matrix.values.diagonal() / 
        relabeled_confusion_matrix.sum(1))

    # This will fail if there are unused j_labels
    specificity = (relabeled_confusion_matrix.values.diagonal() / 
        relabeled_confusion_matrix.sum(0))
    specificity.index = sensitivity.index
    metrics = pandas.concat([sensitivity, specificity], 
        axis=1, verify_integrity=True,
        keys=['sensitivity', 'specificity'])
    print metrics
    
    return metrics
    
def get_confusion_results_and_curated(classified_data, curated_df, 
    curated_num2name):
    """Return confusion matrix and object/color_group mapping
    
    classified_data : results of streaking algorithm
    curated_df : curated data
    curated_num2name : color_group to whisker name
    
    Returns: confusion_matrix, whisker_name2object_id
        confusion_matrix : DataFrame with whisker names on rows and
            identified objects on columns. Values are the number of times
            that whisker was identified as that object. It will be sorted
            according to the hungarian algorithm to maximize the diagonal.
        whisker_name2object_id : Series along the diagonal of confusion_matrix
            The index is whisker name and the values are corresponding object
    """
    # Make labels consistent
    reformatted_results = classified_data[
        ['frame', 'seg', 'object']].dropna().astype(
        np.int).set_index(['frame', 'seg']).rename(
        columns={'object': 'color_group'})

    # Confusion matrix
    sconfmat = score(
        reformatted_results, 
        curated_df.set_index(['frame', 'seg']), 
        curated_num2name,
    )

    # Use rows and columns of confusion matrix to identify correspondence
    whisker_name2object_id = pandas.Series(
        sconfmat.columns.values,
        sconfmat.index.values, 
        name='object',
    )
    
    return sconfmat, whisker_name2object_id    

def parse_classifier_results(classifier, classified_data):
    """Parse out results and metrics on each step
    
    classifier : object that was used
    classified_data : With correct_object added
    
    Returns: dict with keys:
        votes : indexed by frame and streak
        stats : indexed by frame
        metrics_of_choice : indexed by frame
    """
    def deal_with_votes(vdf, chosen_object, correct_object):
        """Merge the votes of each metric with the actual and correct choice
        
        Returns: big_votes_df
            indexed by frame and streak
            columns: MultiIndex (voted, chosen, correct) on each metric
        """
        # Index chosen object like vdf
        comp_obj = chosen_object.loc[vdf.index]
        vote_chosen_df = pandas.DataFrame(
            vdf.values == comp_obj.values[:, None],
            index=vdf.index, columns=vdf.columns
        )
        
        # Index correct object like vdf
        comp_obj = correct_object.loc[vdf.index]
        vote_correct_df = pandas.DataFrame(
            vdf.values == comp_obj.values[:, None],
            index=vdf.index, columns=vdf.columns
        )        
        
        # Concat with vdf
        big_votes_df = pandas.concat([vdf, vote_chosen_df, vote_correct_df],
            axis=1, keys=['voted', 'chosen', 'correct'], verify_integrity=True)        
        
        return big_votes_df
    
    def deal_with_perf_rec(prl):
        """Parse results from perf_rec_l
        
        Returns: stats, metrics
            Indexed by frame
        """
        # Also parse rec_l
        perf_rec_df = pandas.DataFrame.from_records(prl).set_index(
            'frame').sort_index()

        # Parse the metrics of the chosen alignment on each frame
        raw_metrics_of_best = pandas.concat(
            perf_rec_df.pop('metrics_of_best').values, 
            keys=perf_rec_df.index).unstack().sort_index()
        std_metrics_of_best = pandas.concat(
            perf_rec_df.pop('std_metrics_of_best').values, 
            keys=perf_rec_df.index).unstack().sort_index()
        metrics_of_best = pandas.concat([raw_metrics_of_best, std_metrics_of_best],
            keys=['raw', 'std'], axis=1, verify_integrity=True)        
        
        return perf_rec_df, metrics_of_best

    # Parse columns
    cd2 = classified_data.set_index(['frame', 'streak'])
    chosen_object = cd2['object']
    correct_object = cd2['correct_object']
    correct_color_group = cd2['color_group']

    # Get votes
    votes = deal_with_votes(classifier.get_votes_df(), 
        chosen_object, correct_object)
    
    # Get oracular votes
    # This isn't quite right, not sure how to extract the choice that
    # the oracular version would have chosen
    ovotes = deal_with_votes(classifier.get_oracular_votes_df(), 
        correct_color_group, correct_color_group)

    # Get stats and metrics
    stats, metrics = deal_with_perf_rec(classifier.perf_rec_l)
    
    # Get oracular stats and metrics
    ostats, ometrics = deal_with_perf_rec(classifier.oracular_perf_rec_l)
    
    # Concat
    res_votes = pandas.concat([votes, ovotes], keys=['online', 'oracular'], 
        axis=1, verify_integrity=True)
    res_metrics = pandas.concat([metrics, ometrics], keys=['online', 'oracular'], 
        axis=1, verify_integrity=True)        
    res_stats = pandas.concat([stats, ostats], keys=['online', 'oracular'], 
        axis=1, verify_integrity=True)        
    
    return {
        'votes': res_votes,
        'metrics_of_choice': res_metrics,
        'stats': res_stats,
    }
    