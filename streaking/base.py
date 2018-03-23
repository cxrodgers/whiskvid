import numpy as np
import WhiskiWrap
import whiskvid
import runner.models
import pandas
import numpy as np
import my.plot
import matplotlib.pyplot as plt
import sklearn.svm, sklearn.preprocessing
import itertools

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
    
    
    