from builtins import map
from builtins import str
import numpy as np
import whiskvid
import pandas
import my.plot
import matplotlib.pyplot as plt

def init_animation(data3, object2line, f, ax):
    # Ax
    whiskvid.plotting.set_axis_for_image(ax, 800, 800)
    
    # List of known objects
    objects = list(map(int, list(data3.object.dropna().unique())))
    
    # Sort them
    objects = sorted(objects)
    
    # Color bar
    colorbar = my.plot.generate_colorbar(len(objects))
    
    # Label known objects
    for n_color, color in enumerate(colorbar):
        f.text(.9, .9 - n_color * .05, str(objects[n_color]), color=color)
    
    # handles for lines
    for object in objects:
        color = colorbar[objects.index(object)]
        object2line[object], = ax.plot([np.nan] * 2, [np.nan] * 2, '-', color=color)

def update_animation(data3, frame, object2line, f, ax, unclassified_lines):
    data3_grouped_by_frame = data3.groupby('frame')
    
    # delete old
    for line in unclassified_lines:
        line.remove()
    
    # unclassified_lines = []
    del unclassified_lines[:]
    
    for line in list(object2line.values()):
        line.set_xdata([np.nan, np.nan])
    
    # get data and plot each object
    nothing_to_plot = False
    try:
        frame_data = data3_grouped_by_frame.get_group(frame)
    except KeyError:
        nothing_to_plot = True
    
    if not nothing_to_plot:
        for idx in frame_data.index:
            # get object
            object_id = frame_data.loc[idx, 'object']
            
            # if unclassified plot as black
            if np.isnan(object_id):
                unclassified_line, = ax.plot(
                    frame_data.loc[idx, ['fol_x', 'tip_x']],
                    frame_data.loc[idx, ['fol_y', 'tip_y']],
                    ':', color='k'
                )
                unclassified_lines.append(unclassified_line)
            else:
                line = object2line[frame_data.loc[idx, 'object']]
                line.set_xdata(frame_data.loc[idx, ['fol_x', 'tip_x']])
                line.set_ydata(frame_data.loc[idx, ['fol_y', 'tip_y']])
    
    ax.set_title(str(frame))
    plt.pause(.01)
    plt.show()
