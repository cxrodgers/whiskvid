"""For plotting results from this module"""

import numpy as np
import whiskvid
import my
import my.plot
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.mlab import prctile



## Plotting functions
def plot_transparent_histogram(arr, ax, frame_width, frame_height, 
    upper_prctile_clim=95, cmap=plt.cm.gray_r):
    """Imshow a histogram with zero values transparent
    
    H_tip : single 2d array to plot. Should be non-negative
    frame_width, frame_height : passed to imshow to get the data limits right
    
    All zero bins will be transparent. The image color limits will be 
    set to the 95th percentile of non-zero values.
    """
    # Determine the transparent threshold and upper clim
    vals = arr.flatten()
    vals = vals[vals > 0]
    transparent_threshold = vals.min()
    clim_upper = prctile(vals, upper_prctile_clim)    
    
    # Mask the data to make zero bins transparent
    # We use .99 to avoid floating point comparison problems
    masked_data = np.ma.masked_where(arr < transparent_threshold * .99, arr)
    
    # Plot
    im = my.plot.imshow(
        masked_data,
        ax=ax,
        xd_range=(0, frame_width), yd_range=(0, frame_height),
        axis_call='image', cmap=cmap, skip_coerce=True,
    )
    
    # Set the clim to go from 0 to upper
    im.set_clim((0, clim_upper))
    
    return im

def plot_contact_locations_by_whisker(ccs, color2whisker, ax, 
    whisker_color_order=None, **kwargs):
    """Plot the individual locations of each contact for each whisker"""
    # Default plot kwargs
    plot_kwargs = {'marker': '.', 'ms': 4, 'alpha': .2, 'ls': 'none'}
    plot_kwargs.update(**kwargs)
    
    # Whisker colors
    if whisker_color_order is None:
        whisker_color_order = whiskvid.WHISKER_COLOR_ORDER_K
    
    # Plot the contact locations
    for color, whisker in color2whisker.items():
        color_s = whisker_color_order[color]
        sub_ccs = ccs[ccs.color == color]
        ax.plot(sub_ccs.tip_x, sub_ccs.tip_y, color=color_s,
            **plot_kwargs)

def plot_contact_locations_by_rewside(ccs, ax, **kwargs):
    """Plot the individual locations of each contact for each rewside"""
    # Default plot kwargs
    plot_kwargs = {'marker': '.', 'ms': 4, 'alpha': .2, 'ls': 'none'}
    plot_kwargs.update(**kwargs)
    
    # Plot the contact locations
    rewside2color_s = {'right': 'r', 'left': 'b'}
    for rewside, color_s in rewside2color_s.items():
        sub_ccs = ccs[ccs.rewside == rewside]
        ax.plot(sub_ccs.tip_x, sub_ccs.tip_y, color=color_s,
            **plot_kwargs)

def plot_whisker_ends_as_points(cwe, color2whisker, ax, typ='tip',
    whisker_color_order=None, n_points=100, **kwargs):
    """Plot a subset of tip or follicles as colored points
    
    typ: 'tip' or 'fol'
    """
    # Default plot kwargs
    plot_kwargs = {'marker': '.', 'ms': 4, 'alpha': .2, 'ls': 'none'}
    plot_kwargs.update(**kwargs)
    
    # Whisker colors
    if whisker_color_order is None:
        whisker_color_order = whiskvid.WHISKER_COLOR_ORDER_K
    
    # Separate images for all whisker fols
    for color, whisker in color2whisker.items():
        sub_cwe = cwe[cwe.color_group == color]
        color_s = whisker_color_order[color]
        ax.plot(
            my.misc.take_equally_spaced(sub_cwe[typ + '_x'], n_points),
            my.misc.take_equally_spaced(sub_cwe[typ + '_y'], n_points),
            color=color_s, **plot_kwargs
        )    

def set_axis_for_image(ax, frame_width, frame_height,
    half_bin_offset_x=None, half_bin_offset_y=None):
    """Set axis defaults for data from video frames
    
    Grid with 100 increments instead of ticklabels
    Limits set to frame size (though perhaps should be a half-bin
    greater than the frame size)
    
    half_bin_offset_x, half_bin_offset_y : increase xlim and ylim by
        this much on each side
    """
    # Axis limits
    ax.axis('image')
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.grid()
    
    if half_bin_offset_x is None:
        ax.set_xlim((0, frame_width))
    else:
        ax.set_xlim((-half_bin_offset_x, frame_width + half_bin_offset_x))

    if half_bin_offset_y is None:
        ax.set_ylim((frame_height, 0))
    else:
        ax.set_ylim((frame_height + half_bin_offset_y, -half_bin_offset_y))
