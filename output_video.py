"""Module for creating output videos with various overlays"""
from __future__ import print_function
from __future__ import division
from builtins import zip

import pandas
import os
import subprocess
import matplotlib.pyplot as plt
import my.plot
import numpy as np
import whiskvid
import tables

class OutOfFrames(BaseException):
    pass

## Frame updating function
def frame_update(ax, nframe, frame, whisker_handles, contacts_table,
    post_contact_linger, joints, edge_a,
    im2, edge_a_obj, contacts_handle,
    d_spatial, d_temporal, 
    whisker_lw=1, whisker_marker=None, whisker_marker_size=8):
    """Helper function to plot each frame.
    
    Typically this is called by write_video_with_overlays.
    
    nframe : number of frame
        This is used to determine which whiskers and which contacts to plot
    
    frame : the image data
    
    whisker_handles : handles to whiskers lines that will be deleted
    
    contacts_table : contacts to plot
        if None, no contacts are plotted
    
    contacts_handle : handle or None
        If not None, will update this graphics handle with the position
        of the current contacts
    
    joints : DataFrame or None
        If not None, then these joints are plotted as the whiskers.
    
    whisker_handles : if whiskers are plotted, all of these handles
        are first deleted. The new whisker handles are stored in it.
    
    whisker_lw : line width of whiskers
    
    edge_a : edge to plot
        if None, no edge is plotted.
    
    edge_a_obj : if an edge is plotted, this handle is used
    
    Returns: whisker_handles
        These are returned so that they can be deleted next time
    """
    # Get the frame
    im2.set_data(frame[::d_spatial, ::d_spatial])
    
    # Update the edge plotting object
    if edge_a is not None:
        edge_a_frame = edge_a[nframe]
        if edge_a_frame is not None:
            edge_a_obj.set_xdata(edge_a_frame[:, 1])
            edge_a_obj.set_ydata(edge_a_frame[:, 0])
        else:
            edge_a_obj.set_xdata([np.nan])
            edge_a_obj.set_ydata([np.nan])
    
    # Update the contacts plotting object
    if contacts_table is not None:
        # Grab the contacts from frames (nframe - post_contact_linger, nframe]
        subtac = contacts_table[
            (contacts_table.frame <= nframe) & 
            (contacts_table.frame > nframe - post_contact_linger)
            ]
        
        # Use only the first contact_positions_l so they are colored the same
        contacts_handle.set_xdata(subtac['tip_x'])
        contacts_handle.set_ydata(subtac['tip_y'])
    
    # Get the whiskers for this frame
    if joints is not None:
        # Remove old whiskers
        for handle in whisker_handles:
            handle.remove()
        whisker_handles = []            
        
        # Select out whiskers from this frame
        try:
            frame_joints = joints.loc[nframe]
        except KeyError:
            frame_joints = None
        
        if frame_joints is not None:
            # Plot each
            for whisker in frame_joints.index:
                # Get the color of the whisker
                try:
                    color = whiskvid.GLOBAL_WHISKER2COLOR[whisker]
                except KeyError:
                    color = 'yellow'

                # Plot the whisker
                whisker_joints = frame_joints.loc[whisker].unstack().T
                line, = ax.plot(
                    whisker_joints['c'].values,
                    whisker_joints['r'].values,
                    color=color, lw=whisker_lw, marker=whisker_marker, 
                    mfc='none', ms=whisker_marker_size)
                
                # Store the handle
                whisker_handles.append(line)
    
    return whisker_handles

def write_video_with_overlays(output_filename, 
    input_reader, input_width, input_height, verbose=True,
    whiskers_filename=None, whiskers_table=None,
    whiskers_file_handle=None,
    edges_filename=None, contacts_filename=None,
    contacts_table=None,
    **kwargs):
    """Creating a video overlaid with whiskers, contacts, etc.
    
    This is a wrapper function that loads all the data from disk.
    The actual plotting is done by
        write_video_with_overlays_from_data
    See documentation there for all other parameters.
    
    output_filename, input, input_width, input_height :
        See write_video_with_overlays_from_data
    
    whiskers_filename : name of HDF5 table containing whiskers
        If whiskers_filename is None, then you can provide whiskers_table
        AND whiskers_file_handle explicitly.
    edges_filename : name of file containing edges
    contacts_filename : HDF5 file containing contact info    
    contacts_table : pre-loaded or pre-calculated contacts table
        If contacts_table is not None, then this contacts table is used.
        Otherwise, if contacts_filename is not None, then load it.
        Otherwise, do not use any contacts info.
    """
    ## Load the data
    # Load whiskers
    if whiskers_filename is not None:
        if verbose:
            print("loading whiskers")
        # Need this to plot the whole whisker
        whiskers_file_handle = tables.open_file(whiskers_filename)
        
        # Could also use get_whisker_ends_hdf5 because it will switch tip
        # and foll
        whiskers_table = pandas.DataFrame.from_records(
            whiskers_file_handle.root.summary.read())
    
    # Load contacts
    if contacts_table is None:
        if contacts_filename is not None:
            if verbose:
                print("loading contacts")
            contacts_table = pandas.read_pickle(contacts_filename)
        else:
            contacts_table = None
    
    # Load edges
    if edges_filename is not None:
        if verbose:
            print("loading edges")
        
        try:
            edge_a = np.load(edges_filename, allow_pickle=True)
        except UnicodeError:
            # Python 2/3 problem
            # https://stackoverflow.com/questions/38316283/trouble-using-numpy-load/38316603
            edge_a = np.load(edges_filename, allow_pickle=True, encoding='latin1')
            
    else:
        edge_a = None
    
    write_video_with_overlays_from_data(
        output_filename,
        input_reader, input_width, input_height,
        verbose=True,
        whiskers_table=whiskers_table,
        whiskers_file_handle=whiskers_file_handle,
        contacts_table=contacts_table,
        edge_a=edge_a,
        **kwargs)


def write_video_with_overlays_from_data(output_filename, 
    input_reader, input_width, input_height,
    verbose=True,
    frame_triggers=None, trigger_dstart=-250, trigger_dstop=50,
    plot_trial_numbers=True,
    d_temporal=5, d_spatial=1,
    dpi=50, output_fps=30,
    input_video_alpha=1,
    whiskers_table=None, whiskers_file_handle=None, joints=None,
    edge_a=None, edge_alpha=1, typical_edges_hist2d=None, edge_lw=.5,
    contacts_table=None, post_contact_linger=50, contact_ms=8,
    write_stderr_to_screen=True,
    input_frame_offset=0,
    get_extra_text=None,
    text_size=10,
    contact_color='yellow',
    also_plot_traces=False, trace_data_x=None, trace_data_y=None,
    trace_data_kwargs=None,
    ffmpeg_writer_kwargs=None,
    f=None, ax=None,
    func_update_figure=None,
    whisker_lw=1, whisker_marker=None,
    whisker_marker_size=8,
    ):
    """Creating a video overlaid with whiskers, contacts, etc.
    
    The overall dataflow is this:
    1. Load chunks of frames from the input
    2. One by one, plot the frame with matplotlib. Overlay whiskers, edges,
        contacts, whatever.
    3. Dump the frame to an ffmpeg writer.
    
    # Input and output
    output_filename : file to create
    input_reader : PFReader or input video
    
    # Timing and spatial parameters
    frame_triggers : Only plot frames within (trigger_dstart, trigger_dstop)
        of a value in this array.
    trigger_dstart, trigger_dstop : number of frames
    d_temporal : Save time by plotting every Nth frame
    d_spatial : Save time by spatially undersampling the image
        The bottleneck is typically plotting the raw image in matplotlib
    
    # Video parameters
    dpi : The output video will always be pixel by pixel the same as the
        input (keeping d_spatial in mind). But this dpi value affects font
        and marker size.
    output_fps : set the frame rate of the output video (ffmpeg -r)
    input_video_alpha : alpha of image
    input_frame_offset : If you already seeked this many frames in the
        input_reader. Thus, now we know that the first frame to be read is
        actually frame `input_frame_offset` in the source (and thus, in
        the edge_a, contacts_table, etc.). This is the only parameter you
        need to adjust in this case, not frame_triggers or anything else.
    ffmpeg_writer_kwargs : other parameters for FFmpegWriter
    
    # Other sources of input
    edge_alpha : alpha of edge
    post_contact_linger : How long to leave the contact displayed    
        This is the total duration, so 0 will display nothing, and 1 is minimal.
    
    # Misc
    get_extra_text : if not None, should be a function that accepts a frame
        number and returns some text to add to the display. This is a 
        "real" frame number after accounting for any offset.
    text_size : size of the text
    contact_colors : list of color specs to use
    func_update_figure : optional, function that takes the frame number
        as input and updates the figure
    """
    # We need FFmpegWriter
    # Probably that object should be moved to my.video
    # Or maybe a new repo ffmpeg_tricks
    import WhiskiWrap

    # Parse the arguments
    frame_triggers = np.asarray(frame_triggers).astype(np.int)
    announced_frame_trigger = 0
    input_width = int(input_width)
    input_height = int(input_height)

    if ffmpeg_writer_kwargs is None:
        ffmpeg_writer_kwargs = {}

    ## Set up the graphical handles
    if verbose:
        print("setting up handles")

    if ax is None:
        # Create a figure with an image that fills it
        # We want the figsize to be in inches, so divide by dpi
        # And we want one invisible axis containing an image that fills the whole figure
        figsize = (input_width / float(dpi), input_height / float(dpi))
        f = plt.figure(frameon=False, dpi=(dpi / d_spatial), figsize=figsize)
        ax = f.add_axes([0, 0, 1, 1])
        ax.axis('off')
    
        # This return results in pixels, so should be the same as input width
        # and height. If not, probably rounding error above
        canvas_width, canvas_height = f.canvas.get_width_height()
        if (
            (input_width / d_spatial != canvas_width) or
            (input_height / d_spatial != canvas_height)
            ):
            raise ValueError("canvas size is not the same as input size")        
    else:
        assert f is not None
        
        # This is used later in creating the writer
        canvas_width, canvas_height = f.canvas.get_width_height()

    # Plot typical edge images as static alpha
    if typical_edges_hist2d is not None:
        im1 = my.plot.imshow(typical_edges_hist2d, ax=ax, axis_call='image',
            extent=(0, input_width, input_height, 0), cmap=plt.cm.gray)
        im1.set_alpha(edge_alpha)

    # Plot input video frames
    in_image = np.zeros((input_height, input_width))
    im2 = my.plot.imshow(in_image[::d_spatial, ::d_spatial], ax=ax, 
        axis_call='image', cmap=plt.cm.gray, 
        extent=(0, input_width, input_height, 0))
    im2.set_alpha(input_video_alpha)
    im2.set_clim((0, 255))

    # Plot contact positions dynamically
    if contacts_table is not None:
        contacts_handle, = ax.plot(
            [np.nan], [np.nan], '.', ms=contact_ms, color=contact_color)
    else:
        contacts_handle = None

    # Dynamic edge
    if edge_a is not None:
        edge_a_obj, = ax.plot([np.nan], [np.nan], '-', color='pink', lw=edge_lw)
    else:
        edge_a_obj = None
    
    # Text of trial
    if plot_trial_numbers:
        # Generate a handle to text
        txt = ax.text(
            .02, .02, 'waiting for text data',
            transform=ax.transAxes, # relative to axis size
            size=text_size, ha='left', va='bottom', color='w', 
            )
    
    # This will hold whisker objects
    whisker_handles = []
    
    # Create the writer
    writer = WhiskiWrap.FFmpegWriter(
        output_filename=output_filename,
        frame_width=canvas_width,
        frame_height=canvas_height,
        output_fps=output_fps,
        input_pix_fmt='argb',
        write_stderr_to_screen=write_stderr_to_screen,
        **ffmpeg_writer_kwargs
        )
    
    ## Loop until input frames exhausted
    for nnframe, frame in enumerate(input_reader.iter_frames()):
        # Account for the fact that we skipped the first input_frame_offset frames
        nframe = nnframe + input_frame_offset
        
        # Break if we're past the last trigger
        if nframe > np.max(frame_triggers) + trigger_dstop:
            break
        
        # Skip if we're not on a dframe
        if np.mod(nframe, d_temporal) != 0:
            continue
        
        # Skip if we're not near a trial
        nearest_choice_idx = np.nanargmin(np.abs(frame_triggers - nframe))
        nearest_choice = frame_triggers[nearest_choice_idx]
        if not (nframe > nearest_choice + trigger_dstart and 
            nframe < nearest_choice + trigger_dstop):
            continue

        # Announce
        if ((announced_frame_trigger < len(frame_triggers)) and 
            (nframe > frame_triggers[announced_frame_trigger] + trigger_dstart)):
            print("Reached trigger for frame", frame_triggers[announced_frame_trigger])
            announced_frame_trigger += 1

        # Update the trial text
        if plot_trial_numbers:
            if get_extra_text is not None:
                extra_text = get_extra_text(nframe)
            else:
                extra_text = ''
            txt.set_text('frame %d %s' % (nframe, extra_text))

        # Update the frame
        whisker_handles = frame_update(
            ax, nframe, frame, whisker_handles, contacts_table,
            post_contact_linger, joints, edge_a,
            im2, edge_a_obj, contacts_handle,
            d_spatial, d_temporal, whisker_lw=whisker_lw,
            whisker_marker=whisker_marker,
            whisker_marker_size=whisker_marker_size,
            )
        
        if func_update_figure is not None:
            func_update_figure(nframe)
        
        # Write to pipe
        f.canvas.draw()
        string_bytes = f.canvas.tostring_argb()
        writer.write_bytes(string_bytes)
    
    ## Clean up
    if whiskers_file_handle is not None:
        whiskers_file_handle.close()
    if not input_reader.isclosed():
        input_reader.close()
    writer.close()
    plt.close(f)    

def plot_stills_with_overlays_from_data(
    monitor_video_filename,
    frame_triggers=None,
    input_video_alpha=1,
    whiskers_table=None, whiskers_file_handle=None,
    edge_a=None, edge_alpha=1, typical_edges_hist2d=None, 
    contacts_table=None, post_contact_linger=50,
    imshow_interpolation='bilinear',
    contact_colors=None,
    force_contact_color=6,
    frame_clim=None, axa=None,
    contact_ms=15, edge_hist_kwargs=None,
    **kwargs):
    """Clone of write_video_with_overlays_from_data for still images.
    
    This function creates the graphics handles for the still image and
    the contacts and the edges and passes them to `frame_update`.
    
    monitor_video_filename : video to get stills from
    
    frame_triggers : frame numbers of stills
        Will be converted to array of ints
    
    input_video_alpha : the still image will have this alpha applied to it
    
    typical_edges_hist2d : if None, nothing happens
        Otherwise, it is displayed in another image in the axis with
        alpha equal to `edge_alpha`
    
    whiskers_table, whiskers_file_handle, contacts_table, 
        post_contact_linger :
        passed to `frame_update`
        
        `whiskers_file_handle` is closed at the end of this function if
        it is not None
    
    contact_colors : used to initiate the graphics handles for the contacts
        and then passed to `frame_update`. Note that this will also control
        the whiskers colors. If None, then WHISKER_COLOR_ORDER_W is used
    
    force_contact_color : if not None, then all contacts will be colored
        the same color, which is this index into contact_colors. This is done
        by copying contacts_table and overwriting 'color_group' with this
        value.
    
    contact_ms : marker size of contacts
    
    frame_clim : to apply to clim of that image. Default: (0, 255)
    
    axa : if None, a figure and axes are created
        Otherwise axa.flatten() needs to be the same length as frame_triggers
    
    Other kwargs are passed to `frame_update`
    
    """
    # Parse the arguments
    frame_triggers = np.asarray(frame_triggers).astype(np.int)
    announced_frame_trigger = 0
    input_width, input_height = my.video.get_video_aspect(
        monitor_video_filename)

    if contact_colors is None:
        contact_colors = whiskvid.WHISKER_COLOR_ORDER_W
    
    if force_contact_color is not None:
        contacts_table = contacts_table.copy()
        contacts_table['color_group'] = force_contact_color

    if axa is None:
        f, axa = my.plot.auto_subplot(len(frame_triggers), figsize=(15, 10))
    else:
        f = None
    
    for frame_trigger, ax in zip(frame_triggers, axa.flatten()):
        # Load the frame
        frame, stdout, stderr = my.video.get_frame(monitor_video_filename, 
            frame_number=frame_trigger)
        
        # Plot input video frames
        in_image = np.zeros((input_height, input_width))
        im2 = my.plot.imshow(in_image, ax=ax, 
            axis_call='image', cmap=plt.cm.gray, 
            interpolation=imshow_interpolation,
            extent=(0, input_width, input_height, 0))
        im2.set_alpha(input_video_alpha)
        
        if frame_clim is None:
            im2.set_clim((0, 255))
        else:
            im2.set_clim(frame_clim)

        # Plot typical edge images as static alpha
        if typical_edges_hist2d is not None:
            if edge_hist_kwargs is None:
                edge_hist_kwargs = {}
            im1 = whiskvid.plotting.plot_transparent_histogram(
                typical_edges_hist2d, ax=ax, 
                frame_width=input_width, frame_height=input_height,
                **edge_hist_kwargs
            )

        # Plot contact positions dynamically
        if contacts_table is not None:
            contact_positions_l = []
            for color in contact_colors:
                contact_positions_l.append(
                    ax.plot([np.nan], [np.nan], '.', ms=contact_ms, color=color)[0])
            #~ contact_positions, = ax.plot([np.nan], [np.nan], 'r.', ms=15)
        else:
            contact_positions_l = None

        # Dynamic edge
        if edge_a is not None:
            edge_a_obj, = ax.plot([np.nan], [np.nan], '-', color='pink', lw=3)
        else:
            edge_a_obj = None

        # Update the frame
        whisker_handles = frame_update(ax=ax, 
            nframe=frame_trigger, 
            frame=frame,
            whisker_handles=[], 
            contacts_table=contacts_table,
            post_contact_linger=post_contact_linger, 
            whiskers_table=whiskers_table, 
            whiskers_file_handle=whiskers_file_handle, 
            edge_a=edge_a,
            im2=im2, 
            edge_a_obj=edge_a_obj, 
            contact_positions_l=contact_positions_l,
            d_spatial=1, d_temporal=1, contact_colors=contact_colors,
            **kwargs
        )

    # Clean up
    if whiskers_file_handle is not None:
        whiskers_file_handle.close()

    for ax in axa.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    if f is not None:
        f.tight_layout()