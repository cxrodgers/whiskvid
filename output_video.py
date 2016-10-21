"""Module for creating output videos with various overlays"""

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
    post_contact_linger, whiskers_table, whiskers_file_handle, edge_a,
    im2, edge_a_obj, contact_positions_l,
    d_spatial, d_temporal, contact_colors,
    truncate_edge_y=100):
    """Helper function to plot each frame.
    
    Typically this is called by write_video_with_overlays.
    
    nframe : number of frame
        This is used to determine which whiskers and which contacts to plot
    frame : the image data
    whisker_handles : handles to whiskers lines that will be deleted
    contacts_table : contacts to plot
    
    truncate_edge_y : if Not None, drops everything with y < this value
        in the edge. This is useful for when the shape edge connects
        to the top of th frame
    
    Returns: whisker_handles
        These are returned so that they can be deleted next time
    """
    # figure out what the frame column is called
    if whiskers_table is not None:
        if 'frame' in whiskers_table.columns:
            FRAME_LABEL = 'frame'
        else:
            FRAME_LABEL = 'time'
        if FRAME_LABEL not in whiskers_table:
            raise ValueError("cannot find the frame column in the whiskers table")
    
    # Get the frame
    im2.set_data(frame[::d_spatial, ::d_spatial])
    
    # Get the edges
    if edge_a is not None:
        edge_a_frame = edge_a[nframe]
        if edge_a_frame is not None:
            if truncate_edge_y is not None:
                edge_a_obj.set_data(edge_a_frame[
                    edge_a_frame[:, 0] >= truncate_edge_y].T[::-1])
            else:
                edge_a_obj.set_xdata(edge_a_frame[:, 1])
                edge_a_obj.set_ydata(edge_a_frame[:, 0])
        else:
            edge_a_obj.set_xdata([np.nan])
            edge_a_obj.set_ydata([np.nan])
    
    # Get the contacts
    if contacts_table is not None:
        # Grab the contacts from frames (nframe - post_contact_linger, nframe]
        subtac = contacts_table[
            (contacts_table.frame <= nframe) & 
            (contacts_table.frame > nframe - post_contact_linger)
            ]
        
        # Split on group if it exists
        if 'color_group' in subtac.columns:
            # We've already decided how to color the contacts
            for ncolor, contact_positions in enumerate(contact_positions_l):
                subsubtac = subtac[subtac['color_group'] == ncolor]
                contact_positions.set_xdata(subsubtac['tip_x'])
                contact_positions.set_ydata(subsubtac['tip_y'])            
        elif 'group' in subtac.columns:
            # We've grouped the contacts but haven't decided how to
            # color them yet
            for ncolor, contact_positions in enumerate(contact_positions_l):
                subsubtac = subtac[
                    subtac['group'].mod(len(contact_positions_l)) == ncolor]
                contact_positions.set_xdata(subsubtac['tip_x'])
                contact_positions.set_ydata(subsubtac['tip_y'])
        else:
            contact_positions_l[0].set_xdata(subtac['tip_x'])
            contact_positions_l[0].set_ydata(subtac['tip_y'])
    
    # Get the whiskers for this frame
    if whiskers_table is not None and whiskers_file_handle is not None:
        # Remove old whiskers
        for handle in whisker_handles:
            handle.remove()
        whisker_handles = []            
        
        sub_summary = whiskers_table[whiskers_table[FRAME_LABEL] == nframe]
        for idx, row in sub_summary.iterrows():
            if 'color_group' in row:
                color = contact_colors[int(row['color_group'])]
            else:
                color = 'yellow'
            line, = ax.plot(
                whiskers_file_handle.root.pixels_x[idx],
                whiskers_file_handle.root.pixels_y[idx],
                color=color)
            whisker_handles.append(line)
            #~ line, = ax.plot([row['fol_x']], [row['fol_y']], 'gs')
            #~ whisker_handles.append(line)
            #~ line, = ax.plot([row['tip_x']], [row['tip_y']], 'rs')
            #~ whisker_handles.append(line)
    
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
            print "loading whiskers"
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
                print "loading contacts"
            contacts_table = pandas.read_pickle(contacts_filename)
        else:
            contacts_table = None
    
    # Load edges
    if edges_filename is not None:
        if verbose:
            print "loading edges"
        edge_a = np.load(edges_filename)
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
    whiskers_table=None, whiskers_file_handle=None, side='left',
    edge_a=None, edge_alpha=1, typical_edges_hist2d=None, 
    contacts_table=None, post_contact_linger=50,
    write_stderr_to_screen=True,
    input_frame_offset=0,
    get_extra_text=None,
    contact_colors=None,
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
    
    # Other sources of input
    edge_alpha : alpha of edge
    post_contact_linger : How long to leave the contact displayed    
        This is the total duration, so 0 will display nothing, and 1 is minimal.
    
    # Misc
    get_extra_text : if not None, should be a function that accepts a frame
        number and returns some text to add to the display. This is a 
        "real" frame number after accounting for any offset.
    contact_colors : list of color specs to use
    """
    # We need FFmpegWriter
    # Probably that object should be moved to my.video
    # Or maybe a new repo ffmpeg_tricks
    import WhiskiWrap

    # Parse the arguments
    frame_triggers = np.asarray(frame_triggers)
    announced_frame_trigger = 0
    input_width = int(input_width)
    input_height = int(input_height)

    if contact_colors is None:
        n_colors = 7
        contact_colors = my.plot.generate_colorbar(n_colors)

    ## Set up the graphical handles
    if verbose:
        print "setting up handles"

    # Create a figure with an image that fills it
    # We want the figsize to be in inches, so divide by dpi
    # And we want one invisible axis containing an image that fills the whole figure
    figsize = input_width / float(dpi), input_height / float(dpi)
    f = plt.figure(frameon=False, dpi=dpi/d_spatial, figsize=figsize)
    ax = f.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # This return results in pixels, so should be the same as input width
    # and height. If not, probably rounding error above
    canvas_width, canvas_height = f.canvas.get_width_height()
    if \
        input_width / d_spatial != canvas_width or \
        input_height / d_spatial != canvas_height:
        raise ValueError("canvas size is not the same as input size")

    # Plot typical edge images as static alpha
    if typical_edges_hist2d is not None:
        im1 = my.plot.imshow(typical_edges_hist2d, ax=ax, axis_call='image',
            extent=(0, input_width, input_height, 0), cmap=plt.cm.gray)
        im1.set_alpha(edge_alpha)

    # Plot input video frames
    in_image = np.zeros((input_height, input_width))
    im2 = my.plot.imshow(in_image[::d_spatial, ::d_spatial], ax=ax, 
        axis_call='image', cmap=plt.cm.gray, extent=(0, input_width, input_height, 0))
    im2.set_alpha(input_video_alpha)
    im2.set_clim((0, 255))

    # Plot contact positions dynamically
    if contacts_table is not None:
        contact_positions_l = []
        for color in contact_colors:
            contact_positions_l.append(
                ax.plot([np.nan], [np.nan], '.', ms=15, color=color)[0])
        #~ contact_positions, = ax.plot([np.nan], [np.nan], 'r.', ms=15)
    else:
        contact_positions_l = None

    # Dynamic edge
    if edge_a is not None:
        edge_a_obj, = ax.plot([np.nan], [np.nan], '-', color='pink', lw=3)
    else:
        edge_a_obj = None
    
    # Text of trial
    if plot_trial_numbers:
        txt = ax.text(0, ax.get_ylim()[0], 'waiting', 
            size=20, ha='left', va='bottom', color='w')
        trial_number = -1    
    
    # This will hold whisker objects
    whisker_handles = []
    
    # Create the writer
    writer = WhiskiWrap.FFmpegWriter(
        output_filename=output_filename,
        frame_width=input_width/d_spatial,
        frame_height=input_height/d_spatial,
        output_fps=output_fps,
        pix_fmt='argb',
        write_stderr_to_screen=write_stderr_to_screen,
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
            print "Reached trigger for frame", frame_triggers[announced_frame_trigger]
            announced_frame_trigger += 1

        # Update the trial text
        if plot_trial_numbers:# and (nearest_choice_idx > trial_number):
            if get_extra_text is not None:
                extra_text = get_extra_text(nframe)
            else:
                extra_text = ''
            txt.set_text('frame %d trial %d %s' % (nframe, nearest_choice_idx, extra_text))
            trial_number = nearest_choice_idx

        # Update the frame
        whisker_handles = frame_update(ax, nframe, frame, whisker_handles, contacts_table,
            post_contact_linger, whiskers_table, whiskers_file_handle, edge_a,
            im2, edge_a_obj, contact_positions_l,
            d_spatial, d_temporal, contact_colors)
        
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