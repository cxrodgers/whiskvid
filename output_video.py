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
    d_spatial, d_temporal):
    """Helper function to plot each frame.
    
    Typically this is called by write_video_with_overlays.
    
    nframe : number of frame
        This is used to determine which whiskers and which contacts to plot
    frame : the image data
    whisker_handles : handles to whiskers lines that will be deleted
    contacts_table : contacts to plot
    
    Returns: whisker_handles
        These are returned so that they can be deleted next time
    """
    # Get the frame
    im2.set_data(frame[::d_spatial, ::d_spatial])
    
    # Get the edges
    if edge_a is not None:
        edge_a_frame = edge_a[nframe]
        if edge_a_frame is not None:
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
        if 'group' in subtac.columns:
            # Add a color column
            for ncolor, contact_positions in enumerate(contact_positions_l):
                subsubtac = subtac[
                    subtac['group'].mod(len(contact_positions_l)) == ncolor]
                contact_positions.set_xdata(subsubtac['tip_x'])
                contact_positions.set_ydata(subsubtac['tip_y'])
        else:
            contact_positions_l[0].set_xdata(subtac['tip_x'])
            contact_positions_l[0].set_ydata(subtac['tip_y'])
    
    # Get the whiskers for this frame
    if whiskers_file_handle is not None:
        # Remove old whiskers
        for handle in whisker_handles:
            handle.remove()
        whisker_handles = []            
        
        sub_summary = whiskers_table[whiskers_table.time == nframe]
        for idx, row in sub_summary.iterrows():
            line, = ax.plot(
                whiskers_file_handle.root.pixels_x[idx],
                whiskers_file_handle.root.pixels_y[idx],
                color='yellow')
            whisker_handles.append(line)
            #~ line, = ax.plot([row['fol_x']], [row['fol_y']], 'gs')
            #~ whisker_handles.append(line)
            #~ line, = ax.plot([row['tip_x']], [row['tip_y']], 'rs')
            #~ whisker_handles.append(line)
    
    return whisker_handles

def write_video_with_overlays(output_filename, 
    input_reader, input_width, input_height, verbose=True,
    whiskers_filename=None, edges_filename=None, contacts_filename=None,
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
    else:
        whiskers_file_handle = None
    
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
    contacts_table=None, post_contact_linger=100,
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
    
    # Other sources of input
    edge_alpha : alpha of edge
    post_contact_linger : How long to leave the contact displayed    
    """
    # We need FFmpegWriter
    # Probably that object should be moved to my.video
    # Or maybe a new repo ffmpeg_tricks
    import WhiskiWrap

    # Parse the arguments
    frame_triggers = np.asarray(frame_triggers)
    announced_frame_trigger = 0

    ## Set up the graphical handles
    if verbose:
        print "setting up handles"

    # Create a figure with an image that fills it
    figsize = input_width / dpi, input_height / dpi
    f = plt.figure(frameon=False, dpi=dpi/d_spatial, figsize=figsize)
    canvas_width, canvas_height = f.canvas.get_width_height()
    ax = f.add_axes([0, 0, 1, 1])
    ax.axis('off')

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
        n_colors = 7
        contact_colors = my.plot.generate_colorbar(n_colors)
        contact_positions_l = []
        for color in contact_colors:
            contact_positions_l.append(
                ax.plot([np.nan], [np.nan], '.', ms=15, color=color)[0])
        #~ contact_positions, = ax.plot([np.nan], [np.nan], 'r.', ms=15)

    # Dynamic edge
    if edge_a is not None:
        edge_a_obj, = ax.plot([np.nan], [np.nan], 'g-')
    
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
        )
    
    ## Loop until input frames exhausted
    for nframe, frame in enumerate(input_reader.iter_frames()):
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
            txt.set_text('frame %d trial %d' % (nframe, nearest_choice_idx))
            trial_number = nearest_choice_idx

        # Update the frame
        whisker_handles = frame_update(ax, nframe, frame, whisker_handles, contacts_table,
            post_contact_linger, whiskers_table, whiskers_file_handle, edge_a,
            im2, edge_a_obj, contact_positions_l,
            d_spatial, d_temporal)
        plt.draw()

        # Write to pipe
        string_bytes = f.canvas.tostring_argb()
        writer.write_bytes(string_bytes)
    
    ## Clean up
    whiskers_file_handle.close()
    if not input_reader.isclosed():
        input_reader.close()
    writer.close()
    plt.close('f')    