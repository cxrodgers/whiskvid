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

def write_video_with_overlays(output_filename, input, input_width, input_height,
    verbose=True,
    frame_triggers=None, trigger_dstart=-250, trigger_dstop=50,
    plot_trial_numbers=True,
    d_temporal=5, d_spatial=1,
    dpi=50, output_fps=30, in_buff_sz=1500,
    input_video_alpha=1,
    whiskers_filename=None, side='left',
    edges_filename=None, edge_alpha=1, typical_edges_hist2d=None, 
    contacts_filename=None, post_contact_linger=100,
    ):
    """Creating a video overlaid with whiskers, contacts, etc.
    
    The overall dataflow is this:
    1. Load chunks of frames from the input
    2. One by one, plot the frame with matplotlib. Overlay whiskers, edges,
        contacts, whatever.
    3. Dump the frame to an ffmpeg writer.
    
    # Input and output
    output_filename : file to create
    input : PFReader or input video
    
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
    in_buff_sz : Number of frames to load at once from the input
    input_video_alpha : alpha of image
    
    # Other sources of input
    whiskers_filename : name of HDF5 table containing whiskers
    edges_filename : name of file containing edges
    edge_alpha : alpha of edge
    contacts_filename : HDF5 file containing contact info
    post_contact_linger : How long to leave the contact displayed
    """
    import WhiskiWrap
    # Parse the arguments
    frame_triggers = np.asarray(frame_triggers)
    
    
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
    if contacts_filename is not None:
        contact_positions, = ax.plot([np.nan], [np.nan], 'r.', ms=15)

    # Dynamic edge
    if edges_filename is not None:
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
    
    ## Frame updating function
    def update(nframe, frame, whisker_handles, contacts_table):
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
            subtac = contacts_table[(contacts_table.frame < nframe) & 
                (contacts_table.frame >= nframe - post_contact_linger)]
            contact_positions.set_xdata(subtac['tip_x'])
            contact_positions.set_ydata(subtac['tip_y'])
        
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
    
    ## Loop until input frames exhausted
    for nframe, frame in enumerate(input.iter_frames()):
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

        # Update the trial text
        if plot_trial_numbers and (nearest_choice_idx > trial_number):
            txt.set_text('trial %d' % nearest_choice_idx)
            trial_number = nearest_choice_idx

        # Update the frame
        whisker_handles = update(nframe, frame, whisker_handles, contacts_table)
        plt.draw()

        # Write to pipe
        string_bytes = f.canvas.tostring_argb()
        writer.write_bytes(string_bytes)
    
    ## Clean up
    whiskers_file_handle.close()
    input.close()
    writer.close()
    plt.close('f')
    
    
def dump_video_with_edge_and_tac(video_filename, typical_edges_hist2d, tac,
    edge_a, output_filename, frame_triggers, 
    trigger_dstart=-250, trigger_dstop=50,
    d_temporal=5, d_spatial=1, dpi=50, output_fps=30, input_video_fps=30,
    edge_alpha=1, input_video_alpha=.5, post_contact_linger=100,
    in_buff_sz=1500):
    """Overplot the input video with the edge summary and the contacts.
    
    d_temporal : Save time by plotting every Nth frame
    d_spatial : Save time by spatial undersampling the image
        The bottleneck is typically plotting the raw image in matplotlib
    frame_triggers : Only plot frames within (trigger_dstart, trigger_dstop)
        of a value in this array
    dpi : sort of a fictitious dpi, since we always make it pixel by pixel
        the same size as the input video, but it matters for font size and
        marker size. Ideally this divides v_width and v_height, I guess?
    """
    # Get metadata
    v_width, v_height = my.video.get_video_aspect(video_filename)
    n_frames_total = int(my.video.get_video_duration(video_filename) 
        * input_video_fps)
    
    # Set up the input pipe
    in_command = ['ffmpeg', '-i', video_filename, '-f', 'image2pipe', 
        '-pix_fmt', 'gray', '-vcodec', 'rawvideo', '-']
    in_pipe = subprocess.Popen(in_command, 
        stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'), bufsize=10**9)

    # Create a figure with an image that fills it
    figsize = v_width / dpi, v_height / dpi
    f = plt.figure(frameon=False, figsize=figsize, dpi=dpi / d_spatial)
    canvas_width, canvas_height = f.canvas.get_width_height()
    ax = f.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Plot typical edge images as static alpha
    im1 = my.plot.imshow(typical_edges_hist2d, ax=ax, axis_call='image',
        extent=(0, v_width, v_height, 0), cmap=plt.cm.gray)
    im1.set_alpha(edge_alpha)

    # Plot input input ivdeo frames dynamically
    in_image = np.zeros((v_height, v_width))
    im2 = my.plot.imshow(in_image[::d_spatial, ::d_spatial], ax=ax, 
        axis_call='image', cmap=plt.cm.gray, extent=(0, v_width, v_height, 0))
    im2.set_alpha(input_video_alpha)
    im2.set_clim((0, 255))

    # Plot contact positions dynamically
    contact_positions, = ax.plot([np.nan], [np.nan], 'r.', ms=15)

    # Dynamic edge
    edge_a_obj, = ax.plot([np.nan], [np.nan], 'g-')

    # Text of trial
    txt = ax.text(0, ax.get_ylim()[0], 'waiting', 
        size=20, ha='left', va='bottom', color='w')
    trial_number = -1

    def update(frame, in_image):
        """Updating function"""
        # Get the frame
        im2.set_data(in_image[::d_spatial, ::d_spatial])
        
        # Get the edges
        edge_a_frame = edge_a[frame]
        if edge_a_frame is not None:
            edge_a_obj.set_xdata(edge_a_frame[:, 1])
            edge_a_obj.set_ydata(edge_a_frame[:, 0])
        else:
            edge_a_obj.set_xdata([np.nan])
            edge_a_obj.set_ydata([np.nan])
        
        # Get the contacts
        subtac = tac[(tac.frame < frame) & 
            (tac.frame >= frame - post_contact_linger)]
        contact_positions.set_xdata(subtac['tip_x'])
        contact_positions.set_ydata(subtac['tip_y'])

    # Set up the input buffer
    in_buff = []
    read_size = v_width * v_height * in_buff_sz

    # Open an ffmpeg process
    cmdstring = ('ffmpeg', 
        '-y', '-r', '%d' % output_fps,
        '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
        '-pix_fmt', 'argb', # format
        '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
        '-vcodec', 'libx264', '-crf', '15', output_filename) # output encoding
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

    # Iterate over frames
    try:
        for frame in xrange(n_frames_total):
            # Load input image
            if len(in_buff) == 0:
                print "reloading input buffer", frame
                raw_image = in_pipe.stdout.read(read_size)
                if len(raw_image) < read_size:
                    raise OutOfFrames
                flattened_im = np.fromstring(raw_image, dtype='uint8')
                reshaped_im = flattened_im.reshape(
                    (in_buff_sz, v_height, v_width))
                in_buff = list(reshaped_im)            
            in_image = in_buff.pop(0)
            
            # Break if we're past the last trigger
            if frame > np.max(frame_triggers) + trigger_dstop:
                break
            
            # Skip if we're not on a dframe
            if np.mod(frame, d_temporal) != 0:
                continue
            
            # Skip if we're not near a trial
            nearest_choice_idx = np.nanargmin(np.abs(frame_triggers - frame))
            nearest_choice = frame_triggers[nearest_choice_idx]
            if not (frame > nearest_choice + trigger_dstart and 
                frame < nearest_choice + trigger_dstop):
                continue

            # Update the trial text
            if nearest_choice_idx > trial_number:
                txt.set_text('trial %d' % nearest_choice_idx)
                trial_number = nearest_choice_idx

            # Update the rame
            update(frame, in_image)
            plt.draw()
            
            # Write to pipe
            string = f.canvas.tostring_argb()
            p.stdin.write(string)

    except OutOfFrames:
        print "out of frames"

    finally:
        # Finish up
        p.communicate()
        in_pipe.terminate()
        