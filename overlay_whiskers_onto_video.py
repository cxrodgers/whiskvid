import matplotlib.pyplot as plt
import numpy as np, os.path, pandas, my, scipy.misc

def make_movie_tracked_whiskers_and_angle(output_dir, 
    video_file, video_frame_times,
    angles_to_plot, angles_to_plot_t, 
    downsample_ratio=10, truncate=None,
    piv_ws=None, video_frame_numbers=None,
    temp_output_file='temp.png', dpi=200, ylims=None):
    """Writes out frames with tracked whiskers and calculated angles
    
    Creates a figure with two subplots. The top will be an image from
    the video with the whiskers overlaid. The bottom will be a scrolling
    graph of the tracked angle versus time.
    
        output_dir : where to put frames
        video_file : source video to extract frames from
        video_frame_times : time in video timebase to extract each frame
            This will be downsampled by `downsample_ratio`, and that sets
            the output framerate.
        angles_to_plot : 2d array. Each column will be plotted vs time.
            Generally this is some average angle, or individual whisker angles.
            This must be the same size as video_frame_times, so that it knows
            how to link up the trace with the video!
        angles_to_plot_t : time values against which to plot the angle.
            This serves no purpose other than aesthetics.
            Must be same length as angles_to_plot.
        piv_ws : DataFrame of whisker x and y to overlay.
            Must be indexed by frame number.
        video_frame_numbers : how to index into angles_to_plot.
            Must be same length as video_frame_times.
            For each frame dumped, the corresponding frame number here is
            used to index into piv_ws and extract the corresponding whisker
            traces.
    
    Note: I previously commented this:  
        # FFMPEG needs to know the time of the frame
        # It seems to round up, strangely, so subtract a ms and then round to the ms
        frametime = np.round((frame / video_framerate) - .001, 3)
    
    Afterwards, use something like this to create the video:
    ffmpeg -r 30 -i %06d.png -y -vcodec mpeg4 out.mp4
    """
    assert len(angles_to_plot) == len(video_frame_times)

    if ylims is None:
        ylims = (angles_to_plot.min() - 20, angles_to_plot.max() + 20)
    
    if not os.path.exists(output_dir):
        print "creating", output_dir
        os.mkdir(output_dir)

    ## Initialize figure
    f, axa = plt.subplots(2, 1, figsize=(5, 8.2))
    f.subplots_adjust(left=.15, right=.95, top=.97, bottom=.05)

    # Also plot the angle
    axa[1].set_xlabel('time (s)')
    axa[1].set_ylabel('whisker angle (degrees)')

    # Full time course
    axa[1].plot(angles_to_plot_t, angles_to_plot)
    axa[1].set_ylim(ylims)

    # This will be used to show the current time
    time_line, = axa[1].plot([0, 0], axa[1].get_ylim(), 'k')


    ## Iterate over frames
    n_output_frame = 0
    for n_input_frame in range(0, len(video_frame_times), downsample_ratio):
        if truncate is not None and n_output_frame > truncate:
            break
        
        # Get the time of the frame to dump
        frametime = video_frame_times[n_input_frame]

        # status
        print n_output_frame, n_input_frame, frametime
        
        # Dump the frame and re-load it
        my.misc.frame_dump(video_file, frametime=frametime, 
            output_filename=temp_output_file)
        im = scipy.misc.imread(temp_output_file, flatten=True)

        # Now plot it
        axa[0].imshow(im, interpolation='nearest', cmap=plt.cm.gray)
        for imobj in axa[0].get_images():
            imobj.set_clim((0, 255))

        # Also load and plot the whiskers from this frame
        if piv_ws is not None and video_frame_numbers is not None:
            # Index into original frames
            original_frame_number = video_frame_numbers[n_input_frame]
            frame_whiskers = piv_ws.ix[original_frame_number]
            
            for wid in frame_whiskers.index:
                wsx = frame_whiskers['wsx'][wid]
                wsy = frame_whiskers['wsy'][wid]
                axa[0].plot(wsx, wsy)#whiski_color_cycle[wid])

        # Limits match axis
        axa[0].set_xlim((0, im.shape[1] - 1))
        axa[0].set_ylim((im.shape[0] - 1, 0))
        
        # Update the traces
        # Index by n_input_frame to get the corresponding times in whisker angle
        whisker_time = angles_to_plot_t[n_input_frame]
        axa[1].set_xlim((whisker_time - 1.0, whisker_time + 1.0))
        time_line.set_xdata(np.array([whisker_time, whisker_time]))
        
        # Save and prepare to ffmpeg
        f.savefig(os.path.join(output_dir, '%06d.png' % n_output_frame), dpi=dpi)
        n_output_frame = n_output_frame + 1
        
        # Clear the image
        axa[0].clear()    