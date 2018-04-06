import matplotlib.pyplot as plt
import pandas
import numpy as np
import my
import my.plot
import my.video
import whiskvid
import tables

def plot_single_frame(frame_data, ax, vs, frame_number, ds_ratio=2, 
    colors=None, key='object'):
    """Plot frame with whiskers overlaid
    
    Returns: ax, res
    """
    # Get colors
    if colors is None:
        colors = whiskvid.WHISKER_COLOR_ORDER_W[1:] + [
            'crimson', 'lime', 'darksalmon']
    
    # Get the frame
    im, stdout, stderr = my.video.get_frame(vs.data.monitor_video.get_path,
        frame_number=frame_number)
    
    # Downsample the image for speed
    my.plot.imshow(im[::ds_ratio, ::ds_ratio], ax=ax, cmap=plt.cm.gray, 
        xd_range=(0, vs.frame_width),
        yd_range=(0, vs.frame_height),
    )
    ax.axis('image')  

    # Plot whiskers
    lines_l = []
    with tables.open_file(vs.data.whiskers.get_path) as fi:
        for idx in frame_data.index:
            # Get the color
            object_id = int(frame_data.loc[idx, key])
            color = colors[object_id]
            
            # Get the pixels
            pixels_x = fi.root.pixels_x[idx]
            pixels_y = fi.root.pixels_y[idx]
            
            # Plot
            line, = ax.plot(pixels_x, pixels_y, color=color, lw=0.75)
            
            # Text the label
            ax.text(
                pixels_x[len(pixels_x) // 2],
                pixels_y[len(pixels_x) // 2],
                str(object_id), ha='center', va='center', color='k')
            
            # Store handle
            lines_l.append(line)
    
    # Will return index, object, and line of each
    res = frame_data[[key]].copy()
    res.loc[:, 'line'] = lines_l
    
    return ax, res

def clear_axis(ax):
    # Remove all images
    while ax.images:
        im = ax.images[0]
        im.set_visible(False)
        ax.images[0].remove()
    
    # Remove all lines
    while ax.lines:
        line = ax.lines[0]
        line.set_visible(False)
        line.remove()

    # Remove all textx
    while ax.texts:
        txt = ax.texts[0]
        txt.set_visible(False)
        txt.remove()    

def parse_input(choice):
    """Parse text input into decision
    
    choice : text input
    
    If choice is 'f':
        go forward
    If choice is 'b':
        go back
    If choice is 'q':
        quit
    If choice is 'c':
        mark as confirmed and go forward
    If choices is 'm':
        mark as munged and go forward
    If choices is 'M':
        mark as unmunged and go forward
    If choice is 'sMN':
        switch M and N
    If choice is 'u':
        go to next unconfirmed
    
    Otherwise, the result is 'stay', switchdict is {}, confirm is False,
    munged is False
    
    Returns: result, confirm, munged, switchdict
        result: 'next', 'back', 'quit', 'stay', 'next unconfirmed'
        confirm: True or False
        munged: True or False
        unmunged: True or False
        switchdict: dict from M to N and N to M
    """
    # Parse answer into switchdict
    result = 'stay'
    confirm = False
    munged = False
    unmunged = False
    switchdict = {}

    # Strip
    schoice = choice.strip()
    
    # Switch on length of input
    if len(schoice) == 1:
        # One character input
        cmd = schoice[0]
        if schoice == 'f':
            result = 'next'
        elif schoice == 'b':
            result = 'back'
        elif cmd == 'q':
            result = 'quit'
        elif cmd == 'm':
            result = 'next'
            munged = True
        elif cmd == 'M':
            result = 'next'
            unmunged = True
        elif cmd == 'c':
            result = 'next'
            confirm = True
        elif cmd == 'u':
            result = 'next unconfirmed'
    
    elif len(schoice) == 3:
        # Three character input: command, whisker0, whisker1
        cmd, w0s, w1s = schoice
        if str.isalpha(cmd) and str.isdigit(w0s) and str.isdigit(w1s):
            # Intify whisker
            w0 = int(w0s)
            w1 = int(w1s)
            
            # Switch on command
            if cmd == 's':
                # Switch whiskers 0 and 1
                switchdict[w0] = w1
                switchdict[w1] = w0
    
    return result, confirm, munged, unmunged, switchdict

def interactive_curation(keystone_frame, key_frames, classified_data, vs,
    existing_curation_data=None):
    """Interactively collect curation
    
    keystone_frame : frame that defines the object mapping
        It will be displayed on the left
    key_frames : frames to curate
    classified_data : result of a classifier run
    vs : VideoSession for plotting whiskers
    existing_curation_data : previously curated results (DataFrame)
    
    Returns: curation_data, munged_frames
        curation_data : An updated version of existing_curation_data
        munged_frames : list of munged frames
    """
    # Error checking
    if len(key_frames) != len(np.unique(key_frames)):
        raise ValueError("non unique key frames provided")
    if keystone_frame in key_frames:
        raise ValueError("keystone frame in key frames")

    # Interactive mode
    plt.ion()
    
    # Handles
    f, axa = plt.subplots(1, 2, figsize=(9, 4))

    # Plot keystone in left
    frame_data = my.pick_rows(classified_data, frame=keystone_frame)
    plot_single_frame(frame_data, axa[0], vs, keystone_frame, ds_ratio=2, 
        key='object')

    # Convert curation results to dict for backwards compatibility
    curated_results_d = {}
    if existing_curation_data is not None:
        for frame in existing_curation_data.index.levels[0]:
            curated_results_d[frame] = existing_curation_data.loc[frame]

    # Store list of munged frames
    munged_frames = []

    # Plot each key in turn on right
    quit_running = False
    current_frame_idx = 0
    frame_data = None
    while not quit_running:
        # Get current key frame
        key_frame = key_frames[current_frame_idx]
        axa[1].set_title(key_frame)

        # `frame_data` is None iff we just switched frames
        # In that case we need to load the frame data
        if frame_data is None:
            # Get data from this frame
            frame_data = my.pick_rows(classified_data, 
                frame=key_frame).copy()

            # Load confirmed results if there are any
            if key_frame in curated_results_d:
                frame_data['object'] = curated_results_d[key_frame][
                    'object'].copy()
                previously_confirmed = True
            else:
                previously_confirmed = False

            # Intify in any case
            frame_data['object'] = frame_data['object'].astype(np.int).copy()

        # Determine if everything has been confirmed
        all_confirmed = np.all(np.in1d(key_frames, 
            munged_frames + curated_results_d.keys()
        ))

        # Determine if this frame is munged
        if key_frame in munged_frames:
            frame_is_munged = True
        else:
            frame_is_munged = False
        
        # Title accordingly
        if all_confirmed:
            axa[1].set_title('all done %d' % key_frame)
        elif previously_confirmed:
            axa[1].set_title('confirmed %d' % key_frame)
        elif frame_is_munged:
            axa[1].set_title('munged %d' % key_frame)
        else:
            axa[1].set_title('%d' % key_frame)
        
        # Clear axis and then plot single frame
        clear_axis(axa[1])
        plot_single_frame(frame_data, axa[1], vs, key_frame, ds_ratio=2, 
            key='object')
        plt.draw()

        # Get answer
        choice = raw_input("[c]onfirm / [f]orward / [b]ack / "
            "[m]ung / un[M]ung / [q]uit / next [u]nconfirmed / [sXY]switch: ")
        
        # Parse
        result, confirm, munged, unmunged, switchdict = parse_input(choice)

        # Apply switchdict to frame_data['object'] using a temporary column
        frame_data['object2'] = frame_data['object'].copy()
        for w0, w1 in switchdict.items():
            print "switching %d to %d" % (w0, w1)
            frame_data.loc[frame_data['object'] == w0, 'object2'] = w1
        frame_data['object'] = frame_data['object2'].copy()
        frame_data = frame_data.drop('object2', axis=1)
        
        # Append to munged_frames if munged
        if munged and key_frame not in munged_frames:
            munged_frames.append(key_frame)
        
        # Drop from results if munged
        if munged and key_frame in curated_results_d:
            curated_results_d.pop(key_frame)
        
        # Remove from munged_frames if unmunged
        if unmunged:
            if key_frame in munged_frames:
                munged_frames.remove(key_frame)
        
        # Store in dict if confirmed
        if confirm:
            print "storing confirmed result"
            curated_results_d[key_frame] = frame_data[[
                'streak', 'object']].copy()
        
        # Go to next or previous frame, or quit
        if result in ['next', 'back', 'next unconfirmed']:
            frame_data = None
        
            if result == 'next unconfirmed':
                unconfirmed_frames_mask = ~np.in1d(key_frames, 
                    munged_frames + curated_results_d.keys())

                if np.any(unconfirmed_frames_mask):
                    current_frame_idx = np.where(unconfirmed_frames_mask)[0][0]
                else:
                    print "no more unconfirmed"
        
            if result == 'next':
                current_frame_idx = np.mod(current_frame_idx + 1, 
                    len(key_frames))
            
            if result == 'back':
                current_frame_idx = np.mod(current_frame_idx - 1, 
                    len(key_frames))
        
        if result == 'quit':
            break

    # Interactive off
    plt.ioff()
    
    # Concat results
    if len(curated_results_d) == 0:
        cres = None
    else:
        cres = pandas.concat(curated_results_d, names=['frame'])
        cres.index.names = ['frame', 'mwe_index']
    
    return cres, munged_frames