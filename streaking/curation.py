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
        colors = whiskvid.WHISKER_COLOR_ORDER_W[1:]
    
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
    res = frame_data[['object']].copy()
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
    # Parse answer into switchdict
    result = 'stay'
    confirm = False
    switchdict = {}

    # Strip
    schoice = choice.strip()
    
    # Switch on length of input
    if schoice == '':
        # Take this as confirmation
        result = 'next'
        confirm = True
    
    elif len(schoice) == 1:
        # One character input
        cmd = schoice[0]
        if schoice == 'f':
            result = 'next'
        elif schoice == 'b':
            result = 'back'
        elif cmd == 'q':
            result = 'quit'
    
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
            elif cmd == 'd':
                # Relabel whisker 0 as whisker 1
                switchdict[w0] = w1    
    
    return result, confirm, switchdict

def interactive_curation(keystone_frame, key_frames, classified_data, vs):
    """Interactively collect curation"""
    # Interactive mode
    plt.ion()
    
    # Handles
    f, axa = plt.subplots(1, 2, figsize=(9, 4))

    # Plot keystone in left
    frame_data = my.pick_rows(classified_data, frame=keystone_frame)
    plot_single_frame(frame_data, axa[0], vs, keystone_frame, ds_ratio=2, 
        key='object')

    # Plot each key in turn on right
    curated_results_d = {}
    quit_running = False
    current_frame_idx = 0
    frame_data = None
    while not quit_running:
        # Get current key frame
        key_frame = key_frames[current_frame_idx]
        axa[1].set_title(key_frame)

        # Get data from this frame
        if frame_data is None:
            frame_data = my.pick_rows(classified_data, 
                frame=key_frame).copy()

        # Load results if there are any
        if key_frame in curated_results_d:
            frame_data['object'] = curated_results_d[key_frame]['object'].copy()
            previously_confirmed = True
        else:
            previously_confirmed = False

        # Intify in any case
        frame_data['object'] = frame_data['object'].astype(np.int).copy()

        # Determine if everything has been confirmed
        all_confirmed = np.all(np.in1d(key_frames, curated_results_d.keys()))
        
        # Title accordingly
        if all_confirmed:
            axa[1].set_title('all done %d' % key_frame)
        elif previously_confirmed:
            axa[1].set_title('confirmed %d' % key_frame)
        else:
            axa[1].set_title('%d' % key_frame)
        
        # Clear axis and then plot single frame
        clear_axis(axa[1])
        plot_single_frame(frame_data, axa[1], vs, key_frame, ds_ratio=2, 
            key='object')
        plt.draw()

        # Get answer
        choice = raw_input("Confirm [y/n/q]: ")
        
        # Parse
        result, confirm, switchdict = parse_input(choice)

        # Apply switchdict to frame_data['object'] using a temporary column
        frame_data['object2'] = frame_data['object'].copy()
        for w0, w1 in switchdict.items():
            print "switching %d to %d" % (w0, w1)
            frame_data.loc[frame_data['object'] == w0, 'object2'] = w1
        frame_data['object'] = frame_data['object2'].copy()
        frame_data = frame_data.drop('object2', axis=1)
        
        # Store in dict if confirmed
        if confirm:
            print "storing confirmed result"
            curated_results_d[key_frame] = frame_data[['streak', 'object']].copy()
        
        # Go to next or previous frame, or quit
        if result in ['next', 'back']:
            frame_data = None
        
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
        
    return cres