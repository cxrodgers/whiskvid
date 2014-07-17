# Dump frames from video, with and without the tracked whisker outline
# Whisk imports
import sys
sys.path.insert(1, 
    '/home/chris/Downloads/whisk-1.1.0d-Linux/share/whisk/python')
import traj
import trace

# Other imports
import matplotlib.pyplot as plt
import numpy as np, os.path, pandas, my, scipy.misc

# Data location
#~ session = '0509A_cropped_truncated_4'; side = 'top'
session = '0527_cropped_truncated_5'; side = 'left'
whisk_rootdir = os.path.expanduser('~/mnt/bruno-nix/whisker_video/processed')
whisk_file = os.path.join(whisk_rootdir, session, session + '.whiskers')
measure_file = os.path.join(whisk_rootdir, session, session + '.measurements')
video_file = os.path.join(whisk_rootdir, session, session + '.mp4')

# Also load angles
angl_df = pandas.load('angles_%s' % session)
piv_angle = angl_df.pivot_table(rows='frame', cols='wid', 
    values=['angle', 'angle2'])
metric = 'angle2'

# Where to put the resulting video
output_dir = os.path.join('/home/chris/mnt/bruno-nix/side_by_side', session)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Load the traces
frame2segment_id2whisker_seg = trace.Load_Whiskers(whisk_file)

# Load the correspondence between traces and identified whiskers
tmt = traj.MeasurementsTable(measure_file)
whisker_id2frame2segment_id = tmt.get_trajectories()

# Identify available frames
frames_l = frame2segment_id2whisker_seg.keys()
sorted_frames = np.sort(frames_l)

# Set the color cycle
if len(whisker_id2frame2segment_id) == 3:
    whiski_color_cycle = ['r', 'g', 'b']
else:
    whiski_color_cycle = ['r', 'y', 'g', 'cyan', 'b', 'magenta']

# Truncate
#~ sorted_frames = sorted_frames[sorted_frames >= 4950]

# Whether to show traces
SHOW_TRACES = False
STACKING = 'vertical'

# Create figure
if SHOW_TRACES:
    f, axa = plt.subplots(2, 1, figsize=(5, 8.2))
    f.subplots_adjust(left=.15, right=.95, top=.97, bottom=.05)
    ax = axa[0]

    # Also plot the angle
    axa[1].set_color_cycle(whiski_color_cycle)
    axa[1].set_xlabel('time (s)')
    axa[1].set_ylabel('whisker angle (degrees)')

    # Full time course
    frame_nums = piv_angle[metric].index.values
    axa[1].plot(frame_nums / 300., piv_angle[metric].ix[frame_nums])
    axa[1].set_ylim((-140, -80))

    # This will be used to show the current time
    time_line, = axa[1].plot([0, 0], [-140, -80], 'k')
else:
    f, ax = plt.subplots(figsize=(5, 6))


# Iterate over frames
frames_to_check = sorted_frames[
    (sorted_frames >= 300) &
    (sorted_frames < 900)]
for nframe, frame in enumerate(frames_to_check):
    # FFMPEG needs to know the time of the frame
    # It seems to round up, strangely, so subtract a ms and then round to the ms
    frametime = np.round((frame / 30.0) - .001, 3)
    
    # status
    print nframe, frame, frametime
    
    # Dump the frame
    my.misc.frame_dump(video_file, frametime=frametime, output_filename='out.png')
    
    # Load it
    im = scipy.misc.imread('out.png', flatten=True)
    
    # Create a copy with whisker trace overlaid
    if STACKING == 'horizontal':
        concatted = np.concatenate([im, im], axis=1)
    if STACKING == 'vertical':
        concatted = np.concatenate([im, im], axis=0)
    
    # Display concatted
    ax.imshow(concatted, interpolation='nearest', cmap=plt.cm.gray)
    for imobj in ax.get_images():
        imobj.set_clim((0, 255))

    # Also load and plot the whiskers from this frame
    for wid in whisker_id2frame2segment_id:
        try:
            segment_id = whisker_id2frame2segment_id[wid][frame]
        except KeyError:
            # can't find whisker in this frame
            continue
        ws = frame2segment_id2whisker_seg[frame][segment_id]
        ax.plot(ws.x, ws.y, whiski_color_cycle[wid])

    # Limits match axis
    ax.set_xlim((0, concatted.shape[1] - 1))
    ax.set_ylim((concatted.shape[0] - 1, 0))
    
    if SHOW_TRACES:
        # Update the traces
        axa[1].set_xlim((frame / 300. - 1.0, frame / 300. + 1.0))
        time_line.set_xdata(np.array([frame, frame]) / 300.)
    
    # Save and prepare to ffmpeg
    f.savefig(os.path.join(output_dir, '%06d.png' % nframe), dpi=200)

    # Clear the image
    ax.clear()

# And then run something like this on the server
# ffmpeg -r 30 -i %06d.png -y -vcodec mpeg4 out.avi
    