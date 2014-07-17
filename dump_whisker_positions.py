# Load whisker traces from 0509A
# Plot an overall motion trace

# Whisk imports
import sys
sys.path.insert(1, 
    '/home/chris/Downloads/whisk-1.1.0d-Linux/share/whisk/python')
import traj
import trace

# Other imports
import matplotlib.pyplot as plt
import numpy as np, os.path, pandas

# Data location
session = '0509A_cropped_truncated_4'; side = 'top'
#~ session = '0527_cropped_truncated_5'; side = 'left'
whisk_rootdir = os.path.expanduser('~/mnt/bruno-nix/whisker_video/processed')
whisk_file = os.path.join(whisk_rootdir, session, session + '.whiskers')
measure_file = os.path.join(whisk_rootdir, session, session + '.measurements')

# Load the traces
frame2segment_id2whisker_seg = trace.Load_Whiskers(whisk_file)

# Load the correspondence between traces and identified whiskers
tmt = traj.MeasurementsTable(measure_file)
whisker_id2frame2segment_id = tmt.get_trajectories()


# Identify any missing frames
frames_l = frame2segment_id2whisker_seg.keys()
sorted_frames = np.sort(frames_l)

# Iterate over frames
rec_l = []

# Iterate over whiskers
# It looks like it numbers them from Bottom to Top for side == 'left'
# whiski colors them R, G, B
for wid, frame2segment_id in whisker_id2frame2segment_id.items():
    # Iterate over frames
    for frame, segment_id in frame2segment_id.items():
        # Get the actual segment for this whisker and frame
        ws = frame2segment_id2whisker_seg[frame][segment_id]
        
        # fit a line and calculate angle of whisker
        # in the video, (0, 0) is upper left, so we need to take negative of slope
        # This will fail for slopes close to vertical, for instance if it
        # has this shape: (  because least-squares fails here
        # eg Frame 5328 in 0509A_cropped_truncated_4
        p = np.polyfit(ws.x, ws.y, deg=1)
        slope = -p[0]

        # Arctan gives values between -90 and 90
        # Basically, we cannot discriminate a SSW whisker from a NNE whisker
        # Can't simply use diff_x because the endpoints can be noisy
        # Similar problem occurs with ESE and WNW, and then diff_y is noisy
        # Easiest way to do it is just pin the data to a known range
        angle = np.arctan(slope) * 180 / np.pi
        
        # side = left, so theta ~-90 to +90
        # side = top, so theta ~ -180 to 0
        if side == 'top':
            if angle > 0:
                angle = angle - 180
        elif side == 'left':
            if angle > 90:
                angle = angle - 180

        # Separate angle measurement: tip vs follicle
        # This will be noisier
        # Remember to flip up/down here
        # Also remember that ws.x and ws.y go from tip to follicle (I think?)
        # Actually the go from tip to follicle in one video and from follicle
        # to tip in the other; and then occasional exceptions on individual frames
        angle2 = np.arctan2(-(ws.y[0] - ws.y[-1]), ws.x[0] - ws.x[-1]) * 180 / np.pi

        # On rare occasions it seems to be flipped, 
        # eg Frame 9 in 0509A_cropped_truncated_4
        # So apply the same fix, even though it shouldn't be necessary here
        if side == 'top':
            if angle2 > 0:
                angle2 = angle2 - 180
        elif side == 'left':
            if angle2 > 90:
                angle2 = angle2 - 180

        # Store
        rec_l.append({'frame': frame, 'wid': wid, 'angle': angle, 'angle2': angle2})

# DataFrame it
angl_df = pandas.DataFrame.from_records(rec_l)
#~ piv_angle = angl_df.pivot_table(rows='frame', cols='wid', values=['angle', 'angle2'])

# Save
angl_df.save('angles_%s' % session)

