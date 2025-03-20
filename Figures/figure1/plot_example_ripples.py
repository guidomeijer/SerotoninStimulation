# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:32:56 2023

By Guido Meijer
"""
import numpy as np
import pandas as pd
from os.path import join, realpath, dirname, split
import matplotlib.pyplot as plt
import brainbox.io.one as bbone
from scipy.signal import butter, filtfilt
from brainbox.plot import peri_event_time_histogram
from brainbox.singlecell import calculate_peths
from matplotlib.patches import Rectangle
from stim_functions import (query_ephys_sessions, init_one, load_passive_opto_times, figure_style,
                            paths, load_lfp, plot_scalar_on_slice)
from iblatlas.atlas import AllenAtlas
one = init_one()
ba = AllenAtlas(res_um=10)

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Settings
PID = 'fb97e357-0c74-43a4-9556-1f18d6d63f9d'
eid, probe = one.pid2eid(PID)
FILT_ORDER = 5
LOW_CUT = 150
HIGH_CUT = 250
FS = 2500
SCALEBAR = 0.5
BIN_SIZE = 0.25
SMOOTHING = 0.1
 
# Load in opto times
opto_times, _ = load_passive_opto_times(eid, one=one)

# Load in channels
channels = bbone.load_channel_locations(eid, probe=probe, one=one)

# Load in LFP
this_lfp, time = load_lfp(eid, probe,
                          opto_times[0]-10, opto_times[-1]+4,
                          relative_to='begin', one=one)    

# Load in AP band RMS
rms_ap = one.load_object(eid, 'ephysChannels', collection=f'raw_ephys_data/{probe}',
                         attribute=['apRMS'])['apRMS'][0,:]


# Get channels in CA1
ca1_channels = np.where(channels[probe]['acronym'] == 'CA1')[0]

# Get max channel
max_ch = ca1_channels[np.argmax(rms_ap[ca1_channels])]

# Filter
b, a = butter(FILT_ORDER, [LOW_CUT, HIGH_CUT], fs=FS, btype='band')
filt_lfp = filtfilt(b, a, this_lfp[max_ch, :])

# Detect ripples
std_cross = time[filt_lfp > np.std(filt_lfp)*6]
ripple_start = std_cross[np.concatenate(([True], np.diff(std_cross) > 0.1))]

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
time_select = (time > ripple_start[12] - 1) & (time < ripple_start[12]+0.8)
ax1.scatter(ripple_start[10:14], [0.000075, 0.000075, 0.000075, 0.000075],
            marker='*', color='r', zorder=1)
ax1.plot(time[time_select], filt_lfp[time_select], color='k', lw=0.5, zorder=0)
ax1.plot([ax1.get_xlim()[0]+0.1, ax1.get_xlim()[0]+0.3], [-0.00008, -0.00008], color='k')
ax1.text(ax1.get_xlim()[0]+0.2, -0.000085, '0.2s', ha='center', va='top')
ax1.axis('off')

plt.subplots_adjust(bottom=0.1)
plt.tight_layout()
plt.savefig(join(fig_path, 'example_ripple_detection.pdf'))

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
plot_scalar_on_slice(np.array([]), np.array([]), ax=ax1, slice='coronal', coord=-2000, brain_atlas=ba, clevels=[0, 3])
ax1.axis('off')
plt.tight_layout()
plt.savefig(join(fig_path, 'example_brain_regions.jpg'), dpi=1000)

"""
ax2.add_patch(Rectangle((0, -100), 1, 200, color='royalblue', alpha=0.25, lw=0))
peri_event_time_histogram(ripple_start, np.ones(ripple_start.shape[0]), opto_times,
                          1, t_before=1, t_after=4, bin_size=BIN_SIZE, smoothing=SMOOTHING,
                          include_raster=True, error_bars='sem', ax=ax2,
                          pethline_kwargs={'color': 'black', 'lw': 1},
                          errbar_kwargs={'color': 'black', 'alpha': 0.3, 'lw': 0},
                          raster_kwargs={'color': 'black', 'lw': 1},
                          eventline_kwargs={'lw': 0})
ax2.plot([-1.05, -1.05], [0, SCALEBAR], color='k', lw=0.75, clip_on=False)
ax2.text(-1.05, SCALEBAR/2, f'{SCALEBAR} ripples s$^{-1}$', ha='right', va='center', rotation=90)
ax2.plot([0, 1], [ax2.get_ylim()[0]-0.02, ax2.get_ylim()[0]-0.02], color='k', lw=0.75, clip_on=False)
ax2.text(0.5, ax2.get_ylim()[0]-0.03, '1s', ha='center', va='top')
ax2.axis('off')
"""



