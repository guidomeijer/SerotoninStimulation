# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:13:11 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import brainbox.io.one as bbone
from scipy.signal import butter, filtfilt
from brainbox.plot import peri_event_time_histogram
from brainbox.singlecell import calculate_peths
from matplotlib.patches import Rectangle
from stim_functions import (query_ephys_sessions, init_one, load_passive_opto_times, figure_style,
                            paths)
from serotonin_functions import load_lfp
one = init_one()

# Set paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'Ripples')

# Settings
OVERWRITE = False
FILT_ORDER = 5
LOW_CUT = 150
HIGH_CUT = 250
FS = 2500
SCALEBAR = 0.5
colors, dpi = figure_style()
    
# Query sessions
rec = query_ephys_sessions(acronym='CA1', one=one)

if OVERWRITE:
    ripple_df = pd.DataFrame()
else:
    ripple_df = pd.read_csv(join(save_path, 'ripple_freq.csv'))
    rec = rec[~rec['pid'].isin(ripple_df['pid'])]
    
for i in rec.index:
        
    # Get data
    eid = rec.loc[i, 'eid']
    probe = rec.loc[i, 'probe']
    subject = rec.loc[i, 'subject']
    date = rec.loc[i, 'date']
    pid = rec.loc[i, 'pid']

    # Load in opto times
    opto_times, _ = load_passive_opto_times(eid, one=one)

    # Load in channels
    channels = bbone.load_channel_locations(eid, probe=probe, one=one)
    
    # Load in LFP
    this_lfp, time = load_lfp(rec.loc[i, 'eid'], rec.loc[i, 'probe'],
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
    ripple_start = std_cross[np.concatenate(([True], np.diff(std_cross) > 0.01))]
    
    # Get ripple rate
    peths, _ = calculate_peths(ripple_start, np.ones(ripple_start.shape[0]), [1], opto_times,
                               1, 4, 0.5, 0.1)
    
    # Plot
    p, ax = plt.subplots(1, 1, figsize=(1.25, 1.75), dpi=dpi)
    ax.add_patch(Rectangle((0, -100), 1, 200, color='royalblue', alpha=0.25, lw=0))
    peri_event_time_histogram(ripple_start, np.ones(ripple_start.shape[0]), opto_times,
                              1, t_before=1, t_after=4, bin_size=0.5, smoothing=0.1,
                              include_raster=True, error_bars='sem', ax=ax,
                              pethline_kwargs={'color': 'black', 'lw': 1},
                              errbar_kwargs={'color': 'black', 'alpha': 0.3, 'lw': 0},
                              raster_kwargs={'color': 'black', 'lw': 0.5},
                              eventline_kwargs={'lw': 0})
    ax.plot([-1.05, -1.05], [0, SCALEBAR], color='k', lw=0.75, clip_on=False)
    ax.text(-1.05, SCALEBAR/2, f'{SCALEBAR} ripples s$^{-1}$', ha='right', va='center', rotation=90)
    ax.plot([0, 1], [ax.get_ylim()[0]-0.02, ax.get_ylim()[0]-0.02], color='k', lw=0.75, clip_on=False)
    ax.text(0.5, ax.get_ylim()[0]-0.03, '1s', ha='center', va='top')
    ax.axis('off')
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(join(fig_path, f'{subject}_{date}_{probe}.jpg'), dpi=600)
    plt.close(p)
    
    # Add to dataframe
    ripple_df = pd.concat((ripple_df, pd.DataFrame(data={
        'ripple_freq': peths['means'][0], 'time': peths['tscale'],
        'subject': subject, 'date': date, 'probe': probe, 'pid': pid})))
    ripple_df.to_csv(join(save_path, 'ripple_freq.csv'))
   
    
