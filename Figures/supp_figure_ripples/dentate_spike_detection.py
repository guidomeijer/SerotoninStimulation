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
OVERWRITE = True
FILT_ORDER = 1
LOW_CUT = 5
HIGH_CUT = 100
FS = 2500

SCALEBAR = 0.5
MIN_TIME_BETWEEN = 0.1  # s
BIN_SIZE = 0.25
SMOOTHING = 0.1
colors, dpi = figure_style()
    
# Query sessions
rec = query_ephys_sessions(acronym='DG', one=one)

if OVERWRITE:
    dg_spike_df = pd.DataFrame()
else:
    dg_spike_df = pd.read_csv(join(save_path, 'ripple_freq.csv'))
    rec = rec[~rec['pid'].isin(dg_spike_df['pid'])]
    
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
    

    lf_psd = one.load_object(eid, 'ephysSpectralDensityLF', collection=f'raw_ephys_data/{probe}')
    dg_channels = np.where(channels[probe]['acronym'] == 'DG-mo')[0]  
    dg_ch_power = np.mean(lf_psd['power'][np.ix_((lf_psd['freqs'] >= LOW_CUT) & (lf_psd['freqs'] <= HIGH_CUT),
                                                  dg_channels)], axis=0)
    
    # Get the channel with the highest power in the freq band of interest
    #use_ch = dg_channels[np.argmax(channels[probe]['axial_um'][dg_channels])]
    use_ch = dg_channels[np.argmax(dg_ch_power)]
        
    # Filter
    b, a = butter(FILT_ORDER, [LOW_CUT, HIGH_CUT], fs=FS, btype='band')
    filt_lfp = filtfilt(b, a, this_lfp[use_ch, :])
    
    # Detect dentate spikes
    std_cross = time[filt_lfp > np.std(filt_lfp)*4.5]
    if std_cross.shape[0] == 0:
        continue
    dg_spike_start = std_cross[np.concatenate(([True], np.diff(std_cross) > MIN_TIME_BETWEEN))]
    
    # Get ripple rate
    peths, _ = calculate_peths(dg_spike_start, np.ones(dg_spike_start.shape[0]), [1], opto_times,
                               1, 4, BIN_SIZE, SMOOTHING)
    
    # Add to dataframe
    dg_spike_df = pd.concat((dg_spike_df, pd.DataFrame(data={
        'dg_spike_freq': peths['means'][0], 'time': peths['tscale'],
        'subject': subject, 'date': date, 'probe': probe, 'pid': pid})))
    dg_spike_df.to_csv(join(save_path, 'dg_spike_freq.csv'))
   
    
