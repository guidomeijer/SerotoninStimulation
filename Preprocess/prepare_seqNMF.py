#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:20:14 2023
By: Guido Meijer
"""

from ibllib.atlas import AllenAtlas
from one.api import ONE
from stim_functions import (paths, query_ephys_sessions, load_passive_opto_times, high_level_regions,
                            combine_regions, remap)
from brainbox.singlecell import calculate_peths
from brainbox.io.one import SpikeSortingLoader
from scipy.io import savemat
import pandas as pd
from os.path import join
import numpy as np
ba = AllenAtlas()
one = ONE()

# Settings
BIN_SIZE = 0.01  # s
SMOOTHING = 0.1  # s
MIN_NEURONS = 5
INCL_NEURONS = 'all'  # all, sig or non-sig

# Get paths
_, data_path = paths()
_, save_path = paths(save_dir='cache')

# Query sessions
rec = query_ephys_sessions(one=one)

# Get significantly modulated neurons
light_neurons = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))

for i, eid in enumerate(np.unique(rec['eid'])):

    # Get session details
    subject = one.get_details(eid)['subject']
    date = str(one.get_details(eid)['date'])
    print(f'\nStarting {subject}, {date} ({i+1} of {len(np.unique(rec["eid"]))})')

    # Load in laser pulse times
    opto_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_times) == 0:
        print('Could not load light pulses')
        continue

    # Get the insertions from the session
    pids = rec.loc[rec['eid'] == eid, 'pid'].values
    probes = rec.loc[rec['eid'] == eid, 'probe'].values

    firing_rate, acronyms, merged_regions, high_regions = [], [], [], []
    spikes, clusters, channels = dict(), dict(), dict()
    for (pid, probe) in zip(pids, probes):

        # Load in spikes
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes[probe], clusters[probe], channels[probe] = sl.load_spike_sorting()
        clusters[probe] = sl.merge_clusters(spikes[probe], clusters[probe], channels[probe])

        # Select neurons to use
        if INCL_NEURONS == 'all':
            use_neurons = light_neurons.loc[light_neurons['pid'] == pid, 'neuron_id'].values
        elif INCL_NEURONS == 'sig':
            use_neurons = light_neurons.loc[(light_neurons['pid'] == pid) & light_neurons['modulated'],
                                            'neuron_id'].values
        elif INCL_NEURONS == 'non-sig':
            use_neurons = light_neurons.loc[(light_neurons['pid'] == pid) & ~light_neurons['modulated'],
                                            'neuron_id'].values

        # Get binned spikes centered at stimulation onset
        peth, _ = calculate_peths(spikes[probe].times, spikes[probe].clusters, use_neurons,
                                  [opto_times[0]],
                                  pre_time=5*60, post_time=10*60,
                                  bin_size=BIN_SIZE, smoothing=SMOOTHING, return_fr=False)

        firing_rate.append(peth['means'])
        time = peth['tscale']

        # Get brain regions
        beryl_acronyms = remap(clusters[probe]['acronym'][use_neurons])
        acronyms.append(beryl_acronyms)
        merged_regions.append(combine_regions(beryl_acronyms))
        high_regions.append(high_level_regions(clusters[probe]['acronym'][use_neurons]))

    # Stack together spike bin matrices from the two recordings
    if len(pids) == 2:
        firing_rate = np.concatenate(firing_rate, axis=0)
        acronyms = np.concatenate(acronyms)
        merged_regions = np.concatenate(merged_regions)
        high_regions = np.concatenate(high_regions)
    elif len(pids) == 1:
        firing_rate = firing_rate[0]
        acronyms = acronyms[0]
        merged_regions = merged_regions[0]
        high_regions = high_regions[0]

    # Set negative values to 0
    firing_rate[firing_rate < 0] = 0

    # Save firing rates as MATLAB readable file for seqNMF
    savemat(join(save_path, 'seqNMF', f'{subject}_{date}.mat'),
            dict({'firing_rate': firing_rate,
                  'time_ax': time,
                  'opto_times': opto_times,
                  'eid': eid,
                  'subject': subject,
                  'date': date,
                  'acronyms': acronyms,
                  'merged_regions': merged_regions,
                  'high_regions': high_regions
                  }))
