#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:20:14 2023
By: Guido Meijer
"""

from ibllib.atlas import AllenAtlas
from stim_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times, init_one,
                            high_level_regions, N_STATES_CONT)
from brainbox.singlecell import calculate_peths
from brainbox.io.one import SpikeSortingLoader
import pandas as pd
from os.path import join
import ssm
import numpy as np
np.random.seed(0)
ba = AllenAtlas()
one = init_one()

# Settings
BIN_SIZE = 0.1  # s
MIN_NEURONS = 5
CMAP = 'Set2'
PTRANS_SMOOTH = BIN_SIZE
OVERWRITE = True
PLOT = False

# Get paths
f_path, s_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)

# Get significantly modulated neurons
light_neurons = pd.read_csv(join(s_path, 'light_modulated_neurons.csv'))

if OVERWRITE:
    log_likelihood_df = pd.DataFrame()
else:
    log_likelihood_df = pd.read_csv(
        join(s_path, f'hmm_passive_continuous_ll_{BIN_SIZE*1000}ms_bins.csv'))
    rec = rec[~rec['pid'].isin(log_likelihood_df['pid'])]
save_path = join(s_path, 'HMM', 'PassiveContinuous')

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

    print(f'\nStarting {subject}, {date} ({i+1} of {rec.shape[0]})')

    # Load in laser pulse times
    opto_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_times) == 0:
        print('Could not load light pulses')
        continue

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    # Select neurons
    use_neurons = light_neurons.loc[light_neurons['pid'] == pid, 'neuron_id'].values
    spikes.times = spikes.times[np.isin(spikes.clusters, use_neurons)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, use_neurons)]
    use_neurons = use_neurons[np.isin(use_neurons, np.unique(spikes.clusters))]

    # Get regions from Beryl atlas
    clusters['region'] = remap(clusters['acronym'], combine=True)
    clusters['high_level_region'] = high_level_regions(clusters['acronym'])
    clusters_regions = clusters['high_level_region'][use_neurons]

    # Loop over regions
    for r, region in enumerate(np.unique(clusters['high_level_region'])):
        if region == 'root':
            continue

        # Select spikes and clusters in this brain region
        clusters_in_region = use_neurons[clusters_regions == region]

        if len(clusters_in_region) < MIN_NEURONS:
            continue

        # Get binned spikes centered at first stimulation
        peth, binned_spikes = calculate_peths(spikes.times, spikes.clusters, clusters_in_region,
                                              [opto_times[0]],
                                              pre_time=5*60, post_time=10*60,
                                              bin_size=BIN_SIZE, smoothing=0, return_fr=False)
        binned_spikes = np.squeeze(binned_spikes.astype(int)).T
        time_ax = peth['tscale']
        time_ax = time_ax + opto_times[0]

        # Check if all opto pulses are in timewindow
        if (opto_times - opto_times[0])[-1] > peth['tscale'][-1]:
            print('Opto pulses do not fit in timewindow!')

        # Fit HMM
        simple_hmm = ssm.HMM(N_STATES_CONT, binned_spikes.shape[1], observations='poisson')
        lls = simple_hmm.fit(binned_spikes, method='em', transitions='sticky')
        zhat = simple_hmm.most_likely_states(binned_spikes)
        prob = simple_hmm.filter(binned_spikes)

        # Save output
        np.save(join(save_path, f'{subject}_{date}_{probe}_{region}_zhat.npy'), zhat)
        np.save(join(save_path, f'{subject}_{date}_{probe}_{region}_prob.npy'), prob)
        np.save(join(save_path, f'{subject}_{date}_{probe}_{region}_time.npy'), time_ax)
