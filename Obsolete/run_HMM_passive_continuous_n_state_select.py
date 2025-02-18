#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:20:14 2023
By: Guido Meijer
"""

from ibllib.atlas import AllenAtlas
from stim_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times, init_one,
                            high_level_regions, figure_style)
from brainbox.singlecell import calculate_peths
from scipy.ndimage import gaussian_filter
from brainbox.io.one import SpikeSortingLoader
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join
import ssm
import numpy as np
np.random.seed(0)
ba = AllenAtlas()
one = init_one()

# Settings
BIN_SIZE = 0.1  # s
PRE_TIME = 1  # final time window to use
POST_TIME = 4
HMM_PRE_TIME = 2  # time window to run HMM on
HMM_POST_TIME = 5
MIN_NEURONS = 5
CMAP = 'Set2'
N_STATES = np.arange(2, 21)
PTRANS_SMOOTH = BIN_SIZE
OVERWRITE = True
PLOT = False
N_STATE_SELECT = 'global'  # global or region
K_FOLDS = 10

# Get paths
f_path, save_path = paths()

# Initialize k-fold cross validation
kf = KFold(n_splits=K_FOLDS, shuffle=False)

# Query sessions
rec = query_ephys_sessions(one=one)

# Get significantly modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

if OVERWRITE:
    log_likelihood_df = pd.DataFrame()
else:
    log_likelihood_df = pd.read_csv(
        join(save_path, f'hmm_passive_continuous_ll_{BIN_SIZE*1000}ms_bins.csv'))
    rec = rec[~rec['pid'].isin(log_likelihood_df['pid'])]

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

        # Check if all opto pulses are in timewindow
        if (opto_times - opto_times[0])[-1] > peth['tscale'][-1]:
            print('Opto pulses do not fit in timewindow!')

        # Loop over different number of states
        log_likelihood = np.empty(N_STATES.shape[0])
        for j, s in enumerate(N_STATES):
            print(f'Starting state {s} of {N_STATES[-1]}')

            # Cross validate
            train_index, test_index = next(kf.split(binned_spikes))

            # Fit HMM on training data
            simple_hmm = ssm.HMM(s, binned_spikes.shape[1], observations='poisson')
            lls = simple_hmm.fit(binned_spikes[train_index, :], method='em',
                                 transitions='sticky')

            # Get log-likelihood on test data
            log_likelihood[j] = simple_hmm.log_likelihood(binned_spikes[test_index, :])

        # Add to dataframe
        log_likelihood_df = pd.concat((log_likelihood_df, pd.DataFrame(data={
            'log_likelihood': log_likelihood, 'n_states': N_STATES, 'subject': subject, 'pid': pid,
            'region': region})))

        # Save result
        log_likelihood_df.to_csv(
            join(save_path, f'hmm_passive_continuous_ll_{int(BIN_SIZE*1000)}ms_bins.csv'))
