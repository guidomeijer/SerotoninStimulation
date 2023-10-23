#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:20:14 2023
By: Guido Meijer
"""

from ibllib.atlas import AllenAtlas
from stim_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times, init_one,
                            high_level_regions, figure_style, N_STATES)
from brainbox.singlecell import calculate_peths
from scipy.ndimage import gaussian_filter
from brainbox.io.one import SpikeSortingLoader
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
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
INCL_NEURONS = 'all'  # all, sig or non-sig
RANDOM_TIMES = 'spont'  # spont (spontaneous) or jitter (jittered times during stim period)
PRE_TIME = 1  # final time window to use
POST_TIME = 4
HMM_PRE_TIME = 2  # time window to run HMM on
HMM_POST_TIME = 5
MIN_NEURONS = 5
CMAP = 'Set2'
PTRANS_SMOOTH = BIN_SIZE
OVERWRITE = True
PLOT = False
N_STATE_SELECT = 'global'  # global or region

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Awake',
                f'{INCL_NEURONS}', f'{int(BIN_SIZE*1000)}ms')

# Query sessions
rec = query_ephys_sessions(one=one)

# Get significantly modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

p_state_df = pd.DataFrame()

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

    # Generate random times during spontaneous activity
    if RANDOM_TIMES == 'jitter':
        random_times = np.sort(np.random.uniform(opto_times[0]-HMM_PRE_TIME, opto_times[-1]+HMM_POST_TIME,
                                                 size=opto_times.shape[0]))
    elif RANDOM_TIMES == 'spont':
        random_times = np.sort(np.random.uniform(opto_times[0]-360, opto_times[0]-10,
                                                 size=opto_times.shape[0]))

    # Get data from both probes and merge
    pids = rec.loc[rec['eid'] == eid, 'pid'].values
    probes = rec.loc[rec['eid'] == eid, 'probe'].values
    binned_spikes = []
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

        # Select QC pass neurons
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, use_neurons)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(
            spikes[probe].clusters, use_neurons)]
        use_neurons = use_neurons[np.isin(use_neurons, np.unique(spikes[probe].clusters))]

        # Get binned spikes centered at stimulation onset
        peth, these_spikes = calculate_peths(spikes[probe].times, spikes[probe].clusters, use_neurons,
                                             opto_times, pre_time=HMM_PRE_TIME, post_time=HMM_POST_TIME,
                                             bin_size=BIN_SIZE, smoothing=0, return_fr=False)
        binned_spikes.append(these_spikes)
        full_time_ax = peth['tscale']
        use_timepoints = (full_time_ax > -PRE_TIME) & (full_time_ax < POST_TIME)
        time_ax = full_time_ax[use_timepoints]

    # Stack together spike bin matrices from the two recordings
    if len(pids) == 2:
        binned_spikes = np.concatenate(binned_spikes, axis=1)
    elif len(pids) == 1:
        binned_spikes = binned_spikes[0]
    binned_spikes = binned_spikes.astype(int)

      
    full_time_ax = peth['tscale']
    use_timepoints = (full_time_ax > -PRE_TIME) & (full_time_ax < POST_TIME)
    time_ax = full_time_ax[use_timepoints]

    # Create list of (time_bins x neurons) per stimulation trial
    trial_data = []
    for j in range(binned_spikes.shape[0]):
        trial_data.append(np.transpose(binned_spikes[j, :, :]))

    # Initialize HMM
    simple_hmm = ssm.HMM(N_STATES, binned_spikes.shape[0], observations='poisson')
    lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')

    # Loop over trials
    trans_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
    state_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
    prob_mat = np.empty((len(trial_data), full_time_ax.shape[0], N_STATES))

    for t in range(len(trial_data)):

        # Get most likely states for this trial
        zhat = simple_hmm.most_likely_states(trial_data[t])
        prob_mat[t, :, :] = simple_hmm.filter(trial_data[t])

        # Add state to state matrix
        state_mat[t, :] = zhat

    # Select time period to use
    prob_mat = prob_mat[:, np.concatenate(([False], use_timepoints[:-1])), :]
    state_mat = state_mat[:, use_timepoints]

    # Get P(state)
    p_state_mat = np.empty((N_STATES, time_ax.shape[0]))
    for ii in range(N_STATES):

        # Random times
        # Get P state, first smooth, then crop timewindow
        this_p_state = np.mean(prob_mat[:random_times.shape[0], :, ii], axis=0)
        p_state_bl = this_p_state - np.mean(this_p_state[time_ax < 0])

        # Add to dataframe and matrix
        p_state_mat[ii, :] = this_p_state
        p_state_df = pd.concat((p_state_df, pd.DataFrame(data={
            'p_state': this_p_state, 'p_state_bl': p_state_bl, 'state': ii, 'time': time_ax,
            'subject': subject, 'pid': pid, 'opto': 0})))

        # Opto times
        # Get P state, first smooth, then crop timewindow
        this_p_state = np.mean(prob_mat[opto_times.shape[0]:, :, ii], axis=0)
        p_state_bl = this_p_state - np.mean(this_p_state[time_ax < 0])

        # Add to dataframe and matrix
        p_state_df = pd.concat((p_state_df, pd.DataFrame(data={
            'p_state': this_p_state, 'p_state_bl': p_state_bl, 'state': ii, 'time': time_ax,
            'subject': subject, 'pid': pid, 'opto': 1})))

    # Save the trial-level P(state) data and zhat matrix
    np.save(join(save_path, 'HMM', 'PassiveEventAllNeurons', f'{RANDOM_TIMES}', 'prob_mat',
                 f'{subject}_{date}_{probe}.npy'), prob_mat)
    np.save(join(save_path, 'HMM', 'PassiveEventAllNeurons', f'{RANDOM_TIMES}', 'state_mat',
                 f'{subject}_{date}_{probe}.npy'), state_mat)
    np.save(join(save_path, 'HMM', 'PassiveEventAllNeurons', f'{RANDOM_TIMES}', 'log_lambdas',
                 f'{subject}_{date}_{probe}.npy'), simple_hmm.observations.log_lambdas)
