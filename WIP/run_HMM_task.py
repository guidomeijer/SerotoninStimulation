#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:20:14 2023
By: Guido Meijer
"""

import ssm
import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from brainbox.io.one import SpikeSortingLoader
from scipy.ndimage import gaussian_filter
from brainbox.singlecell import calculate_peths
from stim_functions import (paths, remap, query_ephys_sessions, load_trials, get_artifact_neurons,
                            get_neuron_qc, high_level_regions, figure_style, N_STATES)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
BIN_SIZE = 0.1  # s
PRE_TIME = 1  # final time window to use
POST_TIME = 4
HMM_PRE_TIME = 2  # time window to run HMM on
HMM_POST_TIME = 5
STATE_BINS = np.arange(-1, 4.5, 0.5)  # time bins to get max state in
MIN_NEURONS = 5
CMAP = 'Set3'
PTRANS_SMOOTH = BIN_SIZE
PSTATE_SMOOTH = BIN_SIZE
OVERWRITE = True
PLOT = False

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Task', f'{int(BIN_SIZE*1000)}ms')

# Query sessions
rec = query_ephys_sessions(n_trials=400, one=one)

# Get artifact neurons
artifact_neurons = get_artifact_neurons()

if OVERWRITE:
    p_state_df = pd.DataFrame()
    state_df = pd.DataFrame()
else:
    p_state_df = pd.read_csv(join(save_path, f'p_state_task_{int(BIN_SIZE*1000)}msbins.csv'))
    state_df = pd.read_csv(join(save_path, f'state_task_{int(BIN_SIZE*1000)}msbins.csv'))
    rec = rec[~rec['pid'].isin(p_state_df['pid'])]

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

    print(f'\nStarting {subject}, {date} ({i+1} of {rec.shape[0]})')

    # Load in trials
    trials = load_trials(eid, one=one)

    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    
    # Filter neurons that pass QC
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]

    # Exclude artifact neurons
    clusters_pass = np.array([i for i in clusters_pass if i not in artifact_neurons.loc[
        artifact_neurons['pid'] == pid, 'neuron_id'].values])
    if clusters_pass.shape[0] == 0:
            continue

    # Select QC pass neurons
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    clusters_pass = clusters_pass[np.isin(clusters_pass, np.unique(spikes.clusters))]

    # Get regions from Beryl atlas
    clusters['region'] = remap(clusters['acronym'], combine=True)
    clusters['high_level_region'] = high_level_regions(clusters['acronym'])
    clusters_regions = clusters['high_level_region'][clusters_pass]

    # Loop over regions
    for r, region in enumerate(np.unique(clusters['high_level_region'])):
        if region == 'root':
            continue

        # Select spikes and clusters in this brain region
        clusters_in_region = clusters_pass[clusters_regions == region]

        if len(clusters_in_region) < MIN_NEURONS:
            continue

        # Initialize HMM
        simple_hmm = ssm.HMM(N_STATES[region], clusters_in_region.shape[0], observations='poisson')

        # Get binned spikes centered at stimulation onset
        peth, binned_spikes = calculate_peths(spikes.times, spikes.clusters, clusters_in_region,
                                              trials['goCue_times'], pre_time=HMM_PRE_TIME,
                                              post_time=HMM_POST_TIME, bin_size=BIN_SIZE,
                                              smoothing=0, return_fr=False)
        binned_spikes = binned_spikes.astype(int)
        full_time_ax = peth['tscale']
        use_timepoints = (full_time_ax > -PRE_TIME) & (full_time_ax < POST_TIME)
        time_ax = full_time_ax[use_timepoints]

        # Create list of (time_bins x neurons) per trial
        trial_data = []
        for j in range(binned_spikes.shape[0]):
            trial_data.append(np.transpose(binned_spikes[j, :, :]))

        # Initialize HMM
        lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')

        # Loop over trials
        trans_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
        state_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
        for t in range(len(trial_data)):

            # Get most likely states for this trial
            zhat = simple_hmm.most_likely_states(trial_data[t])

            # Get state transitions times
            trans_mat[t, :] = np.concatenate((np.diff(zhat) > 0, [False])).astype(int)

            # Add state to state matrix
            state_mat[t, :] = zhat
            
            # Get max state per timebin
            binned_state = np.empty(STATE_BINS.shape[0] - 1).astype(int)
            for tt, time_bin in enumerate(STATE_BINS[:-1]):
                these_states, these_counts = np.unique(
                    zhat[(full_time_ax >= time_bin) & (full_time_ax <= STATE_BINS[tt+1])],
                    return_counts=True)
                binned_state[tt] = these_states[np.argmax(these_counts)]
            state_df = pd.concat((state_df, pd.DataFrame(data={
                'state': binned_state, 'time': STATE_BINS[:-1] + (np.diff(STATE_BINS) / 2),
                'trial': t, 'subject': subject, 'pid': pid, 'eid': eid, 'region': region})))

        # Smooth P(state change) over entire period
        p_trans = np.mean(trans_mat, axis=0)
        smooth_p_trans = gaussian_filter(p_trans, PTRANS_SMOOTH / BIN_SIZE)

        # Select time period to use
        trans_mat = trans_mat[:, use_timepoints]
        smooth_p_trans = smooth_p_trans[use_timepoints]

        # Get P(state)
        p_state_mat = np.empty((N_STATES[region], time_ax.shape[0]))
        for ii in range(N_STATES[region]):

            # Get P state, first smooth, then crop timewindow
            this_p_state = np.mean(state_mat == ii, axis=0)
            smooth_p_state = gaussian_filter(this_p_state, PSTATE_SMOOTH / BIN_SIZE)
            smooth_p_state = smooth_p_state[use_timepoints]

            # Add to dataframe and matrix
            p_state_mat[ii, :] = smooth_p_state
            p_state_df = pd.concat((p_state_df, pd.DataFrame(data={
                'p_state': smooth_p_state, 'state': ii, 'time': time_ax,
                'subject': subject, 'pid': pid, 'eid': eid, 'region': region})))
            
        if PLOT:
            # Plot example trial
            trial = 1
            cmap = sns.color_palette(CMAP, N_STATES[region])
            colors, dpi = figure_style()
            f, ax = plt.subplots(1, 1, figsize=(3.5, 1.75), dpi=dpi)
            ax.imshow(state_mat[trial, use_timepoints][None, :],
                      aspect='auto', cmap=ListedColormap(cmap), vmin=0, vmax=N_STATES[region]-1, alpha=0.4,
                      extent=(-PRE_TIME, POST_TIME, -1, len(clusters_in_region)+1))
            tickedges = np.arange(0, len(clusters_in_region)+1)
            for k, n in enumerate(clusters_in_region):
                idx = np.bitwise_and(spikes.times[spikes.clusters == n] >= trials.loc[trial, 'goCue_times'] - PRE_TIME,
                                     spikes.times[spikes.clusters == n] <= trials.loc[trial, 'goCue_times'] + POST_TIME)
                neuron_spks = spikes.times[spikes.clusters == n][idx]
                ax.vlines(neuron_spks - trials.loc[trial, 'goCue_times'], tickedges[k + 1], tickedges[k],
                          color='black', lw=0.4, zorder=1)
            ax.set(xlabel='Time (s)', ylabel='Neurons', yticks=[0, len(clusters_in_region)],
                   yticklabels=[1, len(clusters_in_region)], xticks=[-1, 0, 1, 2, 3, 4],
                   ylim=[-1, len(clusters_in_region)+1], title=f'{region}')
            sns.despine(trim=True)
            plt.tight_layout()
    
            plt.savefig(join(
                fig_path, f'{region}_{subject}_{date}_trial.jpg'),
                dpi=600)
            plt.close(f)
    
            # Plot session
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.25, 1.75), dpi=dpi)
            ax1.imshow(state_mat[:, use_timepoints], aspect='auto', cmap=ListedColormap(cmap),
                       vmin=0, vmax=N_STATES[region]-1,
                      extent=(-PRE_TIME, POST_TIME, 1, trials.shape[0]), interpolation=None)
            ax1.plot([0, 0], [1, trials.shape[0]], ls='--', color='k', lw=0.75)
            ax1.set(ylabel='Trials', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4],
                   title=f'{region}')
    
            for ii in range(N_STATES[region]):
                ax2.plot(time_ax, p_state_mat[ii, :], color=cmap[ii])
            ax2.set(xlabel='Time (s)', ylabel='P(state)', xticks=[-1, 0, 1, 2, 3, 4])
    
            ax3.imshow(trans_mat, aspect='auto', cmap='Greys', interpolation=None,
                       extent=(-PRE_TIME, POST_TIME, 1, trials.shape[0]))
            ax32 = ax3.twinx()
            ax32.plot(time_ax, smooth_p_trans)
            ax32.set(ylabel='P(state change)')
            ax3.set(ylabel='Trials', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4])
    
            #sns.despine(trim=True)
            plt.tight_layout()
            plt.savefig(join(
                fig_path, f'{region}_{subject}_{date}_ses.jpg'),
                dpi=600)
            plt.close(f)

    # Save output
    p_state_df.to_csv(join(save_path, f'p_state_task_{int(BIN_SIZE*1000)}msbins.csv'))
    state_df.to_csv(join(save_path, f'state_task_{int(BIN_SIZE*1000)}msbins.csv'))





