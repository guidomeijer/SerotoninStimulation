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
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import FormatStrFormatter
from brainbox.plot import peri_event_time_histogram
from brainbox.singlecell import calculate_peths
import neo
from elephant.gpfa import GPFA
import quantities as pq
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_neuron_qc, calculate_peths,
                                 high_level_regions, figure_style, N_STATES)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
BIN_SIZE = 0.1  # s
INCL_NEURONS = 'all'  # all, sig or non-sig
PRE_TIME = 1  # final time window to use
POST_TIME = 4
HMM_PRE_TIME = 2  # time window to run HMM on
HMM_POST_TIME = 5
MIN_NEURONS = 10
CMAP = 'Set3'
N_GPFA_DIM = 10
PTRANS_SMOOTH = BIN_SIZE
PSTATE_SMOOTH = BIN_SIZE
OVERWRITE = True

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'AwakeGPFA', f'{INCL_NEURONS}', f'{int(BIN_SIZE*1000)}ms')

# Query sessions
rec = query_ephys_sessions(one=one)

# Get significantly modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Initialize GPFA
gpfa = GPFA(bin_size=BIN_SIZE * pq.s, x_dim=N_GPFA_DIM)

# Get time axis
time_ax = np.arange(-HMM_PRE_TIME + (BIN_SIZE/2), HMM_POST_TIME + BIN_SIZE - (BIN_SIZE/2), BIN_SIZE)
use_timepoints = (time_ax > -PRE_TIME) & (time_ax < POST_TIME)
time_ax = time_ax[use_timepoints]

if OVERWRITE:
    state_trans_df = pd.DataFrame()
    p_state_df = pd.DataFrame()
else:
    state_trans_df = pd.read_csv(join(save_path, f'all_state_trans_GPFA_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    p_state_df = pd.read_csv(join(save_path, f'p_state_GPFA_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    rec = rec[~rec['pid'].isin(state_trans_df['pid'])]

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
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

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

        # Initialize HMM & GPFA
        simple_hmm = ssm.HMM(N_STATES[region], N_GPFA_DIM, observations='gaussian')

        # Transform spikes into Elephant lists of neo.SpikeTrain objects
        print('Transforming spike trains..')
        trial_data = []
        for j, this_time in enumerate(opto_times):
            this_trial_data = []
            for jj, neuron_id in enumerate(clusters_in_region):
                trial_spikes = spikes.times[(spikes.clusters == neuron_id)
                                            & (spikes.times > this_time - HMM_PRE_TIME)
                                            & (spikes.times < this_time + HMM_POST_TIME)] - this_time
                this_trial_data.append(neo.SpikeTrain(trial_spikes, HMM_POST_TIME, units='s',
                                                      t_start=-HMM_PRE_TIME))
            trial_data.append(this_trial_data)

        # Fit GPFA
        trajectories = gpfa.fit_transform(trial_data)

        # Transpose
        for j in range(len(trajectories)):
            trajectories[j] = trajectories[j].T

        # Fit HMM
        lls = simple_hmm.fit(list(trajectories), method='em', transitions='sticky')

        # Loop over trials
        trans_mat = np.empty((len(trajectories), trajectories[0].shape[0]))
        state_mat = np.empty((len(trajectories), trajectories[0].shape[0]))
        for t in range(len(trajectories)):

            # Get most likely states for this trial
            zhat = simple_hmm.most_likely_states(trajectories[t])

            # Get state transitions times
            trans_mat[t, :] = np.concatenate((np.diff(zhat) > 0, [False])).astype(int)

            # Add state to state matrix
            state_mat[t, :] = zhat

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
                'subject': subject, 'pid': pid, 'region': region})))

        # Add state change PSTH to dataframe
        state_trans_df = pd.concat((state_trans_df, pd.DataFrame(data={
            'time': time_ax, 'p_trans': smooth_p_trans,
            'p_trans_bl': smooth_p_trans - np.mean(smooth_p_trans[time_ax < 0]),
            'region': region, 'subject': subject, 'pid': pid})))

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
            idx = np.bitwise_and(spikes.times[spikes.clusters == n] >= opto_times[trial] - PRE_TIME,
                                 spikes.times[spikes.clusters == n] <= opto_times[trial] + POST_TIME)
            neuron_spks = spikes.times[spikes.clusters == n][idx]
            ax.vlines(neuron_spks - opto_times[trial], tickedges[k + 1], tickedges[k], color='black',
                      lw=0.4, zorder=1)
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
                  extent=(-PRE_TIME, POST_TIME, 1, len(opto_times)), interpolation=None)
        ax1.plot([0, 0], [1, len(opto_times)], ls='--', color='k', lw=0.75)
        ax1.set(ylabel='Trials', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4],
               title=f'{region}')

        for ii in range(N_STATES[region]):
            ax2.plot(time_ax, p_state_mat[ii, :], color=cmap[ii])
        ax2.set(xlabel='Time (s)', ylabel='P(state)', xticks=[-1, 0, 1, 2, 3, 4])

        ax3.imshow(trans_mat, aspect='auto', cmap='Greys', interpolation=None,
                   extent=(-PRE_TIME, POST_TIME, 1, len(opto_times)))
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
    state_trans_df.to_csv(join(save_path, f'all_state_trans_GPFA_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    p_state_df.to_csv(join(save_path, f'p_state_GPFA_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))





