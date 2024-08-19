#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:20:14 2023
By: Guido Meijer
"""

from iblatlas.atlas import AllenAtlas
from stim_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times, init_one,
                            combine_regions, figure_style, N_STATES_REGIONS, N_STATES)
from brainbox.singlecell import calculate_peths
from scipy.ndimage import gaussian_filter
from brainbox.io.one import SpikeSortingLoader
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pickle
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
PRE_TIME = 2  # time window to run HMM on
POST_TIME = 5
MIN_NEURONS = 5
CMAP = 'Set2'
PTRANS_SMOOTH = BIN_SIZE
OVERWRITE = True
PLOT = False
N_STATE_SELECT = 'global'  # global or region

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Awake')

# Query sessions
rec = query_ephys_sessions(one=one)

# Get significantly modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

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
    clusters['region'] = remap(clusters['acronym'])
    clusters['full_region'] = combine_regions(clusters['region'], abbreviate=True)
    clusters_regions = clusters['full_region'][use_neurons]

    # Loop over regions
    for r, region in enumerate(np.unique(clusters['full_region'])):
        if region == 'root':
            continue

        if N_STATE_SELECT == 'global':
            n_states = N_STATES
        elif N_STATE_SELECT == 'region':
            n_states = N_STATES_REGIONS[region]

        # Select spikes and clusters in this brain region
        clusters_in_region = use_neurons[clusters_regions == region]
        print(f'Region {region}, {clusters_in_region.shape[0]} neurons')
        if len(clusters_in_region) < MIN_NEURONS:
            continue

        # Initialize HMM
        simple_hmm = ssm.HMM(n_states, clusters_in_region.shape[0], observations='poisson')

        # Get binned spikes centered at stimulation onset
        peth, binned_spikes = calculate_peths(spikes.times, spikes.clusters, clusters_in_region,
                                              opto_times,
                                              pre_time=PRE_TIME, post_time=POST_TIME,
                                              bin_size=BIN_SIZE, smoothing=0, return_fr=False)
        binned_spikes = binned_spikes.astype(int)
        time_ax = peth['tscale']

        # Create list of (time_bins x neurons) per stimulation trial
        trial_data = []
        for j in range(binned_spikes.shape[0]):
            trial_data.append(np.transpose(binned_spikes[j, :, :]))

        # Initialize HMM
        lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')

        # Loop over trials
        trans_mat = np.empty((len(trial_data), time_ax.shape[0])).astype(int)
        state_mat = np.empty((len(trial_data), time_ax.shape[0])).astype(int)
        prob_mat = np.empty((len(trial_data), time_ax.shape[0], n_states))

        for t in range(len(trial_data)):

            # Get most likely states for this trial
            zhat = simple_hmm.most_likely_states(trial_data[t])
            prob_mat[t, :, :] = simple_hmm.filter(trial_data[t])

            # Get state transitions times
            trans_mat[t, :] = np.concatenate((np.diff(zhat) > 0, [False])).astype(int)

            # Add state to state matrix
            state_mat[t, :] = zhat

        # Smooth P(state change) over entire period
        p_trans = np.mean(trans_mat, axis=0)
        smooth_p_trans = gaussian_filter(p_trans, PTRANS_SMOOTH / BIN_SIZE)

        # Save the trial-level P(state) data and zhat matrix
        hmm_dict = dict()
        hmm_dict['prob_mat'] = prob_mat
        hmm_dict['state_mat'] = state_mat
        hmm_dict['time_ax'] = time_ax
        hmm_dict['event_times'] = opto_times
        hmm_dict['eid'] = eid
        hmm_dict['pid'] = pid
        with open(join(save_path, 'HMM', 'PassiveEvent', f'{subject}_{date}_{region}.pickle'),
                  'wb') as fp:
            pickle.dump(hmm_dict, fp)
    

        if PLOT:
            # Plot example trial
            trial = 10
            cmap = sns.color_palette(CMAP, n_states)
            colors, dpi = figure_style()
            f, ax = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)

            for kk, time_bin in enumerate(time_ax):
                ax.add_patch(Rectangle((time_bin-BIN_SIZE/2, -1), BIN_SIZE, len(clusters_in_region)+1,
                                       color=cmap[state_mat[trial, kk]],
                                       alpha=0.25, lw=0))

            tickedges = np.arange(0, len(clusters_in_region)+1)
            for k, n in enumerate(clusters_in_region):
                idx = np.bitwise_and(spikes.times[spikes.clusters == n] >= opto_times[trial] - PRE_TIME,
                                     spikes.times[spikes.clusters == n] <= opto_times[trial] + POST_TIME)
                neuron_spks = spikes.times[spikes.clusters == n][idx]
                ax.vlines(neuron_spks - opto_times[trial], tickedges[k + 1], tickedges[k], color='black',
                          lw=0.4, zorder=1)
            ax2 = ax.twinx()
            for k in range(n_states):
                ax2.plot(time_ax, prob_mat[trial, :, k], color=cmap[k])
            ax.set(xlabel='Time (s)', yticks=[0, len(clusters_in_region)],
                   yticklabels=[1, len(clusters_in_region)], xticks=[-1, 0, 1, 2, 3, 4],
                   ylim=[0, len(clusters_in_region)], title=f'{region}')
            ax.set_ylabel('Neurons', labelpad=-5)
            ax2.set(ylim=[0, 1])
            ax2.set_ylabel('P(state)', rotation=270, labelpad=10)
            sns.despine(trim=True, right=False)
            plt.tight_layout()

            plt.savefig(join(fig_path, f'{region}_{subject}_{date}_trial.jpg'), dpi=600)
            plt.close(f)

            # Plot session
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.25, 1.75), dpi=dpi)
            ax1.imshow(state_mat, aspect='auto', cmap=ListedColormap(cmap),
                       vmin=0, vmax=n_states-1,
                       extent=(-PRE_TIME, POST_TIME, 1, len(opto_times)), interpolation=None)
            ax1.plot([0, 0], [1, len(opto_times)], ls='--', color='k', lw=0.75)
            ax1.set(ylabel='Trials', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4],
                    title=f'{region}')

            for ii in range(n_states):
                ax2.plot(time_ax, np.mean(prob_mat[:, :, ii], axis=0), color=cmap[ii])
            ax2.set(xlabel='Time (s)', ylabel='P(state)', xticks=[-1, 0, 1, 2, 3, 4])

            ax3.imshow(trans_mat, aspect='auto', cmap='Greys', interpolation=None,
                       extent=(-PRE_TIME, POST_TIME, 1, len(opto_times)))
            ax32 = ax3.twinx()
            ax32.plot(time_ax, smooth_p_trans)
            ax32.set(ylabel='P(state change)')
            ax3.set(ylabel='Trials', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4])

            # sns.despine(trim=True)
            plt.tight_layout()
            plt.savefig(join(
                fig_path, f'{region}_{subject}_{date}_ses.jpg'),
                dpi=600)
            plt.close(f)

  