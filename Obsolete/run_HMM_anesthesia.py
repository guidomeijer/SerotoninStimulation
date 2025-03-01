#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:52:05 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import ssm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from brainbox.io.one import SpikeSortingLoader
from stim_functions import (load_passive_opto_times, get_neuron_qc, paths, query_ephys_sessions,
                            figure_style, load_subjects, remap, high_level_regions,
                            get_artifact_neurons, calculate_peths)
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
N_STATES = 2
BIN_SIZE = 0.2
MIN_NEURONS = 2
PRE_TIME = 1  # final time window to use
POST_TIME = 4
HMM_PRE_TIME = 2  # time window to run HMM on
HMM_POST_TIME = 5
OVERWRITE = True
PLOT = False

# Get path
fig_path, save_path = paths()

# Query sessions
rec_both = query_ephys_sessions(anesthesia='both', one=one)
rec_both['anesthesia'] = 'both'
rec_anes = query_ephys_sessions(anesthesia='yes', one=one)
rec_anes['anesthesia'] = 'yes'
rec = pd.concat((rec_both, rec_anes)).reset_index(drop=True)
subjects = load_subjects()

# Get artifact neurons
artifact_neurons = get_artifact_neurons()

if OVERWRITE:
    up_down_state_df = pd.DataFrame()
else:
    up_down_state_df = pd.read_csv(join(save_path, 'updown_state_anesthesia.csv'))
    rec = rec[~rec['pid'].isin(up_down_state_df['pid'])]

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'\nStarting {subject}, {date}, {probe} ({i+1} of {len(rec)})')

    if not OVERWRITE:
        if pid in up_down_state_df['pid'].values:
            continue

    # Load opto times
    if rec.loc[i, 'anesthesia'] == 'both':
        opto_times, _ = load_passive_opto_times(eid, anesthesia=True, one=one)
    else:
        opto_times, _ = load_passive_opto_times(eid, one=one)

    # Generate random onset times
    random_times = np.sort(np.random.uniform(opto_times[0], opto_times[-1], size=opto_times.shape[0]))

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

        # Get binned spikes centered at stimulation onset
        peth, binned_spikes = calculate_peths(spikes.times, spikes.clusters, clusters_in_region, 
                                              np.concatenate((random_times, opto_times)),
                                              pre_time=HMM_PRE_TIME, post_time=HMM_POST_TIME, bin_size=BIN_SIZE,
                                              smoothing=0, return_fr=False)
        binned_spikes = binned_spikes.astype(int)
        full_time_ax = peth['tscale']
        use_timepoints = (full_time_ax > -PRE_TIME) & (full_time_ax < POST_TIME)
        time_ax = full_time_ax[use_timepoints]

        # Create list of (time_bins x neurons) per stimulation trial
        trial_data = []
        for i in range(binned_spikes.shape[0]):
            trial_data.append(np.transpose(binned_spikes[i, :, :]))

        # Initialize HMM
        simple_hmm = ssm.HMM(N_STATES, clusters_in_region.shape[0], observations='poisson')

        this_df = pd.DataFrame()
        trans_mat = np.empty((len(trial_data), full_time_ax.shape[0]))
        down_trans_mat = np.empty((len(trial_data), full_time_ax.shape[0]))
        up_trans_mat = np.empty((len(trial_data), full_time_ax.shape[0]))

        # Fit HMM on all data
        lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')

        prob_mat = np.empty((len(trial_data), full_time_ax.shape[0]))
        state_mat = np.empty((len(trial_data), full_time_ax.shape[0]))
        for t in range(len(trial_data)):

            # Get posterior probability and most likely states for this trial
            posterior = simple_hmm.filter(trial_data[t])
            zhat = simple_hmm.most_likely_states(trial_data[t])

            # Make sure 0 is down state and 1 is up state
            if np.mean(binned_spikes[t, :, zhat == 0]) > np.mean(binned_spikes[t, :, zhat == 1]):
                # State 0 is up state
                zhat = np.where((zhat == 0) | (zhat == 1), zhat ^ 1, zhat)
                prob_mat[t, :] = posterior[:, 1]
            else:
                prob_mat[t, :] = posterior[:, 0]
            state_mat[t, :] = zhat
            
        # Crop final timewindow        
        prob_mat = prob_mat[:, np.concatenate(([False], use_timepoints[:-1]))]  
        state_mat = state_mat[:, np.concatenate(([False], use_timepoints[:-1]))]  

        # Get down state probability over trials
        p_down = np.mean(prob_mat[int(prob_mat.shape[0]/2):], axis=0)
        p_down_null = np.mean(prob_mat[:int(prob_mat.shape[0]/2)], axis=0)
        p_down_bl = p_down - np.mean(p_down[time_ax < 0])
        p_down_null_bl = p_down_null - np.mean(p_down_null[time_ax < 0])

        up_down_state_df = pd.concat((up_down_state_df, pd.DataFrame(data={
            'p_down': p_down, 'p_down_bl': p_down_bl, 'time': time_ax, 'opto': 1,
            'subject': subject, 'pid': pid, 'region': region})))
        up_down_state_df = pd.concat((up_down_state_df, pd.DataFrame(data={
            'p_down': p_down_null, 'p_down_bl': p_down_null_bl, 'time': time_ax, 'opto': 0,
            'subject': subject, 'pid': pid, 'region': region})))

        # Save the trial-level P(state) data and zhat matrix
        np.save(join(save_path, 'HMM', 'Anesthesia', 'prob_mat',
                     f'{subject}_{date}_{probe}_{region}.npy'), prob_mat)
        np.save(join(save_path, 'HMM', 'Anesthesia', 'state_mat',
                     f'{subject}_{date}_{probe}_{region}.npy'), state_mat)

        if PLOT:
            # Plot example trial
            trial = 1
            colors, dpi = figure_style()
            cmap = ListedColormap([colors['suppressed'], colors['enhanced']])
            f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
            for kk, time_bin in enumerate(time_ax):
                ax.add_patch(Rectangle((time_bin-BIN_SIZE/2, -1), BIN_SIZE, len(clusters_in_region)+1,
                                       color=cmap.colors[this_df.loc[this_df['trial']
                                                                     == trial, 'state'].values[kk]],
                                       alpha=0.25, lw=0))
            tickedges = np.arange(0, len(clusters_in_region)+1)
            for i, n in enumerate(clusters_in_region):
                idx = np.bitwise_and(spikes.times[spikes.clusters == n] >= opto_times[trial] - PRE_TIME,
                                     spikes.times[spikes.clusters == n] <= opto_times[trial] + POST_TIME)
                neuron_spks = spikes.times[spikes.clusters == n][idx]
                ax.vlines(neuron_spks - opto_times[trial], tickedges[i + 1], tickedges[i], color='black',
                          lw=0.5)

            ax.set(xlabel='Time (s)', yticks=[0, len(clusters_in_region)],
                   yticklabels=[1, len(clusters_in_region)], xticks=[-1, 0, 1, 2, 3, 4],
                   ylim=[0, len(clusters_in_region)], title=f'{region}')
            ax.set_ylabel('Neurons', labelpad=-5)

            ax2 = ax.twinx()
            ax2.plot(time_ax, this_df.loc[this_df['trial'] == trial, 'p_pos_down'], lw=0.5,
                     color=colors['suppressed'])
            ax2.set(ylim=[-0.01, 1.01])
            ax2.set_ylabel('P(down state)', rotation=270, labelpad=10)
            ax2.yaxis.label.set_color(colors['suppressed'])
            ax2.tick_params(axis='y', colors=colors['suppressed'])
            ax2.spines['right'].set_color(colors['suppressed'])

            sns.despine(trim=True, right=False)
            plt.tight_layout()

            plt.savefig(join(fig_path, 'Extra plots', 'State', 'Anesthesia',
                             f'{region}_{subject}_{date}_trial.jpg'),
                        dpi=600)
            plt.close(f)

            # Plot session
            pivot_df = this_df.pivot_table(index='trial', columns='time', values='state').sort_values(
                'trial', ascending=False)
            f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
            ax.imshow(pivot_df, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                      extent=(-PRE_TIME, POST_TIME, 1, len(opto_times)))
            ax.plot([0, 0], [1, len(opto_times)], ls='--', color='k', lw=0.75)
            ax.set(ylabel='Trials', xlabel='Time (s)', yticks=[1, 25, 50], xticks=[-1, 0, 1, 2, 3, 4],
                   title=f'{region}')
            sns.despine(trim=True)
            plt.tight_layout()
            plt.savefig(join(fig_path, 'Extra plots', 'State', 'Anesthesia',
                             f'{region}_{subject}_{date}_ses.jpg'),
                        dpi=600)
            plt.close(f)

    # Save result
    up_down_state_df.to_csv(join(save_path, 'updown_states_anesthesia.csv'))
