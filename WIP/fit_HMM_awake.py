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
from sklearn.model_selection import KFold
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_neuron_qc, calculate_peths,
                                 high_level_regions, figure_style, N_STATES)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
OVERWRITE = True
INCL_NEURONS = 'non-sig'  # all, sig or non-sig
PRE_TIME = 1  # final time window to use
POST_TIME = 4
HMM_PRE_TIME = 2  # time window to run HMM on
HMM_POST_TIME = 5
BIN_SIZE = 0.1  # s
MIN_NEURONS = 5
CROSS_VAL = False
K_FOLDS = 5
CV_SHUFFLE = True
CMAP = 'Set3'
PTRANS_SMOOTH = 0.2  # s

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Awake', f'{INCL_NEURONS}', f'{int(BIN_SIZE*1000)}ms')

# Query sessions
rec = query_ephys_sessions(one=one)

# Get significantly modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Initialize k-fold cross validation
kf = KFold(n_splits=K_FOLDS, shuffle=CV_SHUFFLE, random_state=42)

if OVERWRITE:
    state_trans_df = pd.DataFrame()
    p_state_df = pd.DataFrame()
else:
    state_trans_df = pd.read_csv(join(save_path, f'all_state_trans_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    p_state_df = pd.read_csv(join(save_path, f'p_state_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
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

        # Initialize HMM
        simple_hmm = ssm.HMM(N_STATES[region], clusters_in_region.shape[0], observations='poisson')

        # Get binned spikes centered at stimulation onset
        peth, binned_spikes = calculate_peths(spikes.times, spikes.clusters, clusters_in_region, opto_times,
                                              pre_time=HMM_PRE_TIME, post_time=HMM_POST_TIME,
                                              bin_size=BIN_SIZE, smoothing=0, return_fr=False)
        binned_spikes = binned_spikes.astype(int)
        full_time_ax = peth['tscale']
        use_timepoints = (full_time_ax > -PRE_TIME) & (full_time_ax < POST_TIME)
        time_ax = full_time_ax[use_timepoints]

        # Create list of (time_bins x neurons) per stimulation trial
        trial_data = []
        for i in range(binned_spikes.shape[0]):
            trial_data.append(np.transpose(binned_spikes[i, :, :]))

        # Loop over different number of states
        this_df = pd.DataFrame()
        state_trans = []
        trans_mat = np.empty((len(trial_data), full_time_ax.shape[0]))
        posterior = np.empty((len(trial_data), binned_spikes.shape[2], N_STATES[region]))
        if CROSS_VAL:
            # Cross validate
            for k, (train_index, test_index) in enumerate(kf.split(trial_data)):

                # Fit HMM on training data
                lls = simple_hmm.fit(list(np.array(trial_data)[train_index]), method='em',
                                     transitions='sticky')

                for t in test_index:

                    # Get posterior probability and most likely states for this trial
                    posterior[t, :, :] = simple_hmm.filter(trial_data[t])
                    zhat = simple_hmm.most_likely_states(trial_data[t])
                    zhat = zhat[use_timepoints]

                    # Add to dataframe
                    this_df = pd.concat((this_df, pd.DataFrame(data={
                        'state': zhat, 'region': region, 'time': time_ax, 'trial': t})))

        else:
            # Fit HMM on all data
            lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')

            for t in range(len(trial_data)):

                # Get posterior probability and most likely states for this trial
                posterior[t, :, :] = simple_hmm.filter(trial_data[t])
                zhat = simple_hmm.most_likely_states(trial_data[t])

                # Get state transitions times
                trans_mat[t, :] = np.concatenate((np.diff(zhat) > 0, [False])).astype(int)

                # Select timepoints to use
                zhat = zhat[use_timepoints]

                # Add to dataframe
                this_df = pd.concat((this_df, pd.DataFrame(data={
                    'state': zhat, 'region': region, 'time': time_ax, 'trial': t})))

        # Smooth P(state change) over entire period
        p_trans = np.mean(trans_mat, axis=0)
        smooth_p_trans = gaussian_filter(p_trans, PTRANS_SMOOTH / BIN_SIZE)

        # Select time period to use
        trans_mat = trans_mat[:, use_timepoints]
        smooth_p_trans = smooth_p_trans[use_timepoints]

        # Get P(state)
        p_state = pd.DataFrame()
        mean_state_inc = np.empty(N_STATES[region])
        for ii in range(N_STATES[region]):
            this_df[f'state_{ii}'] = (this_df['state'] == ii).astype(int)
            p_state[f'state_{ii}'] = this_df[['time', f'state_{ii}']].groupby('time').mean()
            p_state[f'state_{ii}_bl'] = (p_state[f'state_{ii}']
                                         - p_state.loc[p_state.index < 0, f'state_{ii}'].mean())
            mean_state_inc[ii] = p_state.loc[(p_state.index > 0) & (p_state.index < 1),
                                             f'state_{ii}_bl'].mean()

        # Add states with biggest increase and biggest decrease to dataframe
        this_p_state = p_state[[f'state_{np.argmax(mean_state_inc)}_bl', f'state_{np.argmin(mean_state_inc)}_bl']].rename(
                    columns={f'state_{np.argmax(mean_state_inc)}_bl': 'state_incr',
                             f'state_{np.argmin(mean_state_inc)}_bl': 'state_decr'})
        this_p_state['subject'] = subject
        this_p_state['pid'] = pid
        this_p_state['region'] = region
        p_state_df = pd.concat((p_state_df, this_p_state))

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
        ax.imshow(this_df.loc[this_df['trial'] == trial, 'state'].values[None, :],
                  aspect='auto', cmap=ListedColormap(cmap), vmin=0, vmax=N_STATES[region]-1, alpha=0.4,
                  extent=(-PRE_TIME, POST_TIME, -1, len(clusters_in_region)+1))
        tickedges = np.arange(0, len(clusters_in_region)+1)
        for i, n in enumerate(clusters_in_region):
            idx = np.bitwise_and(spikes.times[spikes.clusters == n] >= opto_times[trial] - PRE_TIME,
                                 spikes.times[spikes.clusters == n] <= opto_times[trial] + POST_TIME)
            neuron_spks = spikes.times[spikes.clusters == n][idx]
            ax.vlines(neuron_spks - opto_times[trial], tickedges[i + 1], tickedges[i], color='black',
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
        pivot_df = this_df.pivot_table(index='trial', columns='time', values='state').sort_values(
            'trial', ascending=False)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.25, 1.75), dpi=dpi)
        ax1.imshow(pivot_df, aspect='auto', cmap=ListedColormap(cmap), vmin=0, vmax=N_STATES[region]-1,
                  extent=(-PRE_TIME, POST_TIME, 1, len(opto_times)), interpolation=None)
        ax1.plot([0, 0], [1, len(opto_times)], ls='--', color='k', lw=0.75)
        ax1.set(ylabel='Trials', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4],
               title=f'{region}')

        for ii in range(N_STATES[region]):
            ax2.plot(p_state.index.values, p_state[f'state_{ii}_bl'].values, color=cmap[ii])
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
    state_trans_df.to_csv(join(save_path, f'all_state_trans_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    p_state_df.to_csv(join(save_path, f'p_state_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))





