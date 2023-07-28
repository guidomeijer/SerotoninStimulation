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
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from brainbox.io.one import SpikeSortingLoader
from scipy.ndimage import gaussian_filter
from brainbox.singlecell import calculate_peths
from stim_functions import (paths, remap, query_ephys_sessions, load_trials, get_artifact_neurons,
                            high_level_regions, figure_style, N_STATES_TASK, N_STATES_TASK_REGIONS)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
N_STATE_SELECT = 'region'
INCL_NEURONS = 'all'
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
PLOT = True

# Create text to add to save files
add_str = f'{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}_{N_STATE_SELECT}-nstates'

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Task', f'{N_STATE_SELECT}')

# Query sessions
rec = query_ephys_sessions(n_trials=400, one=one)

# Get artifact neurons
artifact_neurons = get_artifact_neurons()

# Get significantly modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

if OVERWRITE:
    p_state_df = pd.DataFrame()
    state_trans_df = pd.DataFrame()
else:
    p_state_df = pd.read_csv(join(save_path, f'p_state_task_{add_str}.csv'))
    state_trans_df = pd.read_csv(join(save_path, f'state_trans_task_{add_str}.csv'))
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

        if N_STATE_SELECT == 'global':
            n_states = N_STATES_TASK
        elif N_STATE_SELECT == 'region':
            n_states = N_STATES_TASK_REGIONS[region]

        # Select spikes and clusters in this brain region
        clusters_in_region = use_neurons[clusters_regions == region]

        if len(clusters_in_region) < MIN_NEURONS:
            continue

        # Initialize HMM
        simple_hmm = ssm.HMM(n_states, clusters_in_region.shape[0], observations='poisson')

        # Get binned spikes centered at stimulation onset
        peth, binned_spikes = calculate_peths(spikes.times, spikes.clusters, clusters_in_region,
                                              trials['stimOn_times'],
                                              pre_time=HMM_PRE_TIME, post_time=HMM_POST_TIME,
                                              bin_size=BIN_SIZE, smoothing=0, return_fr=False)
        binned_spikes = binned_spikes.astype(int)
        full_time_ax = peth['tscale']
        use_timepoints = (full_time_ax > -PRE_TIME) & (full_time_ax < POST_TIME)
        time_ax = full_time_ax[use_timepoints]

        # Create list of (time_bins x neurons) per stimulation trial
        trial_data = []
        for j in range(binned_spikes.shape[0]):
            trial_data.append(np.transpose(binned_spikes[j, :, :]))

        # Initialize HMM
        lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')

        # Loop over trials
        trans_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
        state_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
        prob_mat = np.empty((len(trial_data), full_time_ax.shape[0], n_states))
        
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

        # Select time period to use
        trans_mat = trans_mat[:, use_timepoints]
        smooth_p_trans = smooth_p_trans[use_timepoints]
        prob_mat = prob_mat[:, np.concatenate(([False], use_timepoints[:-1])), :]

        # Get P(state)
        p_state_mat = np.empty((n_states, time_ax.shape[0]))
        for ii in range(n_states):

            # Get P state, first smooth, then crop timewindow
            this_p_state = np.mean(prob_mat[:, :, ii], axis=0)
            p_state_bl = this_p_state - np.mean(this_p_state[time_ax < 0])

            # Add to dataframe and matrix
            p_state_mat[ii, :] = this_p_state
            p_state_df = pd.concat((p_state_df, pd.DataFrame(data={
                'p_state': this_p_state, 'p_state_bl': p_state_bl, 'state': ii, 'time': time_ax,
                'subject': subject, 'pid': pid, 'region': region})))

        # Add state change PSTH to dataframe
        state_trans_df = pd.concat((state_trans_df, pd.DataFrame(data={
            'time': time_ax, 'p_trans': smooth_p_trans,
            'cumsum_trans': np.cumsum(np.sum(trans_mat, axis=0)),
            'p_trans_bl': smooth_p_trans - np.mean(smooth_p_trans[time_ax < 0]),
            'region': region, 'subject': subject, 'pid': pid})))
        
        # Crop timewindow for plotting
        state_mat = state_mat[:, use_timepoints]
        
        # Save the trial-level P(state) data and zhat matrix
        np.save(join(save_path, 'HMM', 'Task', f'{N_STATE_SELECT}', 'prob_mat',
                     f'{subject}_{date}_{probe}_{region}.npy'), prob_mat)
        np.save(join(save_path, 'HMM', 'Task', f'{N_STATE_SELECT}', 'state_mat',
                     f'{subject}_{date}_{probe}_{region}.npy'), state_mat)
        
        if PLOT:
            # Plot example trial
            trial = 0
            cmap = sns.color_palette(CMAP, n_states)
            colors, dpi = figure_style()
            f, ax = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
            
            for kk, time_bin in enumerate(time_ax):
                ax.add_patch(Rectangle((time_bin-BIN_SIZE/2, -1), BIN_SIZE, len(clusters_in_region)+1,
                                       color=cmap[state_mat[trial, kk]],
                                       alpha=0.25, lw=0)) 
                
            tickedges = np.arange(0, len(clusters_in_region)+1)
            for k, n in enumerate(clusters_in_region):
                idx = np.bitwise_and(spikes.times[spikes.clusters == n] >= trials['stimOn_times'][trial] - PRE_TIME,
                                     spikes.times[spikes.clusters == n] <= trials['stimOn_times'][trial] + POST_TIME)
                neuron_spks = spikes.times[spikes.clusters == n][idx]
                ax.vlines(neuron_spks - trials['stimOn_times'][trial], tickedges[k + 1], tickedges[k], color='black',
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
                      extent=(-PRE_TIME, POST_TIME, 1, len(trials['stimOn_times'])), interpolation=None)
            ax1.plot([0, 0], [1, len(trials['stimOn_times'])], ls='--', color='k', lw=0.75)
            ax1.set(ylabel='Trials', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4],
                   title=f'{region}')
    
            for ii in range(n_states):
                ax2.plot(time_ax, p_state_mat[ii, :], color=cmap[ii])
            ax2.set(xlabel='Time (s)', ylabel='P(state)', xticks=[-1, 0, 1, 2, 3, 4])
    
            ax3.imshow(trans_mat, aspect='auto', cmap='Greys', interpolation=None,
                       extent=(-PRE_TIME, POST_TIME, 1, len(trials['stimOn_times'])))
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
        state_trans_df.to_csv(join(save_path, f'state_trans_task_{add_str}.csv'))
        p_state_df.to_csv(join(save_path, f'p_state_task_{add_str}.csv'))





