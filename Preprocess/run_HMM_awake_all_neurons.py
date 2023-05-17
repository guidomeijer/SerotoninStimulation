#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:20:14 2023
By: Guido Meijer
"""

import numpy as np
np.random.seed(42)
import ssm
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from brainbox.io.one import SpikeSortingLoader
from scipy.ndimage import gaussian_filter
from brainbox.singlecell import calculate_peths
from stim_functions import (paths, query_ephys_sessions, load_passive_opto_times,
                            figure_style, N_STATES)
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
MIN_NEURONS = 5
CMAP = 'colorblind'
INCL_NEURONS = 'all'  # all, sig or non-sig
PTRANS_SMOOTH = BIN_SIZE
PSTATE_SMOOTH = BIN_SIZE
OVERWRITE = True
PLOT = False

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Awake', 'All neurons')

# Query sessions
rec = query_ephys_sessions(one=one)

# Get significantly modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

if OVERWRITE:
    state_trans_df, p_state_df = pd.DataFrame(), pd.DataFrame()
    state_trans_null_df, p_state_null_df = pd.DataFrame(), pd.DataFrame()
else:
    state_trans_df = pd.read_csv(join(save_path, f'all_state_trans_all-neurons_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    p_state_df = pd.read_csv(join(save_path, f'p_state_all-neurons_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    p_state_null_df = pd.read_csv(join(save_path, f'p_state_null_all-neurons_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    state_trans_null_df = pd.read_csv(join(save_path, f'all_state_trans_null_all-neurons_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    rec = rec[~rec['pid'].isin(state_trans_df['pid'])]


for i, eid in enumerate(np.unique(rec['eid'])):

    # Get session details
    subject = one.get_details(eid)['subject']
    date = one.get_details(eid)['date']
    print(f'\nStarting {subject}, {date} ({i+1} of {len(np.unique(rec["eid"]))})')

    # Load in laser pulse times
    opto_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_times) == 0:
        print('Could not load light pulses')
        continue
    
    # Get the insertions from the session
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
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, use_neurons)]
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
        
    # Create list of (time_bins x neurons) per stimulation trial
    trial_data = []
    for j in range(binned_spikes.shape[0]):
        trial_data.append(np.transpose(binned_spikes[j, :, :]))
    
    # Fit HMM
    simple_hmm = ssm.HMM(N_STATES, binned_spikes.shape[1], observations='poisson')
    lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')
    
    # Loop over trials
    trans_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
    state_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
    prob_mat = np.empty((len(trial_data), full_time_ax.shape[0], N_STATES))
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
    p_state_mat = np.empty((N_STATES, time_ax.shape[0]))
    for ii in range(N_STATES):

        # Get P state, first smooth, then crop timewindow
        this_p_state = np.mean(state_mat == ii, axis=0)
        smooth_p_state = gaussian_filter(this_p_state, PSTATE_SMOOTH / BIN_SIZE)
        smooth_p_state = smooth_p_state[use_timepoints]
        p_state_bl = smooth_p_state - np.mean(smooth_p_state[time_ax < 0])

        # Add to dataframe and matrix
        p_state_mat[ii, :] = smooth_p_state
        p_state_df = pd.concat((p_state_df, pd.DataFrame(data={
            'p_state': smooth_p_state, 'p_state_bl': p_state_bl, 'state': ii, 'time': time_ax,
            'subject': subject, 'eid': eid})))

    # Add state change PSTH to dataframe
    state_trans_df = pd.concat((state_trans_df, pd.DataFrame(data={
        'time': time_ax, 'p_trans': smooth_p_trans,
        'p_trans_bl': smooth_p_trans - np.mean(smooth_p_trans[time_ax < 0]),
        'subject': subject, 'eid': eid})))
    
    # Crop timewindow for plotting
    state_mat = state_mat[:, use_timepoints]
    
    if PLOT:
        # Plot example trial
        trial = 0
        cmap = sns.color_palette(CMAP, N_STATES)
        colors, dpi = figure_style()
        f, ax = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
        
        for kk, time_bin in enumerate(time_ax):
            ax.add_patch(Rectangle((time_bin-BIN_SIZE/2, -1), BIN_SIZE, binned_spikes.shape[1]+1,
                                   color=cmap[state_mat[trial, kk]], alpha=0.25, lw=0)) 
        tickedges = np.arange(0, binned_spikes.shape[1]+1)
        for pp, probe in enumerate(np.sort(probes)):
            for k, n in enumerate(np.unique(spikes[probe].clusters)):
                idx = np.bitwise_and(spikes[probe].times[spikes[probe].clusters == n] >= opto_times[trial] - PRE_TIME,
                                     spikes[probe].times[spikes[probe].clusters == n] <= opto_times[trial] + POST_TIME)
                neuron_spks = spikes[probe].times[spikes[probe].clusters == n][idx]
                if pp == 0:
                    ax.vlines(neuron_spks - opto_times[trial], tickedges[k + 1], tickedges[k],
                              color='black', lw=0.4, zorder=1)
                elif pp == 1:
                    ax.vlines(neuron_spks - opto_times[trial],
                              tickedges[k + len(np.unique(spikes['probe00'].clusters)) + 1],
                              tickedges[k + len(np.unique(spikes['probe00'].clusters))],
                              color='black', lw=0.4, zorder=1)
        ax2 = ax.twinx()
        for kk in range(N_STATES):
            ax2.plot(time_ax, prob_mat[trial, :, kk], color=cmap[kk])
        ax.set(xlabel='Time (s)', yticks=[0, binned_spikes.shape[1]],
               yticklabels=[1, binned_spikes.shape[1]], xticks=[-1, 0, 1, 2, 3, 4],
               ylim=[0, binned_spikes.shape[1]])
        ax.set_ylabel('Neurons', labelpad=-5)
        ax2.set(ylim=[0, 1])
        ax2.set_ylabel('P(state)', rotation=270, labelpad=10)
        sns.despine(trim=True, right=False)
        plt.tight_layout()
        
        plt.savefig(join(fig_path, f'{subject}_{date}_trial.jpg'), dpi=600)
        plt.close(f)
        
        # Plot session
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.25, 1.75), dpi=dpi)
        ax1.imshow(state_mat, aspect='auto', cmap=ListedColormap(cmap),
                   vmin=0, vmax=N_STATES-1,
                   extent=(-PRE_TIME, POST_TIME, 1, len(opto_times)), interpolation=None)
        ax1.plot([0, 0], [1, len(opto_times)], ls='--', color='k', lw=0.75)
        ax1.set(ylabel='Trials', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4])

        for ii in range(N_STATES):
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
        plt.savefig(join(fig_path, f'{subject}_{date}_ses.jpg'), dpi=600)
        plt.close(f)

    # Run the HMM on random onset times in the spontaneous activity
    random_times = np.sort(np.random.uniform(opto_times[0]-360, opto_times[0]-10, size=100))
        
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
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, use_neurons)]
        use_neurons = use_neurons[np.isin(use_neurons, np.unique(spikes[probe].clusters))]
    
        # Get binned spikes centered at stimulation onset
        peth, these_spikes = calculate_peths(spikes[probe].times, spikes[probe].clusters, use_neurons,
                                             random_times, pre_time=HMM_PRE_TIME, post_time=HMM_POST_TIME,
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
        
    # Create list of (time_bins x neurons) per stimulation trial
    trial_data = []
    for j in range(binned_spikes.shape[0]):
        trial_data.append(np.transpose(binned_spikes[j, :, :]))
    
    # Fit HMM
    simple_hmm = ssm.HMM(N_STATES, binned_spikes.shape[1], observations='poisson')
    lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')
        
    # Loop over trials
    trans_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
    state_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
    prob_mat = np.empty((len(trial_data), full_time_ax.shape[0], N_STATES))
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
    p_state_mat = np.empty((N_STATES, time_ax.shape[0]))
    for ii in range(N_STATES):
    
        # Get P state, first smooth, then crop timewindow
        this_p_state = np.mean(state_mat == ii, axis=0)
        smooth_p_state = gaussian_filter(this_p_state, PSTATE_SMOOTH / BIN_SIZE)
        smooth_p_state = smooth_p_state[use_timepoints]
        p_state_bl = smooth_p_state - np.mean(smooth_p_state[time_ax < 0])
    
        # Add to dataframe and matrix
        p_state_mat[ii, :] = smooth_p_state
        p_state_null_df = pd.concat((p_state_null_df, pd.DataFrame(data={
            'p_state': smooth_p_state, 'p_state_bl': p_state_bl, 'state': ii, 'time': time_ax,
            'subject': subject, 'eid': eid})))
    
    # Add state change PSTH to dataframe
    state_trans_null_df = pd.concat((state_trans_null_df, pd.DataFrame(data={
        'time': time_ax, 'p_trans': smooth_p_trans,
        'p_trans_bl': smooth_p_trans - np.mean(smooth_p_trans[time_ax < 0]),
        'subject': subject, 'eid': eid})))
         
    # Save output
    state_trans_df.to_csv(join(save_path, f'all_state_trans_all-neurons_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    p_state_df.to_csv(join(save_path, f'p_state_all-neurons_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    state_trans_null_df.to_csv(join(save_path, f'all_state_trans_null_all-neurons_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))
    p_state_null_df.to_csv(join(save_path, f'p_state_null_all-neurons_{int(BIN_SIZE*1000)}msbins_{INCL_NEURONS}.csv'))


