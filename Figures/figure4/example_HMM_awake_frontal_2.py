#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:20:14 2023
By: Guido Meijer
"""

import numpy as np
np.random.seed(0)
import ssm
from os.path import join, realpath, dirname, split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from brainbox.io.one import SpikeSortingLoader
from scipy.ndimage import gaussian_filter
from brainbox.singlecell import calculate_peths
from stim_functions import (paths, query_ephys_sessions, load_passive_opto_times,
                            figure_style, N_STATES, N_STATES_REGIONS, remap, high_level_regions)
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
PID_FRONT = 'c6a3def0-9bce-4ef9-a57d-0dc2d4ae3d65'
BIN_SIZE = 0.1  # s
PRE_TIME = 1  # final time window to use
POST_TIME = 3
HMM_PRE_TIME = 2  # time window to run HMM on
HMM_POST_TIME = 4
MIN_NEURONS = 5
INCL_NEURONS = 'all'  # all, sig or non-sig
PTRANS_SMOOTH = BIN_SIZE
PSTATE_SMOOTH = BIN_SIZE
OVERWRITE = True
PLOT = False
TRIAL = 36

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Query sessions
rec = query_ephys_sessions(one=one)

# Get significantly modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Get session details
eid = one.pid2eid(PID_FRONT)[0]
subject = one.get_details(eid)['subject']
date = one.get_details(eid)['date']

# Load in laser pulse times
opto_times, _ = load_passive_opto_times(eid, one=one)

# Load in spikes
sl = SpikeSortingLoader(pid=PID_FRONT, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Select neurons to use
if INCL_NEURONS == 'all':
    use_neurons = light_neurons.loc[light_neurons['pid'] == PID_FRONT, 'neuron_id'].values
elif INCL_NEURONS == 'sig':
    use_neurons = light_neurons.loc[(light_neurons['pid'] == PID_FRONT) & light_neurons['modulated'],
                                    'neuron_id'].values
elif INCL_NEURONS == 'non-sig':
    use_neurons = light_neurons.loc[(light_neurons['pid'] == PID_FRONT) & ~light_neurons['modulated'],
                                    'neuron_id'].values

# Select QC pass neurons
spikes.times = spikes.times[np.isin(spikes.clusters, use_neurons)]
spikes.clusters = spikes.clusters[np.isin(spikes.clusters, use_neurons)]
use_neurons = use_neurons[np.isin(use_neurons, np.unique(spikes.clusters))]

# Get regions from Beryl atlas
clusters['region'] = remap(clusters['acronym'], combine=True)
clusters['high_level_region'] = high_level_regions(clusters['acronym'])
clusters_regions = clusters['high_level_region'][use_neurons]

# Select spikes and clusters in this brain region
clusters_in_region = use_neurons[clusters_regions == 'Frontal cortex']

# Initialize HMM
simple_hmm = ssm.HMM(N_STATES, clusters_in_region.shape[0], observations='poisson')

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
for j in range(binned_spikes.shape[0]):
    trial_data.append(np.transpose(binned_spikes[j, :, :]))
            
# Fit HMM
simple_hmm = ssm.HMM(N_STATES, binned_spikes.shape[1], observations='poisson')
lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')

# Loop over trials
trans_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
state_mat = np.empty((len(trial_data), full_time_ax.shape[0])).astype(int)
prob_mat_frontal = np.empty((len(trial_data), full_time_ax.shape[0], N_STATES))
for t in range(len(trial_data)):

    # Get most likely states for this trial
    zhat = simple_hmm.most_likely_states(trial_data[t])
    prob_mat_frontal[t, :, :] = simple_hmm.filter(trial_data[t])
    
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
prob_mat_frontal = prob_mat_frontal[:, np.concatenate(([False], use_timepoints[:-1])), :]

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

# Crop timewindow for plotting
state_mat = state_mat[:, use_timepoints]


# %% Plot example trial
colors, dpi = figure_style()
cmap = sns.color_palette(colors['states_light'], N_STATES)
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(1.25, 2), dpi=dpi, sharex=True)

ax1.add_patch(Rectangle((0, 0), 1, len(clusters_in_region), color='royalblue', alpha=0.25, lw=0))
tickedges = np.arange(0, binned_spikes.shape[1]+1)
for k, n in enumerate(clusters_in_region):
    idx = np.bitwise_and(spikes.times[spikes.clusters == n] >= opto_times[TRIAL] - PRE_TIME,
                         spikes.times[spikes.clusters == n] <= opto_times[TRIAL] + POST_TIME)
    neuron_spks = spikes.times[spikes.clusters == n][idx]
    ax1.vlines(neuron_spks - opto_times[TRIAL], tickedges[k + 1], tickedges[k],
              color='black', lw=0.4, zorder=1)   

ax1.set(yticks=[0, binned_spikes.shape[1]], yticklabels=[1, binned_spikes.shape[1]], 
        ylim=[0, binned_spikes.shape[1]], xticks=[])
ax1.set_ylabel('Neurons', labelpad=-12)

for kk in range(N_STATES):
    ax2.plot(time_ax, prob_mat_frontal[TRIAL, :, kk], color=cmap[kk], lw=0.75)

ax2.set(ylim=[0, 1], yticks=[0, 1])
ax2.set_ylabel('P(state)', labelpad=-5)
#ax2.set_ylabel('P(state)', rotation=270, labelpad=10)
ax2.plot([0, 2], [-0.025, -0.025], lw=0.75, color='k', clip_on=False)
ax2.text(1, -0.125, '2s', ha='center', va='center')

ax2.add_patch(Rectangle((0, 0), 1, 2, color='royalblue', alpha=0.25, lw=0))

sns.despine(trim=True, bottom=True)
plt.subplots_adjust(hspace=0.15, left=0.2, right=0.99, top=0.95, bottom=0.1)
plt.savefig(join(fig_path, f'hmm_example_trial_frontal.pdf'))


# %%
f, ax1 = plt.subplots(1, 1, figsize=(1.25, 2), dpi=dpi)

ax1.add_patch(Rectangle((0, 1), 1, len(opto_times), color='royalblue', alpha=0.25, lw=0))
ax1.imshow(np.flipud(state_mat), aspect='auto', cmap=ListedColormap(cmap),
           vmin=0, vmax=N_STATES-1,
           extent=(-PRE_TIME, POST_TIME, 1, len(opto_times)+1), interpolation=None)
ax1.plot([-PRE_TIME, POST_TIME], [TRIAL+1, TRIAL+1], color='k', lw=0.5)
ax1.plot([-PRE_TIME, POST_TIME], [TRIAL+2.1, TRIAL+2.1], color='k', lw=0.5)
ax1.set(xticks=[], yticks=np.array([1, 50]) + 0.5, yticklabels=np.array([1, 50]))
ax1.set_ylabel('Trials', labelpad=-10)
ax1.plot([0, 2], [0.5, 0.5], lw=0.75, color='k', clip_on=False)
ax1.text(1, -1.5, '2s', ha='center', va='center')

sns.despine(trim=True, bottom=True)
plt.subplots_adjust(left=0.2, top=0.97, bottom=0.05)
plt.savefig(join(fig_path, 'hmm_example_session_frontal.pdf'))

# %%

f, ax1 = plt.subplots(figsize=(1.25, 2), dpi=dpi)
ax1.add_patch(Rectangle((0, -1.7), 1, 2.1, color='royalblue', alpha=0.25, lw=0))
for i, this_state in enumerate([0, 4, 2, 1, 5, 3, 6]):
    mean_state = (np.mean(prob_mat_frontal[:,:,this_state], axis=0)
                  - np.mean(prob_mat_frontal[:,time_ax < 0,this_state])) - (i/4)
    
    sem_state = np.std(prob_mat_frontal[:,:,this_state], axis=0) / np.sqrt(prob_mat_frontal.shape[0])
    ax1.plot(time_ax, mean_state, color=cmap[this_state])
    ax1.fill_between(time_ax, mean_state + sem_state, mean_state - sem_state, alpha=0.25,
                     color=cmap[this_state], lw=0)
ax1.plot([-1.1, -1.1], [-1.7, -1.45], color='k')
ax1.plot([0, 2], [-1.75, -1.75], color='k')
ax1.text(-1.4, -1.55, '25%', rotation=90, ha='center', va='center')
ax1.text(1, -1.83, '2s', ha='center', va='center')
ax1.set(xticks=[], yticks=[])
ax1.set_ylabel('P(state)', labelpad=0)
sns.despine(trim=True, left=True, bottom=True)
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.98)
plt.savefig(join(fig_path, 'hmm_example_p_states_frontal.pdf'))

