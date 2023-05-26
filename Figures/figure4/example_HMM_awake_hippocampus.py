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
                            figure_style, N_STATES, remap, high_level_regions)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
PID = '5d2acdf4-b7f6-4545-b72c-ee88a64b81c5'
#PID = '7a82c06b-0e33-454b-a98f-786a4024c1d0'
REGION = 'Hippocampus'
BIN_SIZE = 0.1  # s
PRE_TIME = 1  # final time window to use
POST_TIME = 4
HMM_PRE_TIME = 2  # time window to run HMM on
HMM_POST_TIME = 5
MIN_NEURONS = 5
INCL_NEURONS = 'all'  # all, sig or non-sig
PTRANS_SMOOTH = BIN_SIZE
PSTATE_SMOOTH = BIN_SIZE
OVERWRITE = True
PLOT = False

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Query sessions
rec = query_ephys_sessions(one=one)

# Get significantly modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Get session details
eid = one.pid2eid(PID)[0]
subject = one.get_details(eid)['subject']
date = one.get_details(eid)['date']

# Load in laser pulse times
opto_times, _ = load_passive_opto_times(eid, one=one)

# Load in spikes
sl = SpikeSortingLoader(pid=PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Select neurons to use
if INCL_NEURONS == 'all':
    use_neurons = light_neurons.loc[light_neurons['pid'] == PID, 'neuron_id'].values
elif INCL_NEURONS == 'sig':
    use_neurons = light_neurons.loc[(light_neurons['pid'] == PID) & light_neurons['modulated'],
                                    'neuron_id'].values
elif INCL_NEURONS == 'non-sig':
    use_neurons = light_neurons.loc[(light_neurons['pid'] == PID) & ~light_neurons['modulated'],
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
clusters_in_region = use_neurons[clusters_regions == REGION]

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

# Initialize HMM
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

# Crop timewindow for plotting
state_mat = state_mat[:, use_timepoints]
prob_mat = prob_mat[:, np.concatenate(([False], use_timepoints[:-1])), :]

# %% Plot example trial
trial = 11

colors, dpi = figure_style()
cmap = sns.color_palette(colors['states_light'], N_STATES)
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(1.25, 1.75), dpi=dpi, sharex=True)

ax1.add_patch(Rectangle((0, 0), 1, len(clusters_in_region), color='royalblue', alpha=0.25, lw=0))
tickedges = np.arange(0, binned_spikes.shape[1]+1)
for k, n in enumerate(clusters_in_region):
    idx = np.bitwise_and(spikes.times[spikes.clusters == n] >= opto_times[trial] - PRE_TIME,
                         spikes.times[spikes.clusters == n] <= opto_times[trial] + POST_TIME)
    neuron_spks = spikes.times[spikes.clusters == n][idx]
    ax1.vlines(neuron_spks - opto_times[trial], tickedges[k + 1], tickedges[k],
              color='black', lw=0.4, zorder=1)   

ax1.set(yticks=[0, binned_spikes.shape[1]], yticklabels=[1, binned_spikes.shape[1]], 
        ylim=[0, binned_spikes.shape[1]], xticks=[])
ax1.set_ylabel('Neurons', labelpad=-12)

for kk in range(N_STATES):
    ax2.plot(time_ax, prob_mat[trial, :, kk], color=cmap[kk], lw=0.75)

ax2.set(ylim=[0, 1], yticks=[0, 1])
ax2.set_ylabel('P(state)', labelpad=-5)
#ax2.set_ylabel('P(state)', rotation=270, labelpad=10)
ax2.plot([0, 2], [-0.025, -0.025], lw=0.75, color='k', clip_on=False)
ax2.text(1, -0.125, '2s', ha='center', va='center')

ax2.add_patch(Rectangle((0, 0), 1, 2, color='royalblue', alpha=0.25, lw=0))

sns.despine(trim=True, bottom=True)
plt.subplots_adjust(hspace=0.15, left=0.2, right=0.99, top=0.95, bottom=0.1)
plt.savefig(join(fig_path, f'hmm_example_trial_{REGION}.pdf'))

f, ax1 = plt.subplots(1, 1, figsize=(1.25, 1.75), dpi=dpi)

ax1.add_patch(Rectangle((0, 1), 1, len(opto_times), color='royalblue', alpha=0.25, lw=0))
ax1.imshow(np.flipud(state_mat), aspect='auto', cmap=ListedColormap(cmap),
           vmin=0, vmax=N_STATES-1,
           extent=(-PRE_TIME, POST_TIME, 1, len(opto_times)+1), interpolation=None)
#ax1.plot([-1, 4], [trial+1, trial+1], color='k', lw=0.5)
#ax1.plot([-1, 4], [trial+2.1, trial+2.1], color='k', lw=0.5)
ax1.set(xticks=[], yticks=np.array([1, 50]) + 0.5, yticklabels=np.array([1, 50]))
ax1.set_ylabel('Trials', labelpad=-10)
ax1.plot([0, 2], [0, 0], lw=0.75, color='k', clip_on=False)
ax1.text(1, -2, '2s', ha='center', va='center')

sns.despine(trim=True, bottom=True)
plt.subplots_adjust(left=0.2, top=0.97, bottom=0.05)
plt.savefig(join(fig_path, f'hmm_example_session_{REGION}.pdf'))



 