# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:50:10 2023

@author: Guido Meijer
"""

import numpy as np
import ssm
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, realpath, dirname, split
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from brainbox.io.one import SpikeSortingLoader
from stim_functions import (load_passive_opto_times, get_neuron_qc, get_artifact_neurons, remap,
                            high_level_regions, calculate_peths, figure_style, paths)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
PID = '542aa93b-4876-439c-93f7-dd77263e55e8'
PRE_TIME = 1  # final time window to use
POST_TIME = 4
HMM_PRE_TIME = 2  # time window to run HMM on
HMM_POST_TIME = 5
N_STATES = 2
BIN_SIZE = 0.2

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load artifact neurons
artifact_neurons = get_artifact_neurons()

# Load opto times
opto_times, _ = load_passive_opto_times(one.pid2eid(PID)[0], one=one)

# Load in spikes
sl = SpikeSortingLoader(pid=PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Filter neurons that pass QC
qc_metrics = get_neuron_qc(PID, one=one, ba=ba)
clusters_pass = np.where(qc_metrics['label'] == 1)[0]

# Exclude artifact neurons
clusters_pass = np.array([i for i in clusters_pass if i not in artifact_neurons.loc[
    artifact_neurons['pid'] == PID, 'neuron_id'].values])

# Select QC pass neurons
spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
clusters_pass = clusters_pass[np.isin(clusters_pass, np.unique(spikes.clusters))]

# Get regions from Beryl atlas
clusters['region'] = remap(clusters['acronym'], combine=True)
clusters['high_level_region'] = high_level_regions(clusters['acronym'])
clusters_regions = clusters['high_level_region'][clusters_pass]

# Select spikes and clusters in this brain region
clusters_in_region = clusters_pass[clusters_regions == 'Frontal cortex']

# Get binned spikes centered at stimulation onset
peth, binned_spikes = calculate_peths(spikes.times, spikes.clusters, clusters_in_region, opto_times,
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

# Fit HMM on all data
lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')

p_down_mat = np.empty((len(trial_data), time_ax.shape[0]))
state_mat = np.empty((len(trial_data), time_ax.shape[0])).astype(int)
for t in range(len(trial_data)):

    # Get posterior probability and most likely states for this trial
    posterior = simple_hmm.filter(trial_data[t])
    posterior = posterior[np.concatenate(([False], use_timepoints[:-1])), :]  
    zhat = simple_hmm.most_likely_states(trial_data[t])
    
    # Make sure 0 is down state and 1 is up state
    if np.mean(binned_spikes[t, :, zhat==0]) > np.mean(binned_spikes[t, :, zhat==1]):
        # State 0 is up state
        zhat = np.where((zhat==0)|(zhat==1), zhat^1, zhat)
        p_down = posterior[:, 1]
    else:
        p_down = posterior[:, 0]
            
    # Add to 2D arrays
    p_down_mat[t, :] = p_down
    state_mat[t, :] = zhat[use_timepoints]
    
# %% Plot example trial
trial = 3
colors, dpi = figure_style()
cmap = ListedColormap([colors['down-state'], colors['up-state']])
f, ax = plt.subplots(1, 1, figsize=(1.8, 1.75), dpi=dpi)

ax.add_patch(Rectangle((0, 0), 1, len(clusters_in_region), color='royalblue', alpha=0.25, lw=0))
tickedges = np.arange(0, len(clusters_in_region)+1)
for i, n in enumerate(clusters_in_region):
    idx = np.bitwise_and(spikes.times[spikes.clusters == n] >= opto_times[trial] - PRE_TIME,
                         spikes.times[spikes.clusters == n] <= opto_times[trial] + POST_TIME)
    neuron_spks = spikes.times[spikes.clusters == n][idx]
    ax.vlines(neuron_spks - opto_times[trial], tickedges[i + 1], tickedges[i], color='black',
              lw=0.5)

ax.set(xlabel='Time from stimulation start (s)', yticks=[0, len(clusters_in_region)],
       yticklabels=[1, len(clusters_in_region)], xticks=[-1, 0, 1, 2, 3, 4],
       ylim=[0, len(clusters_in_region)])
ax.set_ylabel('Neurons', labelpad=-5)

ax2 = ax.twinx()
ax2.plot(time_ax, p_down_mat[trial, :], color=colors['down-state'], lw=0.75)
ax2.set(ylim=[-0.01, 1.01], title='Frontal cortex')            
ax2.set_ylabel('P(down state)', rotation=270, labelpad=10)
ax2.yaxis.label.set_color(colors['down-state'])
ax2.tick_params(axis='y', colors=colors['down-state'])            
ax2.spines['right'].set_color(colors['down-state'])

sns.despine(trim=True, right=False)
plt.tight_layout()
plt.savefig(join(fig_path, 'example_updown_trial.pdf'))

# %% Plot example session
 
f, ax = plt.subplots(1, 1, figsize=(1.7, 1.75), dpi=dpi)
ax.imshow(np.flipud(state_mat), aspect='auto', cmap=cmap, vmin=0, vmax=1,
          extent=(-PRE_TIME, POST_TIME, 1, len(opto_times)))
ax.add_patch(Rectangle((0, 0), 1, len(opto_times), color='royalblue', alpha=0.25, lw=0))
ax.set(xlabel='Time from stimulation start (s)', yticks=[1, 50],
       xticks=[-1, 0, 1, 2, 3, 4], title='Frontal cortex')
ax.set_ylabel('Trials', labelpad=-8)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'example_updown_session.pdf'))

