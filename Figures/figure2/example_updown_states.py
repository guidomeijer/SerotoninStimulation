# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:46:43 2023

@author: Guido
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from brainbox.singlecell import calculate_peths
from brainbox.io.one import SpikeSortingLoader
from scipy.signal import welch
from stim_functions import (query_ephys_sessions, high_level_regions, get_neuron_qc, remap,
                            figure_style, load_passive_opto_times, paths)
from brainbox.processing import bincount2D
from os.path import join, realpath, dirname, split
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

PID = '04954136-75a8-4a20-9054-37b0bffd3b8b'
BIN_SIZE = 0.2
SMOOTHING = 0.4

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in neural data
sl = SpikeSortingLoader(pid=PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Filter neurons that pass QC
qc_metrics = get_neuron_qc(PID, one=one, ba=ba)
clusters_pass = np.where(qc_metrics['label'] == 1)[0]
spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

# Remap to high level regions
clusters.regions = high_level_regions(remap(clusters.acronym))

# Get spikes in region
region_spikes = spikes.times[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == 'Frontal cortex'])]
region_clusters = spikes.clusters[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == 'Frontal cortex'])]

# Get smoothed firing rates
peth, _ = calculate_peths(region_spikes, region_clusters, np.unique(region_clusters),
                          [region_spikes[-1]-603], pre_time=0, post_time=15,
                          bin_size=BIN_SIZE, smoothing=SMOOTHING)
tscale = peth['tscale'] + spikes.times[0]            
pop_act = peth['means'].T

# Get spike raster
raster_clusters = region_clusters[(region_spikes >= region_spikes[-1] - 603)
                              & (region_spikes <= region_spikes[-1] - 590)]
raster_spikes = region_spikes[(region_spikes >= region_spikes[-1] - 603)
                              & (region_spikes <= region_spikes[-1] - 590)]
R, times, depths = bincount2D(raster_spikes, raster_clusters, xbin=0.05, ybin=5, weights=None)
   
        
# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=dpi)

ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R),
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
ax1.plot([times[0], times[0]+3], [depths[0]-10, depths[0]-10], color='k', lw=0.75)
ax1.text(times[0]+1.5, depths[0]-30, '3s', ha='center', va='center')
ax1.plot([times[0]-.5, times[0]-.5], [1300, 1400], color='k', lw=0.75)
ax1.text(times[0]-1.2, 1350, '0.1 mm', rotation=90, ha='center', va='center')
ax2 = ax1.twinx()
ax2.plot(tscale + region_spikes[-1]-603, np.mean(pop_act, axis=1), color=colors['up-state'], lw=0.75)
ax2.set(ylim=[-0.3, 4])
ax1.axis('off')
ax2.axis('off')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'example_updown_state.pdf'))

