#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:46:00 2022
By: Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, realpath, dirname, split
from stim_functions import (load_passive_opto_times, paths, get_artifact_neurons, figure_style,
                            init_one)
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.processing import bincount2D
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
one = init_one()
ba = AllenAtlas()

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# EID
eid = '0d24afce-9d3c-449e-ac9f-577eefefbd7e'
pid_00 = 'ff02e36b-95e9-4985-a6e3-ba2977063496'
pid_01 = 'cc8dd669-7938-4594-868d-b8ca0663b69a'

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

# Load in opto times
opto_times, _ = load_passive_opto_times(eid, one=one)

# Load in spikes
print('Loading in spikes for probe 00')
sl = SpikeSortingLoader(eid=eid, pname='probe00', one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()

# Select spikes that pass QC and are not artifact neurons
qc_metrics, _ = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths,
                                      cluster_ids=np.arange(clusters.channels.size))
clusters_pass = np.where(qc_metrics['label'] == 1)[0]
clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
    artifact_neurons['pid'] == pid_00, 'neuron_id'].values)]
spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
spikes.depths = spikes.depths[np.isin(spikes.clusters, clusters_pass)]
iok = ~np.isnan(spikes.depths)

# Get spike raster
R_00, times_00, depths_00 = bincount2D(spikes.times[iok], spikes.depths[iok], 0.02, 20, weights=None)

# Load in spikes
print('Loading in spikes for probe 01')
sl = SpikeSortingLoader(eid=eid, pname='probe01', one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()

# Select spikes that pass QC and are not artifact neurons
qc_metrics, _ = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths,
                                      cluster_ids=np.arange(clusters.channels.size))
clusters_pass = np.where(qc_metrics['label'] == 1)[0]
clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
    artifact_neurons['pid'] == pid_01, 'neuron_id'].values)]
spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
spikes.depths = spikes.depths[np.isin(spikes.clusters, clusters_pass)]
iok = ~np.isnan(spikes.depths)

# Get spike raster
R_01, times_01, depths_01 = bincount2D(spikes.times[iok], spikes.depths[iok], 0.02, 20, weights=None)

# %% Plot figure
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 2.5), dpi=dpi)
ax1.imshow(R_00, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R_00) * 2,
          extent=np.r_[times_00[[0, -1]], depths_00[[0, -1]]], origin='lower')
ax1.set(xlim=[opto_times[0] - 4, opto_times[0] + 5.5], ylim=[0, 4000],
       xticks=[])
ax1.plot([opto_times[0], opto_times[0]+1], [0, 0], color='b')
ax1.plot([opto_times[1], opto_times[1]+1], [0, 0], color='b')
ax1.plot([opto_times[2], opto_times[2]+1], [0, 0], color='b')

ax2.imshow(R_01, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R_01) * 2,
          extent=np.r_[times_01[[0, -1]], depths_01[[0, -1]]], origin='lower')
ax2.set(xlim=[opto_times[0] - 4, opto_times[0] + 5.5], ylim=[0,4000],
       xticks=[])
ax2.plot([opto_times[0], opto_times[0]+1], [0, 0], color='b')
ax2.plot([opto_times[1], opto_times[1]+1], [0, 0], color='b')
ax2.plot([opto_times[2], opto_times[2]+1], [0, 0], color='b')

f.suptitle(f'{eid}')

plt.tight_layout()
sns.despine(trim=True, offset=4)
plt.savefig(join(fig_path, 'example_raster.pdf'))
plt.savefig(join(fig_path, 'example_raster.jpg'), dpi=2000)
