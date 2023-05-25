#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:46:00 2022
By: Guido Meijer
"""

import numpy as np
from one.api import ONE
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, realpath, dirname, split
from stim_functions import (load_passive_opto_times, paths, get_artifact_neurons, figure_style,
                            get_neuron_qc)
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.processing import bincount2D
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# EID
#eid = '5531e71f-8ab9-4c4e-8d5b-d92da838ee16'
#pid_00 = 'cf08dda8-478f-4292-a06f-4c4dae9f8755'
#pid_01 = '9a9ee022-80b3-4a88-903f-12f838111818'

eid = '9ee8b642-9d7f-483f-93b4-5b2ed62c7653'
pid_00 = '542aa93b-4876-439c-93f7-dd77263e55e8'
pid_01 = '79f881cb-8140-47ed-8d1a-7cca7e66fb12'

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

# Load in opto times
opto_times, _ = load_passive_opto_times(eid, one=one)

# Load in spikes
print('Loading in spikes for probe 00')
sl = SpikeSortingLoader(eid=eid, pname='probe00', one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Select spikes that pass QC and are not artifact neurons
qc_metrics = get_neuron_qc(pid_00, one=one, ba=ba)
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
clusters = sl.merge_clusters(spikes, clusters, channels)

# Select spikes that pass QC and are not artifact neurons
qc_metrics = get_neuron_qc(pid_01, one=one, ba=ba)
clusters_pass = np.where(qc_metrics['label'] == 1)[0]
clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
    artifact_neurons['pid'] == pid_01, 'neuron_id'].values)]
spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
spikes.depths = spikes.depths[np.isin(spikes.clusters, clusters_pass)]
iok = ~np.isnan(spikes.depths)

# Get spike raster
R_01, times_01, depths_01 = bincount2D(spikes.times[iok], spikes.depths[iok], 0.02, 20, weights=None)

# %% Plot figure
trial = 3
n_secs = 4
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2), dpi=dpi)
ax1.imshow(R_00, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R_00),
          extent=np.r_[times_00[[0, -1]], depths_00[[0, -1]]], origin='lower')
ax1.plot([opto_times[trial], opto_times[trial]+1], [0, 0], color='b')
ax1.set(xlim=[opto_times[trial] - n_secs, opto_times[trial] + n_secs], ylim=[0, 4000], xticks=[])

ax2.imshow(R_01, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R_01),
          extent=np.r_[times_01[[0, -1]], depths_01[[0, -1]]], origin='lower')
ax2.plot([opto_times[trial], opto_times[trial]+1], [0, 0], color='b')
ax2.set(xlim=[opto_times[trial] - n_secs, opto_times[trial] + n_secs], ylim=[0, 4000], xticks=[])

f.suptitle(f'{eid}')

plt.tight_layout()
sns.despine(trim=True, bottom=True, offset=4)
plt.savefig(join(fig_path, 'example_raster_anesthesia.jpg'), dpi=2000)
