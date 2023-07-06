#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, realpath, dirname, split
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from matplotlib.patches import Rectangle
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from brainbox.plot import peri_event_time_histogram
from stim_functions import paths, load_passive_opto_times, figure_style
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings

SUBJECT = 'ZFM-03330'
DATE = '2022-02-17'
PROBE = 'probe00'
NEURON = 138
TITLE = 'Hippocampus (DG)'

T_BEFORE = 1  # for plotting
T_AFTER = 2
ZETA_BEFORE = 0  # baseline period to include for zeta test
PRE_TIME = [1, 0]  # for modulation index
POST_TIME = [0, 1]
BIN_SIZE = 0.05
SMOOTHING = 0.025

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Get session details
ins = one.alyx.rest('insertions', 'list', date=DATE, subject=SUBJECT, name=PROBE)
pid = ins[0]['id']
eid = ins[0]['session']

# Get peak latency from file
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
this_neuron = all_neurons[(all_neurons['pid'] == pid) & (all_neurons['neuron_id'] == NEURON)]
latency = all_neurons.loc[(all_neurons['pid'] == pid) & (all_neurons['neuron_id'] == NEURON),
                          'latency_peak_onset'].values[0]

# Load in laser pulse times
opto_train_times, _ = load_passive_opto_times(eid, one=one)

# Load in spikes
sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# %% Plot PSTH
colors, dpi = figure_style()
p, ax = plt.subplots(1, 1, figsize=(1.5, 2), dpi=dpi)
ax.add_patch(Rectangle((0, 0), 1, 100, color='royalblue', alpha=0.25, lw=0))
ax.add_patch(Rectangle((0, 0), 1, -100, color='royalblue', alpha=0.25, lw=0))

peri_event_time_histogram(spikes.times, spikes.clusters, opto_train_times,
                          NEURON, t_before=T_BEFORE, t_after=T_AFTER, smoothing=SMOOTHING,
                          bin_size=BIN_SIZE, include_raster=True, error_bars='sem', ax=ax,
                          pethline_kwargs={'color': 'black', 'lw': 1},
                          errbar_kwargs={'color': 'black', 'alpha': 0.3, 'lw': 0},
                          raster_kwargs={'color': 'black', 'lw': 0.3},
                          eventline_kwargs={'lw': 0})
ax.plot([-1.05, -1.05], [0, 10], color='k', lw=0.75, clip_on=False)
ax.text(-1.05, 5, '10 sp s$^{-1}$', ha='right', va='center', rotation=90)
ax.plot([0, 1], [ax.get_ylim()[0]-0.5, ax.get_ylim()[0]-0.5], color='k', lw=0.75, clip_on=False)
ax.text(0.5, ax.get_ylim()[0]-1, '1s', ha='center', va='top')
ax.axis('off')
ax.set(title=TITLE)

peths, _ = calculate_peths(spikes.times, spikes.clusters, [NEURON],
                           opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
peak_ind = np.argmin(np.abs(peths['tscale'] - latency))
peak_act = peths['means'][0][peak_ind]
ax.plot([latency, latency], [peak_act, peak_act], 'x', color='red', lw=2)
#ax.plot([latency, latency], [peak_act, 14], ls='--', color='red', lw=0.5)
ax.text(latency+0.15, 12, f'{latency*1000:.0f} ms', color='red', va='center', ha='left', fontsize=5)

plt.tight_layout()

plt.savefig(join(fig_path, f'latency_{SUBJECT}_{DATE}_{PROBE}_neuron{NEURON}.pdf'))

