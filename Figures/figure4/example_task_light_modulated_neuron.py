#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import pandas as pd
from os.path import join, realpath, dirname, split
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from brainbox.io.one import SpikeSortingLoader
from stim_functions import (paths, remap, load_passive_opto_times, figure_style, init_one,
                            load_trials, peri_multiple_events_time_histogram)
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = init_one()

# Settings

TITLE = 'Frontal cortex (ACA)'
SUBJECT = 'ZFM-02600'
DATE = '2021-08-25'
PROBE = 'probe00'
NEURON = 438
SCALEBAR = 15

"""
TITLE = 'Midbrain (PAG)'
SUBJECT = 'ZFM-05170'
DATE = '2022-12-08'
PROBE = 'probe00'
NEURON = 395
SCALEBAR = 15
"""

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

# Load in laser pulse times
opto_train_times, _ = load_passive_opto_times(eid, one=one)

# Load in trials
trials = load_trials(eid, laser_stimulation=True)
zero_contr_trials = trials[trials['signed_contrast'] == 0]

# Load in spikes
sl = SpikeSortingLoader(pid=pid, one=one)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Get region
region = remap(clusters.acronym[NEURON])[0]

# Retreive p-values
task_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
p_task = task_neurons.loc[(task_neurons['pid'] == pid) & (task_neurons['neuron_id'] == NEURON), 'opto_mod_p']
print(f'p value task: {p_task.values[0]}')

# %% Plot PSTH
colors, dpi = figure_style()
p, ax1 = plt.subplots(1, 1, figsize=(1.5, 2), dpi=dpi)

#ax1.add_patch(Rectangle((0, -100), 1, 200, color='royalblue', alpha=0.25, lw=0))
ax1.plot([0, 0], [-100, 100], color='k', ls='--', lw=0.5)
peri_multiple_events_time_histogram(
    spikes.times, spikes.clusters, zero_contr_trials['goCue_times'],
    zero_contr_trials['laser_stimulation'],
    NEURON, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=ax1,
    pethline_kwargs=[{'color': colors['no-stim'], 'lw': 1},
                     {'color': colors['stim'], 'lw': 1}],
    errbar_kwargs=[{'color': colors['no-stim'], 'alpha': 0.3, 'lw': 0},
                   {'color': colors['stim'], 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': colors['no-stim'], 'lw': 0.5},
                   {'color': colors['stim'], 'lw': 0.5}],
    eventline_kwargs={'lw': 0}, include_raster=True)
ax1.plot([-1.05, -1.05], [0, SCALEBAR], color='k', lw=0.75, clip_on=False)
ax1.text(-1.05, SCALEBAR/2, f'{SCALEBAR} sp s$^{-1}$', ha='right', va='center', rotation=90)
#ax1.plot([0.2, 1], [ax1.get_ylim()[1]-0.5, ax1.get_ylim()[1]-0.5], lw=0.5, color='k')
ax1.text(0.6, ax1.get_ylim()[1]-0.5, '***', ha='center', va='center', fontsize=10)
ax1.plot([0, 1], [ax1.get_ylim()[0]-0.5, ax1.get_ylim()[0]-0.5], color='k', lw=0.75, clip_on=False)
ax1.text(0.5, ax1.get_ylim()[0]-2, '1s', ha='center', va='top')
ax1.axis('off')
ax1.set(title=TITLE)

plt.tight_layout()

plt.savefig(join(fig_path, f'{region}_{SUBJECT}_{DATE}_{PROBE}_neuron{NEURON}.pdf'))