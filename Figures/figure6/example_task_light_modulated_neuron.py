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
from brainbox.plot import peri_event_time_histogram
from stim_functions import (paths, remap, load_passive_opto_times, figure_style, init_one,
                            load_trials, peri_multiple_events_time_histogram)
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = init_one()

# Settings
TITLE = ''
SUBJECT = 'ZFM-02600'
DATE = '2021-08-26'
PROBE = 'probe01'
NEURON = 209
SCALEBAR = 30

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
awake_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
task_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
awake_neurons = awake_neurons[awake_neurons['modulated']]
task_neurons = task_neurons[task_neurons['opto_modulated']]
merged_df = pd.merge(awake_neurons, task_neurons, on=[
    'pid', 'neuron_id', 'eid', 'subject', 'date', 'probe', 'region'])
p_passive = merged_df.loc[(merged_df['pid'] == pid) & (merged_df['neuron_id'] == NEURON), 'p_value']
p_task = merged_df.loc[(merged_df['pid'] == pid) & (merged_df['neuron_id'] == NEURON), 'opto_mod_p']
print(f'p value passive: {p_passive.values[0]}')
print(f'p value task: {p_task.values[0]}')

# %% Plot PSTH
colors, dpi = figure_style()
p, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -100), 1, 200, color='royalblue', alpha=0.25, lw=0))
peri_event_time_histogram(spikes.times, spikes.clusters, opto_train_times,
                          NEURON, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                          smoothing=SMOOTHING,  include_raster=True, error_bars='sem', ax=ax1,
                          pethline_kwargs={'color': 'black', 'lw': 1},
                          errbar_kwargs={'color': 'black', 'alpha': 0.3, 'lw': 0},
                          raster_kwargs={'color': 'black', 'lw': 0.3},
                          eventline_kwargs={'lw': 0})
ax1.plot([-1.05, -1.05], [0, SCALEBAR], color='k', lw=0.75, clip_on=False)
ax1.text(-1.05, SCALEBAR/2, f'{SCALEBAR} sp s$^{-1}$', ha='right', va='center', rotation=90)
ax1.text(0.5, 57, '***', ha='center', va='center', fontsize=10)
ax1.plot([0, 1], [ax1.get_ylim()[0]-0.5, ax1.get_ylim()[0]-0.5], color='k', lw=0.75, clip_on=False)
ax1.text(0.5, ax1.get_ylim()[0]-2, '1s', ha='center', va='top')
ax1.axis('off')
ax1.set(title='Passive stimulation')

ax2.add_patch(Rectangle((0, -100), 1, 200, color='royalblue', alpha=0.25, lw=0))
peri_multiple_events_time_histogram(
    spikes.times, spikes.clusters, zero_contr_trials['goCue_times'],
    zero_contr_trials['laser_stimulation'],
    NEURON, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=ax2,
    pethline_kwargs=[{'color': colors['no-stim'], 'lw': 1},
                     {'color': colors['stim'], 'lw': 1}],
    errbar_kwargs=[{'color': colors['no-stim'], 'alpha': 0.3, 'lw': 0},
                   {'color': colors['stim'], 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': colors['no-stim'], 'lw': 0.5},
                   {'color': colors['stim'], 'lw': 0.5}],
    eventline_kwargs={'lw': 0}, include_raster=True)
ax2.plot([-1.05, -1.05], [0, SCALEBAR], color='k', lw=0.75, clip_on=False)
ax2.text(-1.05, SCALEBAR/2, f'{SCALEBAR} sp s$^{-1}$', ha='right', va='center', rotation=90)
ax2.text(0.5, 43.5, '**', ha='center', va='center', fontsize=10)
ax2.plot([0, 1], [ax2.get_ylim()[0]-0.5, ax2.get_ylim()[0]-0.5], color='k', lw=0.75, clip_on=False)
ax2.text(0.5, ax2.get_ylim()[0]-2, '1s', ha='center', va='top')
ax2.axis('off')
ax2.set(title='Trial start')

plt.tight_layout()

plt.savefig(join(fig_path, f'{region}_{SUBJECT}_{DATE}_{PROBE}_neuron{NEURON}.pdf'))