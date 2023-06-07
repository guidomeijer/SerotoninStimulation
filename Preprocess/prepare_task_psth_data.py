#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from stim_functions import (paths, load_passive_opto_times, combine_regions, load_subjects,
                            high_level_regions, load_trials)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
SPLITS = ['L_opto', 'R_opto', 'L_no_opto', 'R_no_opto']
CENTER_ON = 'firstMovement_times'
BIN_SIZE = 0.0125
SMOOTHING = 0.02
T_BEFORE = 0.1
T_AFTER = 0
MIN_FR = 0.1
MIN_RT = 0.1
MAX_RT = 1
_, save_path = paths()

# Load in light modulated neurons
task_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
task_neurons['full_region'] = combine_regions(task_neurons['region'])
task_neurons['high_level_region'] = high_level_regions(task_neurons['region'], input_atlas='Beryl')
task_neurons = task_neurons[task_neurons['full_region'] != 'root']

# Load subject info
subjects = load_subjects()

# %% Loop over sessions
peths_df = pd.DataFrame()
for i, pid in enumerate(np.unique(task_neurons['pid'])):

    # Get session details
    eid = np.unique(task_neurons.loc[task_neurons['pid'] == pid, 'eid'])[0]
    probe = np.unique(task_neurons.loc[task_neurons['pid'] == pid, 'probe'])[0]
    subject = np.unique(task_neurons.loc[task_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(task_neurons.loc[task_neurons['pid'] == pid, 'date'])[0]
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'Starting {subject}, {date}')

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    # Take slice of dataframe
    these_neurons = task_neurons[task_neurons['pid'] == pid]
        
    # Load in trials
    trials = load_trials(eid, laser_stimulation=True)
    
    # Exclude trials with too short or too long reaction times 
    trials = trials[(trials['reaction_times'] <= MAX_RT) & (trials['reaction_times'] >= MIN_RT)]

    for split in SPLITS:
        # Split trials
        if split == 'L_opto':
            trials_split = trials[(trials['choice'] == -1) & (trials['laser_stimulation'] == 1)]
        elif split == 'R_opto':
            trials_split = trials[(trials['choice'] == 1) & (trials['laser_stimulation'] == 1)]
        elif split == 'L_no_opto':
            trials_split = trials[(trials['choice'] == -1) & (trials['laser_stimulation'] == 0)]
        elif split == 'R_no_opto':
            trials_split = trials[(trials['choice'] == 1) & (trials['laser_stimulation'] == 0)]

        # Get peri-event time histogram
        peths, _ = calculate_peths(spikes.times, spikes.clusters, these_neurons['neuron_id'].values,
                                   trials_split[CENTER_ON], T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
        tscale = peths['tscale']
    
        # Loop over neurons
        for n, index in enumerate(these_neurons.index.values):
            if np.mean(peths['means'][n, :]) > MIN_FR:
    
                # Add to dataframe
                peths_df = pd.concat((peths_df, pd.DataFrame(index=[peths_df.shape[0]], data={
                    'peth': [peths['means'][n, :]],  'time': [tscale], 'split': split,
                    'region': these_neurons.loc[index, 'full_region'],
                    'high_level_region': these_neurons.loc[index, 'high_level_region'],
                    'firing_rate': np.mean(peths['means'][n, :]),
                    'neuron_id': these_neurons.loc[index, 'neuron_id'],
                    'subject': these_neurons.loc[index, 'subject'],
                    'eid': these_neurons.loc[index, 'eid'],
                    'acronym': these_neurons.loc[index, 'region'],
                    'probe': probe, 'date': date, 'pid': pid, 'sert-cre': sert_cre})))

# Save output
peths_df.to_pickle(join(save_path, 'psth_task.pickle'))

