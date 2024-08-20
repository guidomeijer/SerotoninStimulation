#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir
from os import mkdir
import pandas as pd
from copy import deepcopy
import random
from brainbox.io.one import SpikeSortingLoader
from brainbox.task.closed_loop import generate_pseudo_blocks
from stim_functions import (paths, combine_regions, load_subjects,
                            high_level_regions, load_trials, calculate_peths)
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
SPLIT_ON = 'choice'
SPLITS = ['L_opto', 'R_opto', 'L_no_opto', 'R_no_opto']
CENTER_ON = 'firstMovement_times'
BIN_SIZE = 0.0125
SMOOTHING = 0.02
T_BEFORE = 0.15
T_AFTER = 0
MIN_FR = 0.1
MIN_RT = 0.1
MAX_RT = 1
MIN_TRIALS = 10  # per split
N_SHUFFLES = 500

# Set paths
# These data are too large to put on the repo so will be saved in the one cache dir
_, s_path = paths(save_dir='cache')
if not isdir(join(s_path, 'manifold', SPLIT_ON)):
    mkdir(join(s_path, 'manifold', SPLIT_ON))
save_path = join(s_path, 'manifold', SPLIT_ON)
_, load_path = paths(save_dir='repo')

# Load in light modulated neurons
task_neurons = pd.read_csv(join(load_path, 'task_modulated_neurons.csv'))
task_neurons['full_region'] = combine_regions(task_neurons['region'])
task_neurons['high_level_region'] = high_level_regions(task_neurons['region'], input_atlas='Beryl')
task_neurons = task_neurons[task_neurons['full_region'] != 'root']

# Load subject info
subjects = load_subjects()

# %% Loop over sessions
for i, pid in enumerate(np.unique(task_neurons['pid'])):
    peth_dict, peths_choice_shuf, peths_opto_shuf = dict(), dict(), dict()

    # Get session details
    eid = np.unique(task_neurons.loc[task_neurons['pid'] == pid, 'eid'])[0]
    probe = np.unique(task_neurons.loc[task_neurons['pid'] == pid, 'probe'])[0]
    subject = np.unique(task_neurons.loc[task_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(task_neurons.loc[task_neurons['pid'] == pid, 'date'])[0]
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'Starting session {i} of {len(np.unique(task_neurons["pid"]))}')

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
    
    # Check if there are enough trials for each split
    if ((trials[(trials[SPLIT_ON] == -1) & (trials['laser_stimulation'] == 1)].shape[0] < MIN_TRIALS)
        | (trials[(trials[SPLIT_ON] == 1) & (trials['laser_stimulation'] == 1)].shape[0] < MIN_TRIALS)
        | (trials[(trials[SPLIT_ON] == -1) & (trials['laser_stimulation'] == 0)].shape[0] < MIN_TRIALS)
        | (trials[(trials[SPLIT_ON] == 1) & (trials['laser_stimulation'] == 0)].shape[0] < MIN_TRIALS)):
        print('Not enough trials for one of the splits')
        continue

    # Get peri-event time histogram and binned spikes for all trials
    peths, binned_spikes = calculate_peths(spikes.times, spikes.clusters,
                                           these_neurons['neuron_id'].values,
                                           trials[CENTER_ON],
                                           T_BEFORE, T_AFTER,
                                           BIN_SIZE, SMOOTHING)
    
    for split in SPLITS:
        
        # Split trials and get mean spike rate per split
        if split == 'L_opto':
            peth_dict[split] = np.mean(binned_spikes[
                (trials[SPLIT_ON] == -1) & (trials['laser_stimulation'] == 1), :, :], axis=0)
        elif split == 'R_opto':
            peth_dict[split] = np.mean(binned_spikes[
                (trials[SPLIT_ON] == 1) & (trials['laser_stimulation'] == 1), :, :], axis=0)
        elif split == 'L_no_opto':
            peth_dict[split] = np.mean(binned_spikes[
                (trials[SPLIT_ON] == -1) & (trials['laser_stimulation'] == 0), :, :], axis=0)
        elif split == 'R_no_opto':
            peth_dict[split] = np.mean(binned_spikes[
                (trials[SPLIT_ON] == 1) & (trials['laser_stimulation'] == 0), :, :], axis=0)
        
    # Get shuffled data
    for ii in range(N_SHUFFLES):
        
        # Generate pseudo trials
        y_ = trials['probabilityLeft'].values
        if SPLIT_ON == 'choice':
            # Permute choice labels among trials with the same block and stimulus side
            stis = trials['contrastLeft'].values
            c0 = np.bitwise_and(y_ == 0.8, np.isnan(stis))
            c1 = np.bitwise_and(y_ != 0.8, np.isnan(stis))
            c2 = np.bitwise_and(y_ == 0.8, ~np.isnan(stis))
            c3 = np.bitwise_and(y_ != 0.8, ~np.isnan(stis))
        elif SPLIT_ON == 'stim_side':
            # Permute stimulus side labels among trials with the same block and choice
            stis = trials['choice'].values
            c0 = np.bitwise_and(y_ == 0.8, stis == 1)
            c1 = np.bitwise_and(y_ != 0.8, stis == 1)
            c2 = np.bitwise_and(y_ == 0.8, stis == -1)
            c3 = np.bitwise_and(y_ != 0.8, stis == -1)
        tr_c = trials[SPLIT_ON]  # true split
        tr_c2 = deepcopy(tr_c)

        # shuffle choices within each class
        for cc in [c0, c1, c2, c3]:
            r = tr_c[cc]
            tr_c2[cc] = np.array(random.sample(list(r), len(r)))
        
        for split in SPLITS:
            
            # Split trials
            if split == 'L_opto':
                this_peth = np.mean(binned_spikes[
                    (tr_c2 == -1) & (trials['laser_stimulation'] == 1), :, :], axis=0)
            elif split == 'R_opto':
                this_peth = np.mean(binned_spikes[
                    (tr_c2 == 1) & (trials['laser_stimulation'] == 1), :, :], axis=0)
            elif split == 'L_no_opto':
                this_peth = np.mean(binned_spikes[
                    (tr_c2 == -1) & (trials['laser_stimulation'] == 0), :, :], axis=0)
            elif split == 'R_no_opto':
                this_peth = np.mean(binned_spikes[
                    (tr_c2 == 1) & (trials['laser_stimulation'] == 0), :, :], axis=0)
    
            # Add to 3d array
            if ii == 0:
                peths_choice_shuf[split] = this_peth
            else:
                peths_choice_shuf[split] = np.dstack((peths_choice_shuf[split], this_peth))
                
        # Generate pseudo stimulation blocks for opto shuffle
        opto_pseudo = generate_pseudo_blocks(trials.shape[0], first5050=0)
        opto_pseudo = opto_pseudo == 0.2
        
        for split in SPLITS:
                        
            # Split trials
            if split == 'L_opto':
                this_peth = np.mean(binned_spikes[
                    (trials[SPLIT_ON] == -1) & (opto_pseudo == 1), :, :], axis=0)
            elif split == 'R_opto':
                this_peth = np.mean(binned_spikes[
                    (trials[SPLIT_ON] == 1) & (opto_pseudo == 1), :, :], axis=0)
            elif split == 'L_no_opto':
                this_peth = np.mean(binned_spikes[
                    (trials[SPLIT_ON] == -1) & (opto_pseudo == 0), :, :], axis=0)
            elif split == 'R_no_opto':
                this_peth = np.mean(binned_spikes[
                    (trials[SPLIT_ON] == 1) & (opto_pseudo == 0), :, :], axis=0)
    
            # Add to 3d array
            if ii == 0:
                peths_opto_shuf[split] = this_peth
            else:
                peths_opto_shuf[split] = np.dstack((peths_opto_shuf[split], this_peth))
            
    # Save output
    peth_dict['time'] = peths['tscale']
    peth_dict['choice_shuffle'] = peths_choice_shuf
    peth_dict['opto_shuffle'] = peths_opto_shuf
    peth_dict['region'] = these_neurons['region'].values
    peth_dict['subject'] = subject
    peth_dict['sert-cre'] = sert_cre
    peth_dict['date'] = date
    peth_dict['probe'] = probe
    peth_dict['eid'] = eid
    peth_dict['n_shuffles'] = N_SHUFFLES
    peth_dict['center_on'] = CENTER_ON
    peth_dict['split_on'] = SPLIT_ON
    
    np.save(join(save_path, f'{pid}.npy'), peth_dict)

