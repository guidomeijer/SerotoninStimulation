#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
np.random.seed(42)
from os.path import join, isdir
from os import mkdir
import pandas as pd
from copy import deepcopy
import random
import shutil
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from brainbox.io.one import SpikeSortingLoader
from brainbox.task.closed_loop import generate_pseudo_blocks
from stim_functions import (paths, combine_regions, load_subjects, binned_rate_timewarped,
                            high_level_regions, load_trials, calculate_peths)
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
scaler = StandardScaler()

# Settings
SPLITS = ['L_opto', 'R_opto', 'L_no_opto', 'R_no_opto', 'L', 'R', 'opto', 'no_opto']
#CENTER_ON = 'firstMovement_times'
CENTER_ON = 'stimOn_times'

BIN_SIZE = 0.0125
SMOOTHING = 0.02
#T_BEFORE = 0.3
#T_AFTER = 0
T_BEFORE = 0
T_AFTER = 0.4

# Good values
#MIN_RT = 0.2
#MAX_RT = 1

MIN_FR = 0.1
MIN_RT = 0.1
MAX_RT = 1.2
MIN_TRIALS = 10  # per split
N_SHUFFLES = 500

# Set paths
# These data are too large to put on the repo so will be saved in the one cache dir
_, s_path = paths(save_dir='cache')
save_path = join(s_path, 'manifold', f'{CENTER_ON}')
if isdir(save_path):
    shutil.rmtree(save_path)
if not isdir(save_path):
    mkdir(save_path)
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
    peth_dict, peths_shuf = dict(), dict()

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
    all_trials = load_trials(eid, laser_stimulation=True)
    
    # Exclude trials with too short or too long reaction times and exclude probe trials
    incl_trials = (((all_trials['reaction_times'] <= MAX_RT) & (all_trials['reaction_times'] >= MIN_RT))
                   & (all_trials['probe_trial'] == 0))
    
    #incl_trials = ((all_trials['reaction_times'] <= MAX_RT) & (all_trials['reaction_times'] >= MIN_RT))
    
    trials = all_trials[incl_trials]
    
    # Check if there are enough trials for each split
    if ((trials[(trials['choice'] == -1) & (trials['laser_stimulation'] == 1)].shape[0] < MIN_TRIALS)
        | (trials[(trials['choice'] == 1) & (trials['laser_stimulation'] == 1)].shape[0] < MIN_TRIALS)
        | (trials[(trials['choice'] == -1) & (trials['laser_stimulation'] == 0)].shape[0] < MIN_TRIALS)
        | (trials[(trials['choice'] == 1) & (trials['laser_stimulation'] == 0)].shape[0] < MIN_TRIALS)):
        print('Not enough trials for one of the splits')
        continue
    
    # Get peri-event time histogram and binned spikes for all trials
    peths, binned_spikes = calculate_peths(spikes.times, spikes.clusters,
                                           these_neurons['neuron_id'].values,
                                           trials[CENTER_ON],
                                           T_BEFORE, T_AFTER,
                                           BIN_SIZE, SMOOTHING)
    
    for split in SPLITS:
        
        # Split trials and get the mean spike rate over trials
        if split == 'L_opto':
            peth_dict[split] = np.mean(binned_spikes[
                (trials['choice'] == -1) & (trials['laser_stimulation'] == 1), :, :], axis=0)
        elif split == 'R_opto':
            peth_dict[split] = np.mean(binned_spikes[
                (trials['choice'] == 1) & (trials['laser_stimulation'] == 1), :, :], axis=0)
        elif split == 'L_no_opto':
            peth_dict[split] = np.mean(binned_spikes[
                (trials['choice'] == -1) & (trials['laser_stimulation'] == 0), :, :], axis=0)
        elif split == 'R_no_opto':
            peth_dict[split] = np.mean(binned_spikes[
                (trials['choice'] == 1) & (trials['laser_stimulation'] == 0), :, :], axis=0)
        elif split == 'L':
            peth_dict[split] = np.mean(binned_spikes[trials['choice'] == -1, :, :], axis=0)
        elif split == 'R':
            peth_dict[split] = np.mean(binned_spikes[trials['choice'] == 1, :, :], axis=0)
        elif split == 'no_opto':
            peth_dict[split] = np.mean(binned_spikes[trials['laser_stimulation'] == 0, :, :], axis=0)
        elif split == 'opto':
            peth_dict[split] = np.mean(binned_spikes[trials['laser_stimulation'] == 1, :, :], axis=0)
        
        
    # Get shuffled data
    for ii in range(N_SHUFFLES):
        
        # Shuffle choices 
        y_ = trials['probabilityLeft'].values
      
        # Permute choice labels among trials with the same block and stimulus side
        stis = trials['contrastLeft'].values
        c0 = np.bitwise_and(y_ == 0.8, np.isnan(stis))
        c1 = np.bitwise_and(y_ != 0.8, np.isnan(stis))
        c2 = np.bitwise_and(y_ == 0.8, ~np.isnan(stis))
        c3 = np.bitwise_and(y_ != 0.8, ~np.isnan(stis))
        shuffled_choices = deepcopy(trials['choice'])

        # shuffle choices within each class
        for cc in [c0, c1, c2, c3]:
            r = trials['choice'][cc]
            shuffled_choices[cc] = np.array(random.sample(list(r), len(r)))
        
        # Split trials
        L_peth = np.mean(binned_spikes[shuffled_choices == -1, :, :], axis=0)
        R_peth = np.mean(binned_spikes[shuffled_choices == 1, :, :], axis=0)
        
        # Create artifical stimulation blocks as random permutation
        laser_block_len = int(np.random.exponential(60))
        while (laser_block_len <= 20) | (laser_block_len >= 100):
            laser_block_len = int(np.random.exponential(60))
        if np.round(np.random.rand()) == 0:
            random_blocks = np.tile([0] * laser_block_len + [1] * laser_block_len, 100)
        else:
            random_blocks = np.tile([1] * laser_block_len + [0] * laser_block_len, 100)
        rand_offset = np.random.randint(100)
        artifical_opto_blocks = random_blocks[rand_offset:all_trials.shape[0] + rand_offset]
        permut_opto = artifical_opto_blocks[incl_trials]
        
        """
        permut_opto = shuffle(trials['laser_stimulation'].values)
        """
        
        # Split trials
        opto_peth = np.mean(binned_spikes[permut_opto == 1, :, :], axis=0)
        no_opto_peth = np.mean(binned_spikes[permut_opto == 0, :, :], axis=0)
         
        
        # For data split four ways shuffle the neural data over trials
        binned_spikes_shuf = np.random.permutation(binned_spikes)
        L_opto_peth = np.mean(binned_spikes_shuf[(trials['choice'] == -1) & (trials['laser_stimulation'] == 1), :, :], axis=0)
        L_no_opto_peth = np.mean(binned_spikes_shuf[(trials['choice'] == -1) & (trials['laser_stimulation'] == 0), :, :], axis=0)
        R_opto_peth = np.mean(binned_spikes_shuf[(trials['choice'] == 1) & (trials['laser_stimulation'] == 1), :, :], axis=0)
        R_no_opto_peth = np.mean(binned_spikes_shuf[(trials['choice'] == 1) & (trials['laser_stimulation'] == 0), :, :], axis=0)
                
        # Add to 3d array
        if ii == 0:
            peths_shuf['L'] = L_peth
            peths_shuf['R'] = R_peth
            peths_shuf['opto'] = opto_peth
            peths_shuf['no_opto'] = no_opto_peth
            peths_shuf['L_opto'] = L_opto_peth
            peths_shuf['L_no_opto'] = L_no_opto_peth
            peths_shuf['R_opto'] = R_opto_peth
            peths_shuf['R_no_opto'] = R_no_opto_peth
        else:
            peths_shuf['L'] = np.dstack((peths_shuf['L'], L_peth))
            peths_shuf['R'] = np.dstack((peths_shuf['R'], R_peth))
            peths_shuf['opto'] = np.dstack((peths_shuf['opto'], opto_peth))
            peths_shuf['no_opto'] = np.dstack((peths_shuf['no_opto'], no_opto_peth))
            peths_shuf['L_opto'] = np.dstack((peths_shuf['L_opto'], L_opto_peth))
            peths_shuf['L_no_opto'] = np.dstack((peths_shuf['L_no_opto'], L_no_opto_peth))
            peths_shuf['R_opto'] = np.dstack((peths_shuf['R_opto'], R_opto_peth))
            peths_shuf['R_no_opto'] = np.dstack((peths_shuf['R_no_opto'], R_no_opto_peth))
    
    # Save output
    peth_dict['time'] = peths['tscale']
    peth_dict['shuffle'] = peths_shuf
    peth_dict['region'] = these_neurons['region'].values
    peth_dict['subject'] = subject
    peth_dict['sert-cre'] = sert_cre
    peth_dict['date'] = date
    peth_dict['probe'] = probe
    peth_dict['eid'] = eid
    peth_dict['n_shuffles'] = N_SHUFFLES
    peth_dict['center_on'] = CENTER_ON
    
    np.save(join(save_path, f'{pid}.npy'), peth_dict)

