# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:26:54 2023

@author: Guido
"""


import numpy as np
from os.path import join, split, isfile
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
from stim_functions import (paths, query_ephys_sessions, load_passive_opto_times, load_subjects,
                            figure_style, remap, high_level_regions, combine_regions,
                            init_one, get_dlc_XYs, get_raw_smooth_pupil_diameter)
one = init_one()

# Settings
OVERWITE = True

# Get paths
parent_fig_path, repo_path = paths()
_, cache_path = paths(save_dir='cache')
data_path = join(repo_path, 'HMM', 'PassiveEvent')
rec_files = glob(join(data_path, '*.pickle'))
subjects = load_subjects()

if OVERWITE:
    state_pupil_df = pd.DataFrame()
else:
    state_pupil_df = pd.read_csv(join(repo_path, 'state_pupil_corr_baseline.csv'))

for i, file_path in enumerate(rec_files):
    
    # Get info
    print(f'Starting {i} of {len(rec_files)}')
    subject = split(file_path)[1][:9]
    date = split(file_path)[1][10:20]
    eid = one.search(subject=subject, date=date)[0]
    if subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0] == 0:
        continue
    
    # Load in data    
    with open(file_path, 'rb') as handle:
        hmm_dict = pickle.load(handle)    
    prob_mat, time_ax = hmm_dict['prob_mat'], hmm_dict['time_ax']
    bin_size = np.mean(np.diff(time_ax))
    n_states = hmm_dict['prob_mat'].shape[2]
    region = split(file_path)[-1].split('_')[-1].split('.')[0]
    bl_time_ax = time_ax[time_ax < 0]

    # Load in pupil diameter
    if not isfile(join(cache_path, 'PupilDiameter', f'{hmm_dict["eid"]}_diameter.npy')):
        print('Loading and smoothing pupil trace..')
        video_times, XYs = get_dlc_XYs(one, eid)
        if XYs is None:
            continue
        _, diameter = get_raw_smooth_pupil_diameter(XYs)
        np.save(join(cache_path, 'PupilDiameter', f'{hmm_dict["eid"]}_diameter.npy'), diameter)
        np.save(join(cache_path, 'PupilDiameter', f'{hmm_dict["eid"]}_times.npy'), video_times)
    else:
        diameter = np.load(join(cache_path, 'PupilDiameter', f'{eid}_diameter.npy'))
        video_times = np.load(join(cache_path, 'PupilDiameter', f'{eid}_times.npy'))
     
    """
    # Get average pupil diameter during states of passive events
    state_pupil = np.empty((hmm_dict['event_times'].shape[0], bl_time_ax.shape[0]))
    for i_e, this_event in enumerate(hmm_dict['event_times']):
        this_diameter = np.empty(bl_time_ax.shape[0])
        for i_tb, this_bin in enumerate(bl_time_ax + this_event):
            this_diameter[i_tb] = np.nanmean(diameter[(video_times > this_bin - (bin_size/2)) & 
                                                      (video_times <= this_bin + (bin_size/2))])
        state_pupil[i_e, :] = this_diameter
    
    # Correlate pupil with state probability
    r_state, p_state = np.empty(n_states), np.empty(n_states)
    for this_state in range(n_states):
        this_prob = prob_mat[:, time_ax < 0, this_state].flatten()
        this_pupil = state_pupil.flatten()
        if this_pupil[~np.isnan(this_pupil)].shape[0] < 10:
            continue
        r_state[this_state], p_state[this_state] = stats.pearsonr(
            this_prob[~np.isnan(this_pupil)], this_pupil[~np.isnan(this_pupil)])
    """
    
    # Get average pupil diameter during states of passive events
    state_pupil = np.empty(hmm_dict['event_times'].shape[0])
    for i_e, this_event in enumerate(hmm_dict['event_times']):
        state_pupil[i_e] = np.nanmean(diameter[(video_times > this_event + time_ax[0]) & 
                                               (video_times <= this_event)])
    if np.sum(np.isnan(state_pupil)) == state_pupil.shape[0]:
        continue
    
    # Correlate pupil with state probability
    r_state, p_state = np.empty(n_states), np.empty(n_states)
    for this_state in range(n_states):  
        state_prob = np.mean(prob_mat[:, time_ax < 0, this_state], axis=1)
        r_state[this_state], p_state[this_state] = stats.pearsonr(
            state_prob[~np.isnan(state_pupil)], state_pupil[~np.isnan(state_pupil)])
    
    # Add to dataframe
    subject = split(file_path)[-1].split('_')[0]
    date = split(file_path)[-1].split('_')[1]
    region = split(file_path)[-1].split('_')[2][:-7]
    file_name = split(file_path)[-1]
    state_pupil_df = pd.concat((state_pupil_df, pd.DataFrame(data={
        'state': np.arange(n_states), 'r': r_state, 'p': p_state,
        'subject': subject, 'date': date, 'region': region,
        'file_name': file_name})))
    state_pupil_df.to_csv(join(repo_path, 'state_pupil_corr_baseline.csv'), index=False)
        
        
            
        
    