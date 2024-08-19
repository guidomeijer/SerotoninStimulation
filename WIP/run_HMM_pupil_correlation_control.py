# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:26:54 2023

@author: Guido
"""


import numpy as np
from os.path import join, realpath, dirname, split
import pandas as pd
import pickle
import gzip
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
from stim_functions import (paths, query_ephys_sessions, load_passive_opto_times, load_subjects,
                            figure_style, remap, high_level_regions, combine_regions,
                            init_one, get_dlc_XYs, get_raw_smooth_pupil_diameter)
one = init_one()

# Settings
OVERWITE = False

# Get paths
parent_fig_path, repo_path = paths()
data_path = join(repo_path, 'HMM', 'PassiveEvent', 'spont')
rec_files = glob(join(data_path, '*.pickle'))
subjects = load_subjects()

if OVERWITE:
    state_pupil_df = pd.DataFrame()
else:
    state_pupil_df = pd.read_csv(join(repo_path, 'state_pupil_corr.csv'))

for i, file_path in enumerate(rec_files):
    
    # Get info
    print(f'Starting {i} of {len(rec_files)}')
    subject = split(file_path)[1][:9]
    date = split(file_path)[1][10:20]
    if subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0] == 0:
        continue
    
    # Load in data    
    with gzip.open(file_path, 'rb') as handle:
        hmm_dict = pickle.load(handle)    
    prob_mat, time_ax = hmm_dict['prob_mat'], hmm_dict['time_ax']
    bin_size = np.mean(np.diff(time_ax))
    n_states = hmm_dict['prob_mat'].shape[2]
    event_times, event_ids = hmm_dict['event_times'], hmm_dict['event_ids']
    region = split(file_path)[-1].split('_')[-1].split('.')[0]
    prob_mat = prob_mat[event_ids == 0, :, :]
    
    # Load in pupil diameter
    print('Loading and smoothing pupil trace..')
    eid = one.search(subject=subject, date=date)[0]
    video_times, XYs = get_dlc_XYs(one, eid)
    if XYs is None:
        continue
    _, diameter = get_raw_smooth_pupil_diameter(XYs)
    
    # Get average pupil diameter during states of passive events
    state_pupil = np.empty((np.sum(event_ids == 0), time_ax.shape[0]))
    for i_e, this_event in enumerate(event_times[event_ids == 0]):
        this_diameter = np.empty(time_ax.shape[0])
        for i_tb, this_bin in enumerate(time_ax + this_event):
            this_diameter[i_tb] = np.nanmean(diameter[(video_times > this_bin - (bin_size/2)) & 
                                                      (video_times <= this_bin + (bin_size/2))])
        state_pupil[i_e, :] = this_diameter
            
    # Correlate pupil with state probability
    r_state, p_state = np.empty(n_states), np.empty(n_states)
    for this_state in range(n_states):
        
        this_prob = prob_mat[:, :, this_state].flatten()
        this_pupil = state_pupil.flatten()
        if this_pupil[~np.isnan(this_pupil)].shape[0] < 10:
            continue
        r_state[this_state], p_state[this_state] = stats.pearsonr(
            this_prob[~np.isnan(this_pupil)], this_pupil[~np.isnan(this_pupil)])
    
    # Add to dataframe
    subject = split(file_path)[-1].split('_')[0]
    date = split(file_path)[-1].split('_')[1]
    region = split(file_path)[-1].split('_')[2][:-7]
    file_name = split(file_path)[-1]
    state_pupil_df = pd.concat((state_pupil_df, pd.DataFrame(data={
        'state': np.arange(n_states), 'r': r_state, 'p': p_state,
        'subject': subject, 'date': date, 'region': region,
        'file_name': file_name})))
    state_pupil_df.to_csv(join(repo_path, 'state_pupil_corr.csv'), index=False)
        
        
            
        
    