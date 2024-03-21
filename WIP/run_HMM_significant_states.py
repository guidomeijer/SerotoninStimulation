# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:26:54 2023

@author: Guido
"""


import numpy as np
from os.path import join, realpath, dirname, split
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
from stim_functions import (paths, query_ephys_sessions, load_passive_opto_times, load_subjects,
                            figure_style, remap, high_level_regions, combine_regions)

# Settings
PRE_TIME = [-1, 0]
POST_TIME = [0.2, 1.2]

# Get paths
parent_fig_path, repo_path = paths()
data_path = join(repo_path, 'HMM', 'PassiveEventAllNeurons')
rec_files = glob(join(data_path, '*.pickle'))

subjects = load_subjects()
state_sig_df = pd.DataFrame()
for i, file_path in enumerate(rec_files):
    
    # Get info
    subject = split(file_path)[1][:9]
    if subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0] == 0:
        continue
    
    # Load in data    
    with open(file_path, 'rb') as handle:
        hmm_dict = pickle.load(handle)    
    state_mat, time_ax = hmm_dict['state_mat'], hmm_dict['time_ax']
    n_states = hmm_dict['prob_mat'].shape[2]
    
    
    # Count states in baseline and stim periods
    p_values, state_sign = np.empty(n_states), np.empty(n_states)
    for s in range(n_states):
        bl_counts = np.sum(state_mat[:, (time_ax > PRE_TIME[0]) & (time_ax < PRE_TIME[1])] == s, axis=1)
        stim_counts = np.sum(state_mat[:, (time_ax > POST_TIME[0]) & (time_ax < POST_TIME[1])] == s, axis=1)
        _, p_values[s] = stats.ranksums(bl_counts, stim_counts)
        if np.sum(bl_counts) > np.sum(stim_counts):
            state_sign[s] = -1
        else:
            state_sign[s] = 1
            
    # Add to dataframe
    state_sig_df = pd.concat((state_sig_df, pd.DataFrame(data={
        'p': p_values, 'sign': state_sign, 'region': )))
   