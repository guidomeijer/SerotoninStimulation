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
colors, dpi = figure_style()

# Settings
PRE_TIME = [-1, 0]
POST_TIME = [0, 1]
MIN_REC = 3

# Get paths
f_path, repo_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])
data_path = join(repo_path, 'HMM', 'PassiveEvent')
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
    region = split(file_path)[-1].split('_')[-1].split('.')[0]
    
    # Count states in baseline and stim periods
    p_values, state_sign = np.empty(n_states), np.empty(n_states)
    for s in range(n_states):
        bl_counts = np.sum(state_mat[:, (time_ax > PRE_TIME[0]) & (time_ax < PRE_TIME[1])] == s, axis=1)
        stim_counts = np.sum(state_mat[:, (time_ax > POST_TIME[0]) & (time_ax < POST_TIME[1])] == s, axis=1)
        if np.sum(bl_counts) == np.sum(stim_counts):
            p_values[s] = 1
            state_sign[s] = 0
            continue
        _, p_values[s] = stats.ttest_rel(bl_counts, stim_counts)
        if np.sum(bl_counts) > np.sum(stim_counts):
            state_sign[s] = -1
        else:
            state_sign[s] = 1
            
    # Add to dataframe
    state_sig_df = pd.concat((state_sig_df, pd.DataFrame(data={
        'p': p_values, 'sign': state_sign, 'region': region, 'subject': subject})))

# Select significant states
state_sig_df['sig_state'] = (state_sig_df['p'] < 0.05).astype(int)

# Create over subjects summary
summary_df = state_sig_df[['sign', 'region', 'subject', 'sig_state']].groupby(
    ['sign', 'region', 'subject']).sum().reset_index()
enh_state_df = summary_df[summary_df['sign'] == 1]
supp_state_df = summary_df[summary_df['sign'] == -1]
supp_state_df['sig_state'] = -supp_state_df['sig_state']
enh_state_df = enh_state_df.groupby('region').filter(lambda x: len(x) >= MIN_REC)
supp_state_df = supp_state_df.groupby('region').filter(lambda x: len(x) >= MIN_REC)

# Order
ordered_regions = enh_state_df.groupby('region').mean(numeric_only=True).sort_values('sig_state', ascending=False).reset_index()


# %% Plot

f, ax1 = plt.subplots(1, 1, figsize=(2.2, 2), dpi=dpi)

#sns.swarmplot(data=enh_state_df, x='sig_states_count', y='region', ax=ax1, s=1)
sns.barplot(data=enh_state_df, x='sig_state', y='region', errorbar='se', ax=ax1,
            order=ordered_regions['region'], color=colors['enhanced'], label='Enhanced')
sns.barplot(data=supp_state_df, x='sig_state', y='region', errorbar='se', ax=ax1,
            order=ordered_regions['region'], color=colors['suppressed'], label='Suppressed')
ax1.set(xlim=[-3.5, 3.5], xticks=[-3, -2, -1, 0, 1, 2, 3], xticklabels=[3, 2, 1, 0, 1, 2, 3],
        xlabel='Number of significant states', ylabel='')
ax1.legend(bbox_to_anchor=(0.6, 0.4))

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'n_significant_states.pdf'))





    