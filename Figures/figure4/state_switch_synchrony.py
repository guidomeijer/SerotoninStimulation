# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:47:09 2023

@author: Guido
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import jaccard_score
from stim_functions import figure_style, paths, load_subjects
from os.path import join, realpath, dirname, split

BIN_SIZE = 100  # ms
NEURONS = 'all'  # non-sig, sig or all
SERT_CRE = 1
TIME_BIN_SIZE = 0.2
BIN_EDGES = np.arange(-1, 4.1, TIME_BIN_SIZE)
time_ax = BIN_EDGES[:-1]+(TIME_BIN_SIZE/2)

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
state_trans_df = pd.read_csv(join(save_path, f'all_state_trans_{BIN_SIZE}msbins_{NEURONS}.csv'))
state_trans_null_df = pd.read_csv(join(save_path, f'all_state_trans_null_{BIN_SIZE}msbins_{NEURONS}.csv'))

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    state_trans_df.loc[state_trans_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    state_trans_null_df.loc[state_trans_null_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
state_trans_df = state_trans_df[state_trans_df['sert-cre'] == SERT_CRE]
state_trans_null_df = state_trans_null_df[state_trans_null_df['sert-cre'] == SERT_CRE]

# Loop over sessions
sync_df = pd.DataFrame()
for i, eid in enumerate(np.unique(state_trans_df['eid'])):
    
    # Take slice of dataframe
    this_df = state_trans_df[state_trans_df['eid'] == eid]
    
    # Loop over pairs of brain regions
    brain_regions = np.unique(this_df['region'])
    for r1, region_1 in enumerate(brain_regions):
        for r2, region_2 in enumerate(brain_regions[r1+1:]):
            sync = np.empty(BIN_EDGES.shape[0]-1)
            for tb, time_bin in enumerate(BIN_EDGES[:-1]):
                this_time_df = this_df[(this_df['time'] > time_bin) & (this_df['time'] < BIN_EDGES[tb+1])]
                sync[tb] = jaccard_score(this_time_df.loc[this_time_df['region'] == region_1, 'state_change'].values,
                                         this_time_df.loc[this_time_df['region'] == region_2, 'state_change'].values)
            
            # Add to dataframe
            sync_df = pd.concat((sync_df, pd.DataFrame(data={
                'time': time_ax, 'sync': sync, 'sync_bl': sync - np.mean(sync[time_ax < 0]),
                'region_pair': f'{region_1}-{region_2}', 'region_1': region_1, 'region_2': region_2})))
            
# %% Plot
colors, dpi = figure_style()
f, ax = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(data=sync_df[sync_df['region_pair'] == 'Hippocampus-Striatum'], x='time', y='sync_bl',
             errorbar='se', hue='region_pair', ax=ax, palette='tab20')

# %%
f, ax = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(data=sync_df, x='time', y='sync', errorbar='se', err_kws={'lw': 0}, ax=ax)
ax.set(ylabel='State switch synchrony', xlabel='Time from stimulus onset (s)')
plt.tight_layout()
#plt.savefig(join(fig_path, 'state_switch_synchrony.pdf'))

# %%
f, ax = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(data=sync_df, x='time', y='sync', errorbar='se', hue='region_pair', 
             palette='tab20', ax=ax)
#plt.tight_layout()

            