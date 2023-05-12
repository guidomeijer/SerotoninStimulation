#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:00:47 2023
By: Guido Meijer
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
from stim_functions import figure_style, paths, load_subjects
from os.path import join

BIN_SIZE = 100  # ms
NEURONS = 'all'  # non-sig, sig or all
SERT_CRE = 1

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State')

# Load in data
state_trans_df = pd.read_csv(join(save_path, f'all_state_trans_all-neurons_{BIN_SIZE}msbins_{NEURONS}.csv'))
p_state_df = pd.read_csv(join(save_path, f'p_state_all-neurons_{BIN_SIZE}msbins_{NEURONS}.csv'))
state_trans_null_df = pd.read_csv(join(save_path, f'all_state_trans_null_all-neurons_{BIN_SIZE}msbins_{NEURONS}.csv'))
p_state_null_df = pd.read_csv(join(save_path, f'p_state_null_all-neurons_{BIN_SIZE}msbins_{NEURONS}.csv'))

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    state_trans_df.loc[state_trans_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    p_state_df.loc[p_state_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    state_trans_null_df.loc[state_trans_null_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    p_state_null_df.loc[p_state_null_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
state_trans_df = state_trans_df[state_trans_df['sert-cre'] == SERT_CRE]
p_state_df = p_state_df[p_state_df['sert-cre'] == SERT_CRE]
state_trans_null_df = state_trans_null_df[state_trans_null_df['sert-cre'] == SERT_CRE]
p_state_null_df = p_state_null_df[p_state_null_df['sert-cre'] == SERT_CRE]

# Correlate states and match them over recordings
time_ax = np.unique(p_state_df['time'])
df_copy = p_state_df.copy()
df_pivot = p_state_df.pivot(index=['eid', 'state'], columns='time', values='p_state')
   
# Loop over states
these_states = np.unique(p_state_df['state']).astype(int)
eids = np.unique(df_copy['eid'])
for main_state in these_states:
    these_main_states = np.empty(np.unique(df_copy['eid']).shape[0]).astype(int)
    
    for jj, this_eid in enumerate(np.unique(df_copy['eid'])):
        
        if jj == 0:                   
            # Start with state with the highest variance
            rem_states = np.unique(df_copy.loc[df_copy['eid'] == this_eid, 'state'])
            state_var = np.empty(rem_states.shape[0])
            for dd, state in enumerate(rem_states):
                state_var[dd] = np.std(df_copy.loc[(df_copy['eid'] == this_eid)
                                                        & (df_copy['state'] == state), 'p_state'].values)
            these_main_states[jj] = rem_states[np.argmax(state_var)]
            
        else:
            # Correlate state from previous session to each of these
            rem_states = np.unique(df_copy.loc[df_copy['eid'] == this_eid, 'state'])
            state_r = np.empty(rem_states.shape[0])
            for nn, state2 in enumerate(rem_states):
                state_r[nn], _ = pearsonr(df_pivot.loc[eids[jj-1], these_main_states[jj-1]].values,
                                          df_pivot.loc[this_eid, state2].values)
            these_main_states[jj] = rem_states[np.argmax(state_r)]
                            
    # Add main state to overall df and remove from slice copy for next iteration
    for tt, this_main_state in enumerate(these_main_states):
        
        # Set main state in dataframe
        p_state_df.loc[(p_state_df['eid'] == eids[tt])
                       & (p_state_df['state'] == this_main_state), 'main_state'] = main_state 
                                
        # Remove from df slice copy for next iteration
        df_copy = df_copy.drop(df_copy[(df_copy['eid'] == eids[tt])
                                                   & (df_copy['state'] == this_main_state)].index)
p_state_df['main_state'] = p_state_df['main_state'].astype(int)

# Order states by if they go up or down
main_states = np.unique(p_state_df['main_state'])
state_scores = np.empty(main_states.shape[0])
for state in main_states:
    p_state_slice = p_state_df[p_state_df['main_state'] == state]
    p_slice_group = p_state_slice[['time', 'p_state_bl']].groupby('time').mean().reset_index()
    state_trace = p_slice_group.loc[(p_slice_group['time'] > 0) & (p_slice_group['time'] < 2), 'p_state_bl'].values
    state_scores[state] = state_trace[np.argmax(np.abs(state_trace))]
sorted_states = np.argsort(-state_scores)
state_map = {sorted_states[ii]: main_states[ii] for ii in range(len(sorted_states))}
p_state_df['main_state'] = p_state_df['main_state'].replace(state_map)

# Correlate states and match them over recordings
df_copy = p_state_null_df.copy()
df_pivot = p_state_df.pivot(index=['eid', 'state'], columns='time', values='p_state')
   
# Loop over states
these_states = np.unique(p_state_df['state']).astype(int)
eids = np.unique(df_copy['eid'])
for main_state in these_states:
    these_main_states = np.empty(np.unique(df_copy['eid']).shape[0]).astype(int)
    
    for jj, this_eid in enumerate(np.unique(df_copy['eid'])):
        
        if jj == 0:                   
            # Start with state with the highest variance
            rem_states = np.unique(df_copy.loc[df_copy['eid'] == this_eid, 'state'])
            state_var = np.empty(rem_states.shape[0])
            for dd, state in enumerate(rem_states):
                state_var[dd] = np.std(df_copy.loc[(df_copy['eid'] == this_eid)
                                                        & (df_copy['state'] == state), 'p_state'].values)
            these_main_states[jj] = rem_states[np.argmax(state_var)]
            
        else:
            # Correlate state from previous session to each of these
            rem_states = np.unique(df_copy.loc[df_copy['eid'] == this_eid, 'state'])
            state_r = np.empty(rem_states.shape[0])
            for nn, state2 in enumerate(rem_states):
                state_r[nn], _ = pearsonr(df_pivot.loc[eids[jj-1], these_main_states[jj-1]].values,
                                          df_pivot.loc[this_eid, state2].values)
            these_main_states[jj] = rem_states[np.argmax(state_r)]
                            
    # Add main state to overall df and remove from slice copy for next iteration
    for tt, this_main_state in enumerate(these_main_states):
        
        # Set main state in dataframe
        p_state_null_df.loc[(p_state_null_df['eid'] == eids[tt])
                       & (p_state_null_df['state'] == this_main_state), 'main_state'] = main_state 
                                
        # Remove from df slice copy for next iteration
        df_copy = df_copy.drop(df_copy[(df_copy['eid'] == eids[tt])
                                       & (df_copy['state'] == this_main_state)].index)
p_state_null_df['main_state'] = p_state_null_df['main_state'].astype(int)

# Order states by if they go up or down
main_states = np.unique(p_state_null_df['main_state'])
state_scores = np.empty(main_states.shape[0])
for state in main_states:
    p_state_slice = p_state_null_df[p_state_df['main_state'] == state]
    p_slice_group = p_state_slice[['time', 'p_state_bl']].groupby('time').mean().reset_index()
    state_trace = p_slice_group.loc[(p_slice_group['time'] > 0) & (p_slice_group['time'] < 2), 'p_state_bl'].values
    state_scores[state] = state_trace[np.argmax(np.abs(state_trace))]
sorted_states = np.argsort(-state_scores)
state_map = {sorted_states[ii]: main_states[ii] for ii in range(len(sorted_states))}
p_state_null_df['main_state'] = p_state_null_df['main_state'].replace(state_map)

# %%
colors, dpi = figure_style()
f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi, sharey=True, sharex=True)
ax.add_patch(Rectangle((0, -0.05), 1, 0.1, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=state_trans_null_df, x='time', y='p_trans_bl',
             color=colors['grey'], errorbar='se', ax=ax, err_kws={'lw': 0})
sns.lineplot(data=state_trans_df, x='time', y='p_trans_bl',
             color=colors['stim'], errorbar='se', ax=ax, err_kws={'lw': 0})
ax.set(xlabel='Time from stimulation start (s)', ylim=[-0.03, 0.03], yticks=[-0.03, 0, 0.03],
       yticklabels=[-3, 0, 3], xticks=[-1, 0, 1, 2, 3, 4], ylabel='State transition probability (%)')

plt.tight_layout()
sns.despine(trim=True)

#plt.savefig(join(fig_path, 'state_change_rate_baseline.jpg'), dpi=600)

# %%
p_plot_df = p_state_df.copy()
p_plot_null_df = p_state_null_df.copy()
for i in np.unique(p_plot_df['main_state']):
    p_plot_df.loc[p_plot_df['main_state'] == i, 'p_state_bl'] -= i/6
    #p_plot_null_df.loc[p_plot_null_df['main_state'] == i, 'p_state_bl'] -= i/6

f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi, sharey=True, sharex=True)

n_states = np.unique(p_plot_df['main_state']).shape[0]
ax.add_patch(Rectangle((0, -0.16*n_states), 1, 2, color='royalblue', alpha=0.25, lw=0))
#sns.lineplot(data=p_plot_null_df[p_plot_null_df['region'] == region], x='time', y='p_state_bl',
#             hue='main_state', ax=axs[i], errorbar='se', legend=None, err_kws={'lw': 0},
#             palette=[colors['grey']]*n_states)
sns.lineplot(data=p_plot_df, x='time', y='p_state_bl',
             hue='main_state', ax=ax, errorbar='se', legend=None, err_kws={'lw': 0},
             palette=sns.diverging_palette(20, 210, l=55, center='dark', as_cmap=True))
  
#plt.savefig(join(fig_path, 'p_state_awake.jpg'), dpi=600)
    


