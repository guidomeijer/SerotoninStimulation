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
from os.path import join, realpath, dirname, split

BIN_SIZE = 100  # ms
NEURONS = 'all'  # non-sig, sig or all
SERT_CRE = 1
REGION_ORDER = ['Frontal cortex', 'Hippocampus', 'Thalamus', 'Amygdala', 'Sensory cortex',
                'Midbrain', 'Striatum']

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
p_state_df = pd.read_csv(join(save_path, f'p_state_{BIN_SIZE}msbins_{NEURONS}_global-nstates.csv'))

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    p_state_df.loc[p_state_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
p_state_df = p_state_df[p_state_df['sert-cre'] == SERT_CRE]

# Correlate states and match them over recordings
colors, dpi = figure_style()
time_ax = np.unique(p_state_df['time'])
for i, region in enumerate(np.unique(p_state_df['region'])):
    region_copy = p_state_df[(p_state_df['region'] == region) & (p_state_df['opto'] == 1)].copy()
    region_pivot = region_copy.pivot(index=['pid', 'state'], columns='time', values='p_state')
       
    # Loop over states
    these_states = np.unique(region_copy['state']).astype(int)
    these_pids = np.unique(region_copy['pid'])
    
    for main_state in these_states:
        these_main_states = np.empty(these_pids.shape[0]).astype(int)
        
        for jj, this_pid in enumerate(these_pids):
            
            if jj == 0:                   
                # Start with state with the highest variance
                rem_states = np.unique(region_copy.loc[region_copy['pid'] == this_pid, 'state'])
                state_var = np.empty(rem_states.shape[0])
                for dd, state in enumerate(rem_states):
                    state_var[dd] = np.std(region_copy.loc[(region_copy['pid'] == this_pid)
                                                            & (region_copy['state'] == state), 'p_state'].values)
                these_main_states[jj] = rem_states[np.argmax(state_var)]
                
            else:
                # Correlate state from previous session to each of these
                rem_states = np.unique(region_copy.loc[region_copy['pid'] == this_pid, 'state'])
                state_r = np.empty(rem_states.shape[0])
                for nn, state2 in enumerate(rem_states):
                    state_r[nn], _ = pearsonr(region_pivot.loc[these_pids[jj-1], these_main_states[jj-1]].values,
                                              region_pivot.loc[this_pid, state2].values)
                these_main_states[jj] = rem_states[np.argmax(state_r)]
                                
        # Add main state to overall df and remove from slice copy for next iteration
        for tt, this_main_state in enumerate(these_main_states):
            
            # Set main state in dataframe
            p_state_df.loc[(p_state_df['pid'] == these_pids[tt])
                           & (p_state_df['region'] == region)
                           & (p_state_df['state'] == this_main_state), 'main_state'] = main_state 
                                    
            # Remove from df slice copy for next iteration
            region_copy = region_copy.drop(region_copy[(region_copy['pid'] == these_pids[tt])
                                                       & (region_copy['state'] == this_main_state)].index)
p_state_df['main_state'] = p_state_df['main_state'].astype(int)

# Order states by if they go up or down
for i, region in enumerate(np.unique(p_state_df['region'])):
    main_states = np.unique(p_state_df.loc[p_state_df['region'] == region, 'main_state'])
    state_scores = np.empty(main_states.shape[0])
    for state in main_states:
        p_state_slice = p_state_df[(p_state_df['main_state'] == state) & (p_state_df['region'] == region)]
        p_slice_group = p_state_slice[['time', 'p_state_bl']].groupby('time').mean().reset_index()
        state_trace = p_slice_group.loc[(p_slice_group['time'] > 0) & (p_slice_group['time'] < 2), 'p_state_bl'].values
        state_scores[state] = state_trace[np.argmax(np.abs(state_trace))]
    sorted_states = np.argsort(-state_scores)
    state_map = {sorted_states[ii]: main_states[ii] for ii in range(len(sorted_states))}
    p_state_df.loc[p_state_df['region'] == region, 'main_state'] = p_state_df.loc[
        p_state_df['region'] == region, 'main_state'].replace(state_map)


# %%
p_plot_df = p_state_df.copy()
for i in np.unique(p_plot_df['main_state']):
    p_plot_df.loc[p_plot_df['main_state'] == i, 'p_state_bl'] -= i/7

f, axs = plt.subplots(1, 7, figsize=(5.25, 3.5), dpi=dpi, sharey=True, sharex=True)

for i, region in enumerate(REGION_ORDER):
    n_states = np.unique(p_plot_df.loc[p_plot_df['region'] == region, 'main_state']).shape[0]
    axs[i].add_patch(Rectangle((0, -0.2*n_states), 1, 2, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=p_plot_df[p_plot_df['region'] == region], x='time', y='p_state_bl',
                 hue='main_state', ax=axs[i], errorbar='se', legend=None, err_kws={'lw': 0},
                 palette=sns.diverging_palette(20, 210, l=55, center='dark', as_cmap=True))
    axs[i].axis('off')
    axs[i].set_title(region)
    axs[i].set(ylim=[-1, 0.2])
axs[0].plot([-1.5, -1.5], [0, 0.1], color='k')
axs[0].text(-1.6, 0.05, '10%', ha='right', va='center')
axs[0].plot([0, 2], [-0.97, -0.97], color='k')
axs[0].text(1, -0.99, '2s', ha='center', va='top')
axs[0].text(-2.5, -0.5, 'State probability', rotation=90, ha='left', va='center')

plt.subplots_adjust(left=0.05, right=0.98)
plt.savefig(join(fig_path, 'p_state_awake.pdf'))
    


