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

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
state_trans_df = pd.read_csv(join(save_path, f'state_trans_all-neurons_{BIN_SIZE}msbins_{NEURONS}.csv'))
p_state_df = pd.read_csv(join(save_path, f'p_state_all-neurons_{BIN_SIZE}msbins_{NEURONS}.csv'))
state_trans_null_df = pd.read_csv(join(save_path, f'state_trans_null_all-neurons_{BIN_SIZE}msbins_{NEURONS}.csv'))
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
ax.set(xlabel='Time from stimulation start (s)', ylim=[-0.02, 0.032],
       yticks=[-0.02, -0.01, 0, 0.01, 0.02, 0.03],
       yticklabels=[-2, -1, 0, 1, 2, 3], xticks=[-1, 0, 1, 2, 3, 4],
       ylabel='State transition probability (%)')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'p_state_change_baseline_all-neurons.pdf'))

# %%
f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi, sharey=True, sharex=True)
ax.add_patch(Rectangle((0, -0.05), 1, 0.1, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=state_trans_null_df, x='time', y='p_trans',
             color=colors['grey'], errorbar='se', ax=ax, err_kws={'lw': 0})
sns.lineplot(data=state_trans_df, x='time', y='p_trans',
             color=colors['stim'], errorbar='se', ax=ax, err_kws={'lw': 0})
ax.set(xlabel='Time from stimulation start (s)', ylim=[-0.02, 0.032],
       xticks=[-1, 0, 1, 2, 3, 4],
       ylabel='State transition probability (%)')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'p_state_change_all-neurons.pdf'))

# %%
f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi, sharey=True, sharex=True)
ax.add_patch(Rectangle((0, -0.2), 1, 0.4, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=p_state_df, x='time', y='p_state_bl', hue='main_state',
             palette=colors['states'],
             errorbar='se', ax=ax, err_kws={'lw': 0}, legend=None)
ax.set(ylabel='State probability (%)', xlabel='Time from stimulation start (s)',
       yticks=[-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15], yticklabels=[-15, -10, -5, 0, 5, 10, 15],
       xticks=[-1, 0, 1, 2, 3, 4], ylim=[-0.16, 0.16])
plt.tight_layout()
sns.despine(trim=True)

# %%
p_plot_df = p_state_df.copy()
p_plot_null_df = p_state_null_df.copy()
for i in np.unique(p_plot_df['main_state']):
    p_plot_df.loc[p_plot_df['main_state'] == i, 'p_state_bl'] -= i/6
    p_plot_null_df.loc[p_plot_null_df['main_state'] == i, 'p_state_bl'] -= i/6

f, ax = plt.subplots(1, 1, figsize=(1.75, 3.5), dpi=dpi, sharey=True, sharex=True)

n_states = np.unique(p_plot_df['main_state']).shape[0]
ax.add_patch(Rectangle((0, -1), 1, 2, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=p_plot_null_df, x='time', y='p_state_bl',
             hue='main_state', ax=ax, errorbar='se', legend=None, err_kws={'lw': 0},
             palette=[colors['grey']]*n_states)
sns.lineplot(data=p_plot_df, x='time', y='p_state_bl',
             hue='main_state', ax=ax, errorbar='se', legend=None, err_kws={'lw': 0},
             palette=sns.diverging_palette(20, 210, l=55, center='dark', as_cmap=True))
ax.axis('off')
ax.set(ylim=[-1.03, 0.2])

ax.plot([-1.5, -1.5], [0, 0.1], color='k')
ax.text(-1.6, 0.05, '10%', ha='right', va='center')
ax.plot([0, 2], [-1.02, -1.02], color='k')
ax.text(1, -1.04, '2s', ha='center', va='top')
ax.text(-2.5, -0.5, 'State probability', rotation=90, ha='left', va='center')

ax.text(4.4, 0.11, 'State', ha='center', va='center')
ax.text(4.4, 0.01, '1', ha='center', va='center', fontsize=10,
        color=sns.diverging_palette(20, 210, l=55, center='dark')[0])
ax.text(4.4, -0.17, '2', ha='center', va='center', fontsize=10,
        color=sns.diverging_palette(20, 210, l=55, center='dark')[1])
ax.text(4.4, -0.33, '3', ha='center', va='center', fontsize=10,
        color=sns.diverging_palette(20, 210, l=55, center='dark')[2])
ax.text(4.4, -0.51, '4', ha='center', va='center', fontsize=10,
        color=sns.diverging_palette(20, 210, l=55, center='dark')[3])
ax.text(4.4, -0.67, '5', ha='center', va='center', fontsize=10,
        color=sns.diverging_palette(20, 210, l=55, center='dark')[4])
ax.text(4.4, -0.85, '6', ha='center', va='center', fontsize=10,
        color=sns.diverging_palette(20, 210, l=55, center='dark')[5])

plt.subplots_adjust(left=0.8)
plt.tight_layout()
plt.savefig(join(fig_path, 'p_state_all-neurons_awake_vertical.pdf'))

# %%
f, axs = plt.subplots(1, 6, figsize=(5.25, 1.75), dpi=dpi, sharey=True, sharex=True)
for i in np.unique(p_plot_df['main_state']):
    axs[i].add_patch(Rectangle((0, -0.12), 1, 0.24, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=p_state_null_df[p_state_null_df['main_state'] == i], x='time', y='p_state_bl',
                 color=colors['grey'], errorbar='se', ax=axs[i], err_kws={'lw': 0})
    sns.lineplot(data=p_state_df[p_state_df['main_state'] == i], x='time', y='p_state_bl',
                 color=colors['main_states'][i], errorbar='se', ax=axs[i], err_kws={'lw': 0})
    axs[i].set(ylim=[-0.12, 0.12], yticks=[-0.12, 0, 0.12], yticklabels=[-12, 0, 12])
    if i == 0:
        axs[i].set(ylabel='State transition probability (%)', xticks=[0, 2])
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].text(1, -0.125, '2s', ha='center', va='top')
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
            
plt.subplots_adjust(left=0.08, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
#plt.tight_layout(rect=(0.05, 0.05, 1, 1))
sns.despine(trim=True)
plt.savefig(join(fig_path, 'p_state_all-neurons_awake_horizontal.pdf'))


