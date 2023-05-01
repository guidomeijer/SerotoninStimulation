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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

PCA_DIM = 10
BIN_SIZE = 100  # ms
NEURONS = 'all'  # non-sig, sig or all
SERT_CRE = 1

# Initialize
pca = PCA(n_components=PCA_DIM, random_state=42)
tsne = TSNE(n_components=2, random_state=42)

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State')

# Load in data
state_trans_df = pd.read_csv(join(save_path, f'all_state_trans_{BIN_SIZE}msbins_{NEURONS}.csv'))
p_state_df = pd.read_csv(join(save_path, f'p_state_{BIN_SIZE}msbins_{NEURONS}.csv'))

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    state_trans_df.loc[state_trans_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    p_state_df.loc[p_state_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
state_trans_df = state_trans_df[state_trans_df['sert-cre'] == SERT_CRE]
p_state_df = p_state_df[p_state_df['sert-cre'] == SERT_CRE]

# Do PCA on states and cluster them
colors, dpi = figure_style()
time_ax = np.unique(p_state_df['time'])
for i, region in enumerate(np.unique(p_state_df['region'])):
    region_copy = p_state_df[p_state_df['region'] == region].copy()
    region_pivot = region_copy.pivot(index=['pid', 'state'], columns='time', values='p_state')
       
    # Loop over states
    these_states = np.unique(region_copy['state'])
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

# %% Plot states
f, axs = plt.subplots(2, 4, figsize=(5.25, 3.5), dpi=dpi, sharey=True, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(p_state_df['region'])):
    axs[i].add_patch(Rectangle((0, -4), 1, 5, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='p_state',
                 hue='main_state', ax=axs[i], errorbar='se', legend=None, palette='tab10',
                 err_kws={'lw': 0})
    axs[i].set(ylabel='P(state)', xlabel='Time (s)', title=region, ylim=[0, 0.5],
               xticks=[-1, 0, 1, 2, 3, 4])
    if i > 4:
        axs[i].set(xlabel='Time (s)')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'brain_states.jpg'), dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(5.25, 3.5), dpi=dpi, sharey=True, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(p_state_df['region'])):
    axs[i].add_patch(Rectangle((0, -4), 1, 5, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='p_state_bl',
                 hue='main_state', ax=axs[i], errorbar='se', legend=None, palette='tab10',
                 err_kws={'lw': 0})
    axs[i].set(ylabel='P(state)', xlabel='Time (s)', title=region, ylim=[-0.12, 0.2],
               xticks=[-1, 0, 1, 2, 3, 4])
    if i > 4:
        axs[i].set(xlabel='Time (s)')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'brain_states_baseline.jpg'), dpi=600)


# %% Plot P(state change)

f, axs = plt.subplots(2, 4, figsize=(5.25, 3.5), dpi=dpi, sharey=True, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(p_state_df['region'])):

    axs[i].add_patch(Rectangle((0, -4), 1, 5, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_trans',
                 color='k', errorbar='se', ax=axs[i], err_kws={'lw': 0})
    axs[i].set(title=region, ylim=[0.05, 0.3], xticks=[-1, 0, 1, 2, 3, 4],
               ylabel='', xlabel='')
axs[-1].axis('off')
f.text(0.5, 0.04, 'Time relative to stimulation onset (s)', ha='center')
f.text(0.04, 0.5, 'P(state change) over baseline', va='center', rotation='vertical')
plt.tight_layout(rect=(0.05, 0.05, 1, 1))
sns.despine(trim=True)
plt.savefig(join(fig_path, 'state_change_rate.jpg'), dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(5.25, 3.5), dpi=dpi, sharey=True, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(p_state_df['region'])):

    axs[i].add_patch(Rectangle((0, -4), 1, 5, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_trans_bl',
                 color='k', errorbar='se', ax=axs[i], err_kws={'lw': 0})
    axs[i].set(title=region, ylim=[-0.05, 0.055], yticks=[-0.05, 0, 0.05], xticks=[-1, 0, 1, 2, 3, 4],
               ylabel='', xlabel='')
axs[-1].axis('off')
f.text(0.5, 0.04, 'Time relative to stimulation onset (s)', ha='center')
f.text(0.04, 0.5, 'P(state change) over baseline', va='center', rotation='vertical')
plt.tight_layout(rect=(0.05, 0.05, 1, 1))
sns.despine(trim=True)
plt.savefig(join(fig_path, 'state_change_rate_baseline.jpg'), dpi=600)