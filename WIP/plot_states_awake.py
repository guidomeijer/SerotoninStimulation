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
from serotonin_functions import figure_style, paths, load_subjects
from os.path import join
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

PCA_DIM = 10
BIN_SIZE = 100  # ms
NEURONS = 'non-sig'  # non-sig, sig or all

# Initialize
pca = PCA(n_components=PCA_DIM, random_state=42)
tsne = TSNE(n_components=2, random_state=42)

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State')

# Load in data
state_trans_df = pd.read_csv(join(save_path, f'all_state_trans_{BIN_SIZE}msbins_{NEURONS}.csv'))
p_state_df = pd.read_csv(join(save_path, f'p_state_{BIN_SIZE}msbins_{NEURONS}.csv'))

"""
# Correlate each state with each other state
for i, region in enumerate(np.unique(p_state_df['region'])):
    region_slice = p_state_df[p_state_df['region'] == region]
    for state in np.unique(region_slice['state']):
        for pid in np.unique(region_slice['pid']):

            region_slice[].pivot(index='pid', columns='time', values='p_state')
"""

# Do PCA on states and cluster them
for i, region in enumerate(np.unique(p_state_df['region'])):
    region_slice = p_state_df[p_state_df['region'] == region]
    region_pivot = region_slice.pivot(index=['pid', 'state'], columns='time', values='p_state')

    # Do PCA
    dim_red_pca = pca.fit_transform(region_pivot.values)

    # Do clustering
    n_states = np.unique(region_slice['state']).shape[0]
    state_clusters = KMeans(n_clusters=n_states, random_state=42, n_init='auto').fit_predict(dim_red_pca)
    region_pivot['state_cluster'] = state_clusters

    for j in range(n_states):
        plt.plot(np.mean(region_pivot.values[state_clusters == j, :], axis=0))


# Average over mice first
state_trans_df = state_trans_df.groupby(['subject', 'time', 'region']).mean(numeric_only=True).reset_index()
p_state_df = p_state_df.groupby(['subject', 'time', 'region']).mean(numeric_only=True).reset_index()

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    state_trans_df.loc[state_trans_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    p_state_df.loc[p_state_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
state_trans_df = state_trans_df[state_trans_df['sert-cre'] == 1]
p_state_df = p_state_df[p_state_df['sert-cre'] == 1]

# %% Plot
colors, dpi = figure_style()

f, axs = plt.subplots(2, 4, figsize=(7, 3.5), dpi=dpi)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(p_state_df['region'])):

    axs[i].add_patch(Rectangle((0, -4), 1, 5, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_trans_bl',
                 color='k', estimator=None, units='subject', ax=axs[i])
    axs[i].set(ylabel='State change rate', xlabel='Time (s)', title=region, ylim=[-0.1, 0.1],
               yticks=[-0.1, 0, 0.1], xticks=[-1, 0, 1, 2, 3, 4])
axs[-1].axis('off')
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'state_change_rate.jpg'), dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(7, 3.5), dpi=dpi)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(p_state_df['region'])):

    axs[i].add_patch(Rectangle((0, -0.3), 1, 0.6, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='state_incr',
                 color=colors['enhanced'], errorbar='se', ax=axs[i])
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='state_decr',
                 color=colors['suppressed'], errorbar='se', ax=axs[i])
    axs[i].set(ylabel='P(state)', xlabel='Time (s)', title=region, ylim=[-0.2, 0.2],
               yticks=[-0.2, 0, 0.2], xticks=[-1, 0, 1, 2, 3, 4])
axs[-1].axis('off')
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'opto_state.jpg'), dpi=600)