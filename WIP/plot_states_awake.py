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
from stim_functions import figure_style, paths, load_subjects, N_CLUSTERS
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
    region_slice = p_state_df[p_state_df['region'] == region]
    region_pivot = region_slice.pivot(index=['pid', 'state'], columns='time', values='p_state')

    # Do PCA
    dim_red_pca = pca.fit_transform(region_pivot.values)

    # Do clustering
    state_clusters = KMeans(n_clusters=N_CLUSTERS[region], random_state=42, n_init='auto').fit_predict(dim_red_pca)

    f, axs = plt.subplots(2, 5, figsize=(7, 3.5), dpi=dpi)
    axs = np.concatenate(axs)
    for j in range(N_CLUSTERS[region]):
        state_mean = np.mean(region_pivot.values[state_clusters == j, :], axis=0)
        state_sem = (np.std(region_pivot.values[state_clusters == j, :], axis=0)
                     / np.sqrt(np.sum(state_clusters == j)))
        axs[j].plot(time_ax, state_mean, color='k', zorder=2)
        axs[j].fill_between(time_ax, state_mean - state_sem, state_mean + state_sem, alpha=0.25,
                            color='k', zorder=1, lw=0)
        axs[j].add_patch(Rectangle((0, axs[j].get_ylim()[0]), 1, axs[j].get_ylim()[1] - axs[j].get_ylim()[0],
                                   color='royalblue', alpha=0.25, lw=0))
        axs[j].set(xlabel='Time (s)', title=f'State {j+1} (n={np.sum(state_clusters == j)})',
                   xticks=[-1, 0, 1, 2, 3, 4])
    axs[0].set(ylabel='P(state)')
    f.suptitle(f'{region}')
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, f'states_{region}.jpg'), dpi=600)

# %% Plot P(state change)

f, axs = plt.subplots(2, 4, figsize=(5.25, 2.5), dpi=dpi, sharey=True, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(p_state_df['region'])):

    axs[i].add_patch(Rectangle((0, -4), 1, 5, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_trans',
                 color='k', errorbar='se', ax=axs[i], err_kws={'lw': 0})
    axs[i].set(title=region, ylim=[0.05, 0.3], yticks=[0.1], xticks=[-1, 0, 1, 2, 3, 4],
               ylabel='', xlabel='')
axs[-1].axis('off')
f.text(0.5, 0.04, 'Time relative to stimulation onset (s)', ha='center')
f.text(0.04, 0.5, 'P(state change) over baseline', va='center', rotation='vertical')
plt.tight_layout(rect=(0.05, 0.05, 1, 1))
sns.despine(trim=True)
plt.savefig(join(fig_path, 'state_change_rate.jpg'), dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(5.25, 2.5), dpi=dpi, sharey=True, sharex=True)
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
plt.savefig(join(fig_path, 'state_change_rate.jpg'), dpi=600)