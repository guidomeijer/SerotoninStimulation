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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PCA_DIM = 10
BIN_SIZE = 100  # ms
NEURONS = 'all'  # non-sig, sig or all
CLUSTER_N = np.arange(2, 19)  # number of clusters

# Initialize
pca = PCA(n_components=PCA_DIM, random_state=42)

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State')

# Load in data
p_state_df = pd.read_csv(join(save_path, f'p_state_{BIN_SIZE}msbins_{NEURONS}.csv'))

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    p_state_df.loc[p_state_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
p_state_df = p_state_df[p_state_df['sert-cre'] == 1]

# Do PCA on states and cluster them
colors, dpi = figure_style()
time_ax = np.unique(p_state_df['time'])
f, axs = plt.subplots(2, 4, figsize=(5.25, 3), dpi=dpi)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(p_state_df['region'])):
    region_slice = p_state_df[p_state_df['region'] == region]
    region_pivot = region_slice.pivot(index=['pid', 'state'], columns='time', values='p_state')

    # Do PCA
    dim_red_pca = pca.fit_transform(region_pivot.values)

    # Do clustering
    silhouette_avg = []
    for j in CLUSTER_N:
        kmeans_fit = KMeans(n_clusters=j, random_state=42, n_init='auto').fit(dim_red_pca)
        state_clusters = kmeans_fit.labels_
        silhouette_avg.append(silhouette_score(dim_red_pca, state_clusters))

    # Plot
    axs[i].plot(CLUSTER_N, silhouette_avg, marker='o')
    axs[i].set(xlabel='Number of clusters', ylabel='Silhouette score', xticks=np.arange(2, 21, 4),
               title=f'{region}')
axs[-1].axis('off')
sns.despine(trim=True)
plt.tight_layout()
