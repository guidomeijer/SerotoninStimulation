#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:50:59 2022
By: Guido Meijer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, realpath, dirname, split
from serotonin_functions import paths, figure_style, high_level_regions
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Settings
CLUSTER_N = np.arange(2, 19)  # number of clusters
PCA_DIM = 10

# Initialize
tsne = TSNE(n_components=2, random_state=42)
pca = PCA(n_components=PCA_DIM, random_state=42)

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
psth_df = pd.read_pickle(join(save_path, 'psth.pickle'))

# Get high level regions
psth_df['high_level_region'] = high_level_regions(psth_df['acronym'])
psth_df = psth_df[psth_df['high_level_region'] != 'root']

# Do dimensionality reduction on PSTHs
all_psth = np.column_stack(psth_df['peth'].to_numpy()).T
time_ax = psth_df['time'][0]
for i in range(all_psth.shape[0]):
    #all_psth[i, :] = all_psth[i, :] / np.max(all_psth[i, :])  # normalize
    #all_psth[i, :] = all_psth[i, :] - np.mean(all_psth[i, time_ax < 0])  # baseline subtract
    #all_psth[i, :] = all_psth[i, :] / np.mean(all_psth[i, time_ax < 0])  # divide over baseline
    all_psth[i, :] = all_psth[i, :] / (np.mean(all_psth[i, time_ax < 0]) + 1)  # divide over baseline + 0.1 spks/s (Steinmetz, 2019)

# PCA
dim_red_pca = pca.fit_transform(all_psth)

# Clustering
silhouette_avg = []
for i in CLUSTER_N:
    print(f'Cluster size {i} of {CLUSTER_N[-1]}')
    kmeans_fit = KMeans(n_clusters=i, random_state=42, n_init='auto').fit(dim_red_pca)
    psth_clusters = kmeans_fit.labels_

    # silhouette score
    silhouette_avg.append(silhouette_score(dim_red_pca, psth_clusters))


# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=dpi)
ax1.plot(CLUSTER_N, silhouette_avg, marker='o')
ax1.set(xlabel='Number of clusters', ylabel='Silhouette score', xticks=np.arange(2, 21, 4))
sns.despine(trim=True)
plt.tight_layout()
