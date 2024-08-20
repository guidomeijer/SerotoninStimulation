#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:24:48 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import ttest_1samp
from stim_functions import paths, figure_style
import networkx as nx

# Settings
MIN_SUBJECTS = 2
BIN_SIZE = 0.1
TIME_WIN = [-1, 3]

# Get paths
fig_path, save_path = paths()

# Load in data
mi_df = pd.read_csv(join(save_path, f'region_mutual_information_{int(BIN_SIZE*1000)}ms.csv'))

# Select sert-cre animals
sert_df = mi_df[mi_df['sert-cre'] == 1]

# Get time axis extend
time_min = np.min(sert_df['time']) - (BIN_SIZE/1000)/2
time_max = np.max(sert_df['time']) + (BIN_SIZE/1000)/2
xticks = [-1, 0, 1, 2, 3]

# Plot all region pairs
colors, dpi = figure_style()
all_regions = np.unique(np.concatenate((np.unique(mi_df['region_1']), np.unique(mi_df['region_2']))))
summary_df = pd.DataFrame()
for r1, region_1 in enumerate(all_regions[:-1]):
    for r2, region_2 in enumerate(all_regions[r1+1:]):

        # Take a slice out of the dataframe
        slice_df = sert_df.loc[(((sert_df['region_1'] == region_1) & (sert_df['region_2'] == region_2))
                                | ((sert_df['region_1'] == region_2) & (sert_df['region_2'] == region_1)))]

        if len(np.unique(slice_df['subject'])) < MIN_SUBJECTS:
            continue

        # Add to summary df
        for i, subject in enumerate(np.unique(slice_df['subject'])):
            mi_mean = np.mean(slice_df.loc[(slice_df['subject'] == subject)
                                          & (slice_df['time'] > TIME_WIN[0])
                                          & (slice_df['time'] < TIME_WIN[1]), 'mi_over_baseline'])
            summary_df = pd.concat((summary_df, pd.DataFrame(index=[summary_df.shape[0]+1], data={
                'mi': mi_mean, 'subject': subject, 'region_pair': f'{region_1}-{region_2}',
                'region_1': region_1, 'region_2': region_2})))


        # Plot this region pair
        f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
        ax1.add_patch(Rectangle((0, -1), 1, 2, color='royalblue', alpha=0.25, lw=0))
        ax1.plot([time_min, time_max], [0, 0], color='grey', ls='--')
        sns.lineplot(x='time', y='mi_over_baseline', data=slice_df, legend=None, errorbar='se',
                     color='k', err_kws={'lw': 0})
        n_sert = np.unique(slice_df.loc[slice_df['sert-cre'] == 1, 'subject']).shape[0]
        ax1.set(xlabel='Time (s)', ylabel='Baseline subtracted \n mutual information',
                title=f'{region_1} - {region_2} (n={n_sert})', xticks=xticks,
                xlim=[np.min(xticks), np.max(xticks)], ylim=[-0.4, 0.4])
      
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(join(fig_path, 'Extra plots', 'Mutual information', 
                         f'{region_1} - {region_2}.jpg'), dpi=600)
        plt.close(f)

# %%
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -1), 1, 2, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=mi_df[mi_df['sert-cre'] == 1], x='time', y='mi_over_baseline', ax=ax1,
             errorbar='se', err_kws={'lw': 0}, color='k')
ax1.set(xlabel='Time from stimulation start (s)', ylim=[-0.005, 0.01], title='All region pairs',
        yticks=[-0.2, 0.2], xticks=xticks, xlim=[np.min(xticks), np.max(xticks)])
ax1.set_ylabel('Baseline subtracted \n mutual information', labelpad=-10)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'Extra plots', 'Mutual information', 'Summary.jpg'), dpi=600)

