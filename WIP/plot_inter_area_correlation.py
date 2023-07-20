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
BIN_SIZE = 100
MIN_SUBJECTS = 2
TIME_WIN = [0.2, 0.8]

# Get paths
fig_path, save_path = paths()

# Load in data
corr_df = pd.read_csv(join(save_path, f'region_corr_{BIN_SIZE}ms-bins.csv'))

# Remove nans
corr_df = corr_df[~np.isnan(corr_df['r'])]

# Select sert-cre animals
sert_df = corr_df[corr_df['sert-cre'] == 1]

# Plot all region pairs
colors, dpi = figure_style()
all_regions = np.unique(np.concatenate((np.unique(corr_df['region_1']), np.unique(corr_df['region_2']))))
summary_df = pd.DataFrame()
for r1, region_1 in enumerate(all_regions[:-1]):
    for r2, region_2 in enumerate(all_regions[r1+1:]):

        # Take a slice out of the dataframe
        slice_df = sert_df.loc[(((sert_df['region_1'] == region_1) & (sert_df['region_2'] == region_2))
                                | ((sert_df['region_1'] == region_2) & (sert_df['region_2'] == region_1)))]

        if len(np.unique(slice_df['subject'])) < MIN_SUBJECTS:
            continue

        # Do statistics
        p_values = np.empty(np.unique(slice_df['time']).shape[0])
        for i, time_bin in enumerate(np.unique(slice_df['time'])):
            _, p_values[i] = ttest_1samp(slice_df.loc[(slice_df['time'] == time_bin), 'r_baseline'], 0)

        # Add to summary df
        for i, subject in enumerate(np.unique(slice_df['subject'])):
            r_mean = np.mean(slice_df.loc[(slice_df['subject'] == subject)
                                          & (slice_df['time'] > TIME_WIN[0])
                                          & (slice_df['time'] < TIME_WIN[1]), 'r_baseline'])
            summary_df = pd.concat((summary_df, pd.DataFrame(index=[summary_df.shape[0]+1], data={
                'r': r_mean, 'subject': subject, 'region_pair': f'{region_1}-{region_2}',
                'region_1': region_1, 'region_2': region_2})))


        # Plot this region pair
        f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
        ax1.add_patch(Rectangle((0, -1), 1, 2, color='royalblue', alpha=0.25, lw=0))
        sns.lineplot(x='time', y='r_baseline', data=slice_df, legend=None, errorbar='se',
                     color='k')
        n_sert = np.unique(slice_df.loc[slice_df['sert-cre'] == 1, 'subject']).shape[0]
        ax1.set(xlabel='Time (s)', ylabel='Baseline subtracted \n pairwise correlation (r)',
                title=f'{region_1} - {region_2} (n={n_sert})', xticks=[-1, 0, 1, 2, 3],
                ylim=[-0.04, 0.04])
        ax1.scatter(np.unique(slice_df['time'])[p_values < 0.05], np.ones(np.sum(p_values < 0.05)) * 0.038,
                    color='k', marker='*', s=2)

        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(join(fig_path, 'Extra plots', 'Correlation',
                         f'{region_1} - {region_2} {BIN_SIZE}ms bins.jpg'), dpi=600)
        plt.close(f)

# %%
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(data=corr_df, x='time', y='r_baseline', ax=ax1, errorbar='se', hue='sert-cre')

