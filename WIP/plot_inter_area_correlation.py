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

# Get time axis extend
time_min = np.min(sert_df['time']) - (BIN_SIZE/1000)/2
time_max = np.max(sert_df['time']) + (BIN_SIZE/1000)/2
xticks = [time_min, 0, 1, time_max]

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
        ax1.plot([time_min, time_max], [0, 0], color='grey', ls='--')
        sns.lineplot(x='time', y='r', data=slice_df, legend=None, errorbar='se',
                     color='k', err_kws={'lw': 0})
        n_sert = np.unique(slice_df.loc[slice_df['sert-cre'] == 1, 'subject']).shape[0]
        ax1.set(xlabel='Time (s)', ylabel='Baseline subtracted \n pairwise correlation (r)',
                title=f'{region_1} - {region_2} (n={n_sert})', xticks=xticks,
                ylim=[-0.02, 0.05])
      
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(join(fig_path, 'Extra plots', 'Correlation', f'{BIN_SIZE}ms',
                         f'{region_1} - {region_2} {BIN_SIZE}ms bins.jpg'), dpi=600)
        plt.close(f)

# %%
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -1), 1, 2, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=corr_df, x='time', y='r', ax=ax1, errorbar='se', hue='sert-cre',
             err_kws={'lw': 0}, hue_order=[0, 1], palette=[colors['wt'], colors['sert']])
ax1.set(ylabel='Correlation (r)', xlabel='Time from stimulation start (s)', ylim=[0, 0.02],
        yticks=[0, 0.02], xticks=xticks)
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.25), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['WT', 'SERT']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'Extra plots', 'Correlation', f'{BIN_SIZE}ms', 'Summary.jpg'), dpi=600)

