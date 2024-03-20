# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:41:09 2024 by Guido Meijer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style, high_level_regions, load_subjects
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
psth_df = pd.read_pickle(join(save_path, 'psth.pickle'))
all_psth = np.column_stack(psth_df['peth'].to_numpy()).T
time_ax = psth_df['time'][0]
after_time = time_ax[time_ax > 0]

# Get high level regions
psth_df['full_region'] = high_level_regions(psth_df['acronym'])

# Get peak or through time
latency = np.empty(psth_df.shape[0])
for i in range(all_psth.shape[0]):
    if psth_df.loc[i, 'modulation'] > 0:
        latency[i] = after_time[np.argmax(all_psth[i, time_ax > 0])] * 1000
    elif psth_df.loc[i, 'modulation'] < 0:
        latency[i] = after_time[np.argmin(all_psth[i, time_ax > 0])] * 1000
psth_df['latency'] = latency
    
# Exclude root
psth_df = psth_df[psth_df['full_region'] != 'root']

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    psth_df.loc[psth_df['subject'] == nickname,
                'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
psth_df = psth_df[psth_df['sert-cre'] == 1]

# Order regions
ordered_regions = psth_df.groupby('full_region').median(numeric_only=True).sort_values(
    'latency', ascending=True).reset_index()



# %%

PROPS = {'boxprops': {'facecolor': 'none', 'edgecolor': 'none'}, 'medianprops': {'color': 'red'},
         'whiskerprops': {'color': 'none'}, 'capprops': {'color': 'none'}}

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2.7, 2), dpi=dpi)
# sns.pointplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
#              join=False, ci=68, color=colors['general'], ax=ax1)
# sns.boxplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
#            color=colors['general'], fliersize=0, linewidth=0.75, ax=ax1)
sns.violinplot(x='latency', y='full_region', data=psth_df, order=ordered_regions['full_region'],
               color=colors['grey'], linewidth=0, ax=ax1)
sns.boxplot(x='latency', y='full_region', ax=ax1, data=psth_df,
            order=ordered_regions['full_region'],
            fliersize=0, zorder=2, **PROPS)
sns.stripplot(x='latency', y='full_region', data=psth_df, order=ordered_regions['full_region'],
              color='k', size=1, ax=ax1)
ax1.set(xlabel='Modulation onset latency (ms)', ylabel='',
        xticks=np.arange(0, 1001, 200), xlim=[-150, 3000])
# plt.xticks(rotation=90)
# for i, region in enumerate(ordered_regions['full_region']):
#    this_lat = ordered_regions.loc[ordered_regions['full_region'] == region, 'latency'].values[0] * 1000
#    ax1.text(1200, i+0.25, f'{this_lat:.0f} ms', fontsize=5)
plt.tight_layout()
sns.despine(trim=True, offset=3)