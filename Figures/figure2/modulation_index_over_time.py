# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:50:06 2022

@author: Guido
"""

import numpy as np
from os.path import join, realpath, dirname, split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
from stim_functions import paths, load_subjects, figure_style, combine_regions

# Settings
MIN_NEURONS = 20
MIN_REC = 3

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in results
mod_idx_df = pd.read_pickle(join(save_path, 'mod_over_time.pickle'))
mod_idx_df['full_region'] = combine_regions(mod_idx_df['region'], abbreviate=True)
#mod_idx_df['full_region'] = high_level_regions(mod_idx_df['region'])
mod_idx_df = mod_idx_df[mod_idx_df['full_region'] != 'root']
time_ax = mod_idx_df['time'].mean()

# Only include sert mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    mod_idx_df.loc[mod_idx_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
mod_idx_df = mod_idx_df[mod_idx_df['sert-cre'] == 1]

# Make into long form dataframe
mod_long_df = pd.DataFrame()
for i, region in enumerate(np.unique(mod_idx_df['full_region'])):
    if mod_idx_df[mod_idx_df['full_region'] == region].shape[0] < MIN_NEURONS:
        continue
    for ind in mod_idx_df[mod_idx_df['full_region'] == region].index.values:
        mod_long_df = pd.concat((mod_long_df, pd.DataFrame(data={
            'time': time_ax, 'mod_idx': mod_idx_df.loc[ind, 'mod_idx'],
            'region': region})), ignore_index=True)
mod_long_df = mod_long_df[mod_long_df['time'] < 3]

# Statistical test over regions: mean modulation index per neuron in a time window
WIN_START, WIN_END = 0.3, 0.8
time_mask = (np.asarray(time_ax) >= WIN_START) & (np.asarray(time_ax) <= WIN_END)
win_df = mod_idx_df.copy()
win_df['mod_win'] = [np.nanmean(np.asarray(m)[time_mask]) for m in win_df['mod_idx']]
win_df = win_df.groupby('full_region').filter(lambda x: len(x) >= MIN_NEURONS)
regions = np.unique(win_df['full_region'])
groups = [win_df.loc[win_df['full_region'] == r, 'mod_win'].dropna().values for r in regions]
h_stat, p_kw = stats.kruskal(*groups)
print(f'\nModulation index {WIN_START}-{WIN_END} s: Kruskal-Wallis across {len(regions)} regions, '
      f'H({len(regions) - 1}) = {h_stat:.2f}, p = {p_kw:.4g}')

# Post-hoc pairwise Mann-Whitney U with Holm correction
pairs, pvals = [], []
for i in range(len(regions)):
    for j in range(i + 1, len(regions)):
        pairs.append((regions[i], regions[j]))
        pvals.append(stats.mannwhitneyu(groups[i], groups[j])[1])
order = np.argsort(pvals)
p_holm = np.empty(len(pvals))
running_max = 0
for rank, idx in enumerate(order):
    running_max = max(running_max, (len(pvals) - rank) * pvals[idx])
    p_holm[idx] = min(running_max, 1)
posthoc_win = pd.DataFrame({'region_1': [p[0] for p in pairs], 'region_2': [p[1] for p in pairs],
                            'p_uncorrected': pvals, 'p_holm': p_holm})
print(posthoc_win[posthoc_win['p_holm'] < 0.05].to_string(index=False))


# %% Plot
colors, dpi = figure_style()

mod_mean_df = mod_long_df.groupby(['region', 'time']).mean(numeric_only=True).reset_index()
mod_matrix = pd.pivot(mod_mean_df, columns='time', index='region', values='mod_idx')

mod_matrix['order'] = np.max(mod_matrix.iloc[:, 11:20], axis=1)
mod_matrix = mod_matrix.sort_values(by='order', axis=0)
mod_matrix.drop('order', axis=1, inplace=True)

f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
im_mod = ax1.imshow(np.flipud(mod_matrix), aspect='auto', extent=[-1, 3, 0, mod_matrix.shape[0]],
                    cmap='coolwarm', clim=[-0.1, 0.1], interpolation='none')
ax1.set(yticks=np.arange(mod_matrix.shape[0])+0.5, yticklabels=mod_matrix.index,
        xticks=[-1, 0, 1, 2, 3], xlabel='Time from 5-HT onset (s)')
cbar = plt.colorbar(im_mod)
cbar.ax.set_ylabel('Modulation index', rotation=270)
cbar.ax.set_yticks([-0.1, 0, 0.1])
cbar.ax.set_yticklabels([-0.1, 0, 0.1])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'modulation_index_over_time_regions.pdf'))

