#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from matplotlib.colors import ListedColormap
from stim_functions import paths, figure_style, load_subjects, combine_regions

# Settings
MIN_NEURONS = 15

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_neurons['full_region'] = combine_regions(all_neurons['region'], abbreviate=True)

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname,
                    'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only modulated neurons in sert-cre mice
sert_neurons = all_neurons[(all_neurons['sert-cre'] == 1) & (all_neurons['modulated'] == 1)]
sert_neurons['latency'] = sert_neurons['latency_peak_onset']

# Get percentage modulated per region
reg_neurons = sert_neurons.groupby('full_region').median(numeric_only=True)['latency'].to_frame()
reg_neurons['n_neurons'] = sert_neurons.groupby(['full_region']).size()
reg_neurons['perc_mod'] = (sert_neurons.groupby(['full_region']).sum(numeric_only=True)['modulated']
                           / sert_neurons.groupby(['full_region']).size()) * 100
reg_neurons = reg_neurons.loc[reg_neurons['n_neurons'] >= MIN_NEURONS]
reg_neurons = reg_neurons.reset_index()
reg_neurons = reg_neurons[reg_neurons['full_region'] != 'root']
reg_neurons = reg_neurons[reg_neurons['n_neurons'] >= MIN_NEURONS]
reg_neurons = reg_neurons.sort_values('latency')

# Apply selection criteria
sert_neurons = sert_neurons[sert_neurons['full_region'].isin(reg_neurons['full_region'])]
sert_neurons.loc[sert_neurons['latency'] == 0, 'latency'] = np.nan

# Order regions
ordered_regions = sert_neurons.groupby('full_region').median(numeric_only=True).sort_values(
    'latency', ascending=True).reset_index()

# Convert to log scale
sert_neurons['log_latency'] = np.log10(sert_neurons['latency'])

# Get absolute
sert_neurons['mod_index_abs'] = sert_neurons['mod_index'].abs()

# Group by region
grouped_df = sert_neurons.groupby('full_region').median(
    numeric_only=True).reset_index().reset_index()

# Convert to ms
grouped_df['latency'] = grouped_df['latency'] * 1000
sert_neurons['latency'] = sert_neurons['latency'] * 1000


# %%

PROPS = {'boxprops': {'facecolor': 'none', 'edgecolor': 'none'}, 'medianprops': {'color': 'red'},
         'whiskerprops': {'color': 'none'}, 'capprops': {'color': 'none'}}

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.6, 2), dpi=dpi)
# sns.pointplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
#              join=False, ci=68, color=colors['general'], ax=ax1)
# sns.boxplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
#            color=colors['general'], fliersize=0, linewidth=0.75, ax=ax1)
#sns.violinplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
#               color=colors['grey'], linewidth=0, ax=ax1)
sns.boxplot(x='latency', y='full_region', ax=ax1, data=sert_neurons,
            order=ordered_regions['full_region'],
            fliersize=0, zorder=2, **PROPS)
sns.stripplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
              color='k', size=1, ax=ax1)
ax1.set(xlabel='Modulation onset latency (ms)', ylabel='',
        xticks=np.arange(0, 801, 200), xlim=[-10, 800])
# plt.xticks(rotation=90)
# for i, region in enumerate(ordered_regions['full_region']):
#    this_lat = ordered_regions.loc[ordered_regions['full_region'] == region, 'latency'].values[0] * 1000
#    ax1.text(1200, i+0.25, f'{this_lat:.0f} ms', fontsize=5)
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'modulation_latency_per_region.pdf'))


# %%

# Add colormap
grouped_df['color'] = [colors[i] for i in grouped_df['full_region']]

f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
# this only plots the line
(
    so.Plot(grouped_df, x='mod_index', y='latency')
    .add(so.Dot(pointsize=0))
    .add(so.Line(color='grey', linewidth=1), so.PolyFit(order=1))
    .on(ax1)
    .plot()
)
# this plots the colored region names
for i in grouped_df.index:
    ax1.text(grouped_df.loc[i, 'mod_index'],
             grouped_df.loc[i, 'latency'],
             grouped_df.loc[i, 'full_region'],
             ha='center', va='center',
             color=grouped_df.loc[i, 'color'], fontsize=6, fontweight='bold')
ax1.set(yticks=[0, 100, 200], xticks=[-0.25, 0, 0.25], xticklabels=[-0.25, 0, 0.25],
        ylabel='Modulation latency (ms)', xlabel='Modulation index')
r, p = pearsonr(grouped_df['mod_index'], grouped_df['latency'])
# ax1.text(0.1, 100, f'r = {r:.2f}', fontsize=6)
ax1.text(0, 200, '**', fontsize=10, ha='center')

sns.despine(offset=2, trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'modulation_latency_vs_index.pdf'))
