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
MIN_NEURONS = 5

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in results
mod_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
mod_over_time = pd.read_pickle(join(save_path, 'mod_over_time.pickle'))
all_neurons = pd.merge(mod_over_time, mod_neurons,
                           on=['pid', 'subject', 'date', 'neuron_id', 'region'])
all_neurons['full_region'] = combine_regions(all_neurons['region'], abbreviate=True)
all_neurons['full_region_name'] = combine_regions(all_neurons['region'], abbreviate=False)

# Get max modulation
all_neurons['max_mod_index'] = [i[np.argmax(np.abs(i))] for i in all_neurons['mod_idx']]

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname,
                    'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only modulated neurons in sert-cre mice
sert_neurons = all_neurons[(all_neurons['sert-cre'] == 1) & (all_neurons['modulated'] == 1)]
#sert_neurons['latency'] = sert_neurons['latency_peak_onset']
sert_neurons['latency'] = sert_neurons['latenzy']
print(np.sum(~np.isnan(sert_neurons['latency'])) / sert_neurons.shape[0])

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
stderr_df = sert_neurons.groupby('full_region_name').sem(
    numeric_only=True).reset_index().reset_index()
grouped_df['latency_sem'] = stderr_df['latency']
grouped_df['mod_index_sem'] = stderr_df['mod_index']
grouped_df['full_region_name'] = stderr_df['full_region_name']


# %%

PROPS = {'boxprops': {'facecolor': 'none', 'edgecolor': 'none'}, 'medianprops': {'color': 'red'},
         'whiskerprops': {'color': 'none'}, 'capprops': {'color': 'none'}}

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.6, 2), dpi=dpi)
sns.boxplot(x='latency', y='full_region', ax=ax1, data=sert_neurons,
            order=ordered_regions['full_region'],
            fliersize=0, zorder=2, **PROPS)
sns.stripplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
              color='k', size=1.5, ax=ax1)
ax1.set(xlabel='Modulation onset latency (s)', ylabel='',
        xticks=[0, 0.6, 1.2], xticklabels=[0, 0.6, 1.2], xlim=[-0.01, 1.2])
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'modulation_latency_per_region.pdf'))


# %%

use_neurons = sert_neurons[~np.isnan(sert_neurons['latency'])].copy()
r, p = pearsonr(use_neurons['mod_index'], use_neurons['latency'])

coeffs = np.polyfit(use_neurons['mod_index'], use_neurons['latency'], 2)
x_fit = np.linspace(use_neurons['mod_index'].min(), use_neurons['mod_index'].max(), 100)
y_fit = np.poly1d(coeffs)(x_fit)

f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.scatter(use_neurons['mod_index'], use_neurons['latency'], color='grey', s=3)
ax1.plot(x_fit, y_fit, color='tab:red')
ax1.set(xlabel='Modulation index', ylabel='Modulation latency (s)', xticks=[-1, 0, 1])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'modulation_latency_vs_index_latenzy.pdf'))

# %%

use_neurons['mod_index_abs'] = np.abs(use_neurons['mod_index'])
r, p = pearsonr(use_neurons['mod_index_abs'], use_neurons['latency'])

slope, intercept = np.polyfit(use_neurons['mod_index_abs'], use_neurons['latency'], 1)
x_fit = np.linspace(use_neurons['mod_index_abs'].min(), use_neurons['mod_index_abs'].max(), 100)
y_fit = slope * x_fit + intercept

f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.scatter(use_neurons['mod_index_abs'], use_neurons['latency'], color='grey', s=3)
ax1.plot(x_fit, y_fit, color='tab:red')
ax1.text(0.5, 1.1, '***', fontsize=12, ha='center')
ax1.set(xlabel='Absolute modulation index', ylabel='Modulation latency (s)',
        xticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xticklabels=[0, 0.2, 0.4, 0.6, 0.8, 1],
        yticks=[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], yticklabels=[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'modulation_latency_vs_absindex_latenzy.pdf'))

