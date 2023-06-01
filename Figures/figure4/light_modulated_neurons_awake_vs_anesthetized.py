#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:45:43 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style, combine_regions, load_subjects, high_level_regions

# Settings
MIN_NEURONS_POOLED = 5
MIN_NEURONS_PER_MOUSE = 5
MIN_MOD_NEURONS = 10
MIN_REC = 2

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in results
awake_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
anes_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_anesthesia.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    awake_neurons.loc[awake_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    anes_neurons.loc[anes_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Merge regions
awake_neurons['full_region'] = combine_regions(awake_neurons['region'], abbreviate=True)
anes_neurons['full_region'] = combine_regions(anes_neurons['region'], abbreviate=True)

# Drop root 
awake_neurons = awake_neurons.reset_index(drop=True)
awake_neurons = awake_neurons.drop(index=[i for i, j in enumerate(awake_neurons['full_region']) if 'root' in j])
anes_neurons = anes_neurons.reset_index(drop=True)
anes_neurons = anes_neurons.drop(index=[i for i, j in enumerate(anes_neurons['full_region']) if 'root' in j])

# Mean over regions
anes_df = anes_neurons.groupby('full_region').mean(numeric_only=True).reset_index()
awake_df = awake_neurons.groupby('full_region').mean(numeric_only=True).reset_index()

# Merge dataframes
grouped_df = anes_df.merge(awake_df, left_on='full_region', right_on='full_region')


# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)

# Add colormap
grouped_df['color'] = [colors[i] for i in grouped_df['full_region']]

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
#ax1.plot([-0.3, 0], [0, 0], color='grey', ls='--')
#ax1.plot([0, 0], [-0.06, 0.03], color='grey', ls='--')
ax1.plot([-0.3, 0.1], [-0.3, 0.1], color='grey', ls='--')

for i in grouped_df.index:
    ax1.text(grouped_df.loc[i, 'mod_index_late_x'] ,
             grouped_df.loc[i, 'mod_index_late_y'],
             grouped_df.loc[i, 'full_region'],
             ha='center', va='center',
             color=grouped_df.loc[i, 'color'], fontsize=4.5, fontweight='bold')
ax1.set(ylabel='Modulation index awake', xlabel='Modulation index anesthetized',
        xticks=np.arange(-0.3, 0.11, 0.1), yticks=np.arange(-0.3, 0.11, 0.1))
#r, p = pearsonr(grouped_df['mod_index_late'], grouped_df['latency'])
#ax1.text(0.1, 100, f'r = {r:.2f}', fontsize=6)
#ax1.text(-0.1, 520, '***', fontsize=10, ha='center')

sns.despine(offset=2, trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'modulation_awake_vs_anesthesia.pdf'))


