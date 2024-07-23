#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import json
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from scipy.stats import pearsonr, kruskal
from matplotlib.colors import ListedColormap
from stim_functions import paths, figure_style, load_subjects, combine_regions

# Settings
MIN_NEURONS = 15

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])
colors, dpi = figure_style()

# Load in results
task_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
awake_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Add genotype and subject number
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    task_neurons.loc[task_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    task_neurons.loc[task_neurons['subject'] == nickname, 'subject_nr'] = subjects.loc[subjects['subject'] == nickname, 'subject_nr'].values[0]
    awake_neurons.loc[awake_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    awake_neurons.loc[awake_neurons['subject'] == nickname, 'subject_nr'] = subjects.loc[subjects['subject'] == nickname, 'subject_nr'].values[0]

# Only sert-cre mice
task_neurons = task_neurons[task_neurons['sert-cre'] == 1]
awake_neurons = awake_neurons[awake_neurons['sert-cre'] == 1]

# Only modulated neurons
task_neurons = task_neurons[task_neurons['opto_modulated'] == 1]
awake_neurons = awake_neurons[awake_neurons['modulated'] == 1]

# Drop root and void
task_neurons = task_neurons[(task_neurons['region'] != 'root') & (task_neurons['region'] != 'void')]
awake_neurons = awake_neurons[(awake_neurons['region'] != 'root') & (awake_neurons['region'] != 'void')]

# Only high level regions
task_neurons['full_region'] = combine_regions(task_neurons['region'])
awake_neurons['full_region'] = combine_regions(awake_neurons['region'])
task_neurons = task_neurons[task_neurons['full_region'] != 'root']
awake_neurons = awake_neurons[awake_neurons['full_region'] != 'root']

# Get mean per region
grouped_df = awake_neurons.groupby(['full_region']).median(numeric_only=True)['mod_index'].to_frame()
grouped_df['n_neurons'] = awake_neurons.groupby(['full_region']).size()
grouped_df['opto_mod_roc'] = task_neurons.groupby(['full_region']).median(numeric_only=True)['opto_mod_roc']
grouped_df = grouped_df.rename(columns={'opto_mod_roc': 'task', 'mod_index': 'passive'}).reset_index()
grouped_df = grouped_df.loc[grouped_df['n_neurons'] >= MIN_NEURONS]

# %%
colors, dpi = figure_style()

# Add colormap
grouped_df['color'] = [colors[i] for i in grouped_df['full_region']]

f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
ax1.plot([-0.3, 0.3], [-0.3, 0.3], ls='--', color='grey')
# this plots the colored region names
for i in grouped_df.index:
    ax1.text(grouped_df.loc[i, 'passive'],
             grouped_df.loc[i, 'task'],
             grouped_df.loc[i, 'full_region'],
             ha='center', va='center',
             color=grouped_df.loc[i, 'color'], fontsize=6, fontweight='bold')
ax1.set(yticks=[-0.3, 0, 0.3], xticks=[-0.3, 0, 0.3],
        xticklabels=[-0.3, 0, 0.3], yticklabels=[-0.3, 0, 0.3],
        ylabel='Modulation index task', xlabel='Modulation index passive')

sns.despine(offset=2, trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'modulation_passive_vs_task.pdf'))

