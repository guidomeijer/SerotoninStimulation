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
from scipy.stats import pearsonr
from matplotlib.colors import ListedColormap
from stim_functions import paths, figure_style, load_subjects

# Settings
N_BINS = 30
MIN_NEURONS = 0
AP = [2, -1.5, -3.5]

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in results
all_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
expression_df = pd.read_csv(join(save_path, 'expression_levels.csv'))

# Add genotype and subject number
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    all_neurons.loc[all_neurons['subject'] == nickname, 'subject_nr'] = subjects.loc[subjects['subject'] == nickname, 'subject_nr'].values[0]

# Only sert-cre mice
sert_neurons = all_neurons[all_neurons['sert-cre'] == 1]
wt_neurons = all_neurons[all_neurons['sert-cre'] == 0]

# Calculate percentage modulated neurons
all_mice = ((sert_neurons.groupby(['subject', 'subject_nr']).sum()['opto_modulated']
             / sert_neurons.groupby(['subject', 'subject_nr']).size() * 100).to_frame().reset_index())
all_mice['sert-cre'] = 1
wt_mice = ((wt_neurons.groupby(['subject', 'subject_nr']).sum()['opto_modulated']
            / wt_neurons.groupby(['subject', 'subject_nr']).size() * 100).to_frame().reset_index())
wt_mice['sert-cre'] = 0
all_mice = pd.concat((all_mice, wt_mice), ignore_index=True)
all_mice = all_mice.rename({0: 'perc_mod'}, axis=1)
all_mice['subject_nr'] = all_mice['subject_nr'].astype(int)

# Merge dataframes
merged_df = pd.merge(all_mice, expression_df, on=['subject', 'subject_nr', 'sert-cre'])
merged_df = merged_df[merged_df['sert-cre'] == 1]
merged_df['subject_nr'] = merged_df['subject_nr'].astype(int)

# %% Plot percentage mod neurons
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1), dpi=dpi)

# For some reason I don't understand I have to plot it this way to get the subect color right
this_cmap = [colors['subject_palette'][i] for i in
             all_mice.loc[all_mice['sert-cre'] == 1, 'subject_nr']]
sns.swarmplot(x=[1]*np.sum(all_mice['sert-cre'] == 1),
              y=all_mice.loc[all_mice['sert-cre'] == 1, 'perc_mod'],
              hue=all_mice.loc[all_mice['sert-cre'] == 1, 'subject_nr'],
              palette=this_cmap, legend=None, size=2.5, ax=ax1)
this_cmap = [colors['subject_palette'][i] for i in
             all_mice.loc[all_mice['sert-cre'] == 0, 'subject_nr']]
sns.swarmplot(x=[2]*np.sum(all_mice['sert-cre'] == 0),
              y=all_mice.loc[all_mice['sert-cre'] == 0, 'perc_mod'],
              hue=all_mice.loc[all_mice['sert-cre'] == 0, 'subject_nr'],
              palette=this_cmap, legend=None, size=2.5, ax=ax1)
#sns.swarmplot(x='sert-cre', y='perc_mod', data=all_mice, order=[1, 0], size=2.5, hue='subject_nr',
#              palette=this_cmap, legend=None, ax=ax1)
f.subplots_adjust(bottom=0.2, left=0.35, right=0.85, top=0.9)
ax1.set(xticklabels=['SERT', 'WT'], ylabel='Mod. neurons (%)', ylim=[-1, 32], xlabel='',
        yticks=[0, 30])

sns.despine(trim=True)
#plt.tight_layout()

plt.savefig(join(fig_path, 'light_mod_summary.pdf'))

# %% Plot percentage mod neurons vs expression
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1.1), dpi=dpi)

this_cmap = ListedColormap([colors['subject_palette'][i] for i in merged_df['subject_nr']])

sns.regplot(data=merged_df, x='perc_mod', y='rel_fluo', ax=ax1, ci=None,
            scatter_kws={'c': range(merged_df.shape[0]), 'color': None,
                         'cmap': this_cmap, 'alpha': 1, 's': 3},
            line_kws={'color': 'k', 'lw': 1})

ax1.set(xlim=[0, 32], xticks=[0, 30],
        yticks=[0, 175, 350])
ax1.tick_params(axis='x', which='major', pad=2)
ax1.set_ylabel('Rel. expression (%)', rotation=90, labelpad=2)
ax1.set_xlabel('Mod. neurons (%)', rotation=0, labelpad=2)
r, p = pearsonr(merged_df['rel_fluo'], merged_df['perc_mod'])
print(f'correlation p-value: {p:.3f}')
ax1.text(15, 300, '***', fontsize=10, ha='center')


f.subplots_adjust(bottom=0.3, left=0.32, right=0.88, top=0.9)
sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, 'light_mod_vs_expression.pdf'))





