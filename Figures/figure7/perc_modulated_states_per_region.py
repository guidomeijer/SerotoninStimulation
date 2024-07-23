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
import scikit_posthocs as sp
from matplotlib.colors import ListedColormap
from stim_functions import paths, figure_style, load_subjects, combine_regions

# Settings
N_BINS = 30
MIN_NEURONS = 0

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])
colors, dpi = figure_style()

# Load in results
task_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
awake_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
anes_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_anesthesia.csv'))

# Add genotype and subject number
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    task_neurons.loc[task_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    task_neurons.loc[task_neurons['subject'] == nickname, 'subject_nr'] = subjects.loc[subjects['subject'] == nickname, 'subject_nr'].values[0]
    awake_neurons.loc[awake_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    awake_neurons.loc[awake_neurons['subject'] == nickname, 'subject_nr'] = subjects.loc[subjects['subject'] == nickname, 'subject_nr'].values[0]
    anes_neurons.loc[anes_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    anes_neurons.loc[anes_neurons['subject'] == nickname, 'subject_nr'] = subjects.loc[subjects['subject'] == nickname, 'subject_nr'].values[0]

# Only sert-cre mice
task_neurons = task_neurons[task_neurons['sert-cre'] == 1]
awake_neurons = awake_neurons[awake_neurons['sert-cre'] == 1]
anes_neurons = anes_neurons[anes_neurons['sert-cre'] == 1]

# Drop root and void
task_neurons = task_neurons[(task_neurons['region'] != 'root') & (task_neurons['region'] != 'void')]
awake_neurons = awake_neurons[(awake_neurons['region'] != 'root') & (awake_neurons['region'] != 'void')]
anes_neurons = anes_neurons[(anes_neurons['region'] != 'root') & (anes_neurons['region'] != 'void')]

# Only high level regions
task_neurons['combined_region'] = combine_regions(task_neurons['region'])
awake_neurons['combined_region'] = combine_regions(awake_neurons['region'])
anes_neurons['combined_region'] = combine_regions(anes_neurons['region'])
task_neurons = task_neurons[task_neurons['combined_region'] != 'root']
awake_neurons = awake_neurons[awake_neurons['combined_region'] != 'root']
anes_neurons = anes_neurons[anes_neurons['combined_region'] != 'root']

# Calculate percentage modulated neurons
task_mice = ((task_neurons.groupby(['subject', 'combined_region']).sum()['opto_modulated']
              / task_neurons.groupby(['subject', 'combined_region']).size() * 100).to_frame())
task_mice = task_mice.rename(columns={0: 'perc_mod'})
awake_mice = ((awake_neurons.groupby(['subject', 'combined_region']).sum()['modulated']
              / awake_neurons.groupby(['subject', 'combined_region']).size() * 100).to_frame())
awake_mice = awake_mice.rename(columns={0: 'perc_mod'})
anes_mice = ((anes_neurons.groupby(['subject', 'combined_region']).sum()['modulated']
              / anes_neurons.groupby(['subject', 'combined_region']).size() * 100).to_frame())
anes_mice = anes_mice.rename(columns={0: 'perc_mod'})

# Add modulation index
task_mice['mod_index'] = task_neurons[task_neurons['opto_modulated']].groupby(
    ['subject', 'combined_region']).mean(numeric_only=True)['opto_mod_roc']
awake_mice['mod_index'] = awake_neurons[awake_neurons['modulated']].groupby([
    'subject', 'combined_region']).mean(numeric_only=True)['mod_index']
anes_mice['mod_index'] = anes_neurons[anes_neurons['modulated']].groupby(
    ['subject', 'combined_region']).mean(numeric_only=True)['mod_index']

# Reset index
task_mice = task_mice.reset_index()
awake_mice = awake_mice.reset_index()
anes_mice = anes_mice.reset_index()

# Merge dataframes
task_mice['state'] = 'Task'
awake_mice['state'] = 'Q.W.'
anes_mice['state'] = 'Anesthetized'
all_mice = pd.concat((task_mice, awake_mice, anes_mice)).reset_index()
all_mice = all_mice.sort_values(by='state')

# %% Calculate state dependency
coef_df = pd.DataFrame()
for i, region in enumerate(np.unique(all_mice['combined_region'])):
    temp_df = all_mice[all_mice['combined_region'] == region]
    this_perc_mod = [temp_df[temp_df['state'] == i].mean(numeric_only=True)['perc_mod']
                     for i in np.unique(all_mice['state'])]
    if any(np.isnan(this_perc_mod)):
        continue
    
    # Fit first order polynomial
    coef = np.polyfit(np.arange(len(this_perc_mod)), this_perc_mod, 1)
    
    # Add to df
    coef_df = pd.concat((coef_df, pd.DataFrame(index=[coef_df.shape[0]], data={
        'region': region, 'coef_1': coef[0], 'coef_2': coef[1]})))
   
coef_df = coef_df.sort_values(by='coef_1')

# %%
PLOT_REGIONS = ['PAG', 'Amyg.']
plot_colors = [colors[i] for i in colors if i in PLOT_REGIONS]

f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(data=all_mice[all_mice['combined_region'].isin(PLOT_REGIONS)],
             x='state', y='perc_mod', hue='combined_region', errorbar='se', ax=ax1,
             err_kws={'lw': 0}, palette=plot_colors, hue_order=PLOT_REGIONS)
for i, region in enumerate(PLOT_REGIONS):
    ax1.plot([0, 1, 2],
             np.polyval(coef_df.loc[coef_df['region'] == PLOT_REGIONS[i], ['coef_1', 'coef_2']].values[0], [0, 1, 2]),
             ls='--', color=plot_colors[i], lw=0.75)
ax1.set(ylabel='5-HT modulated neurons (%)', yticks=[0, 20, 40, 60, 80], xlabel='')

plt.legend(title='')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'example_regions_state_dep.pdf'))

f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.barplot(data=coef_df, x='coef_1', y='region')
ax1.set(ylabel='', xlabel='State dependency (coef.)')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'all_regions_state_dep.pdf'))