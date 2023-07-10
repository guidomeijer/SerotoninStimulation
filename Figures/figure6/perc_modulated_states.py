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
from stim_functions import paths, figure_style, load_subjects, high_level_regions

# Settings
N_BINS = 30
MIN_NEURONS = 0

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

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

# Only high level regions
task_neurons['high_level_region'] = high_level_regions(task_neurons['region'])
awake_neurons['high_level_region'] = high_level_regions(awake_neurons['region'])
anes_neurons['high_level_region'] = high_level_regions(anes_neurons['region'])
task_neurons = task_neurons[task_neurons['high_level_region'] != 'root']
awake_neurons = awake_neurons[awake_neurons['high_level_region'] != 'root']
anes_neurons = anes_neurons[anes_neurons['high_level_region'] != 'root']

# Calculate percentage modulated neurons
task_mice = ((task_neurons.groupby(['subject', 'subject_nr']).sum()['opto_modulated']
              / task_neurons.groupby(['subject', 'subject_nr']).size() * 100).to_frame())
task_mice = task_mice.rename(columns={0: 'perc_mod'})
awake_mice = ((awake_neurons.groupby(['subject', 'subject_nr']).sum()['modulated']
              / awake_neurons.groupby(['subject', 'subject_nr']).size() * 100).to_frame())
awake_mice = awake_mice.rename(columns={0: 'perc_mod'})
anes_mice = ((anes_neurons.groupby(['subject', 'subject_nr']).sum()['modulated']
              / anes_neurons.groupby(['subject', 'subject_nr']).size() * 100).to_frame())
anes_mice = anes_mice.rename(columns={0: 'perc_mod'})

# Add modulation index
task_mice['mod_index'] = task_neurons[task_neurons['opto_modulated']].groupby(
    ['subject', 'subject_nr']).median(numeric_only=True)['opto_mod_roc']
awake_mice['mod_index'] = awake_neurons[awake_neurons['modulated']].groupby([
    'subject', 'subject_nr']).median(numeric_only=True)['mod_index_late']
anes_mice['mod_index'] = anes_neurons[anes_neurons['modulated']].groupby(
    ['subject', 'subject_nr']).median(numeric_only=True)['mod_index_late']

# Reset index
task_mice = task_mice.reset_index()
awake_mice = awake_mice.reset_index()
anes_mice = anes_mice.reset_index()

# Merge dataframes
task_mice['state'] = 'task'
awake_mice['state'] = 'awake'
anes_mice['state'] = 'anesthetized'
all_mice = pd.concat((task_mice, awake_mice, anes_mice)).reset_index()

# Stats
_, p = kruskal(all_mice.loc[all_mice['state'] == 'task', 'perc_mod'],
               all_mice.loc[all_mice['state'] == 'awake', 'perc_mod'],
               all_mice.loc[all_mice['state'] == 'anesthetized', 'perc_mod'])
print(f'Percentage modulated Kruskal-wallis p value: {p}')
dunn_ph = sp.posthoc_conover(all_mice, val_col='perc_mod', group_col='state', p_adjust='bonferroni')
print(dunn_ph)  

_, p = kruskal(all_mice.loc[all_mice['state'] == 'task', 'mod_index'],
               all_mice.loc[all_mice['state'] == 'awake', 'mod_index'],
               all_mice.loc[all_mice['state'] == 'anesthetized', 'mod_index'])
print(f'Modulation index Kruskal-wallis p value: {p}')
dunn_ph = sp.posthoc_conover(all_mice, val_col='mod_index', group_col='state', p_adjust='bonferroni')
print(dunn_ph)        



# %%
f, ax1 = plt.subplots(figsize=(1.2, 1.75), dpi=dpi)
for i in all_mice[all_mice['state'] == 'anesthetized'].index:
    if all_mice.loc[i, 'subject'] == 'ZFM-05169':
        jitter = 0.05
    elif all_mice.loc[i, 'subject'] == 'ZFM-05492':
        jitter = -0.05
    else:
        jitter = 0
    ax1.plot(1+jitter, all_mice.loc[i, 'perc_mod'],
             color='k', marker='o', ms=3,
             markeredgewidth=0.4, markeredgecolor='w')
for i in all_mice[all_mice['state'] == 'awake'].index:
    task_mod = all_mice.loc[(all_mice['subject'] == all_mice.loc[i, 'subject'])
                            & (all_mice['state'] == 'task'), 'perc_mod']
    if len(task_mod) > 0:
        ax1.plot([2, 3],
                 [all_mice.loc[i, 'perc_mod'], task_mod.values[0]],
                 marker='o', ms=3, markeredgewidth=0.4, lw=0.75, color='grey',
                 markeredgecolor='w', markerfacecolor='k')
    else:
        ax1.plot(2, all_mice.loc[i, 'perc_mod'],
                 color='k', marker='o', ms=3, markeredgewidth=0.4,
                 markeredgecolor='w')
ax1.plot([1, 2, 3], all_mice.groupby('state').mean(numeric_only=True)['perc_mod'], marker='_',
         color='tab:red', lw=0, ms=7)
ax1.plot([1, 3], [82, 82], color='k', lw=0.75)
ax1.plot([1, 1.95], [80, 80], color='k', lw=0.75)
ax1.plot([2.05, 3], [80, 80], color='k', lw=0.75)
ax1.text(2, 80, '**', fontsize=10, ha='center')
ax1.set(ylabel='Modulated neurons (%)', xlabel='', yticks=np.arange(0, 81, 20), xlim=[0.75, 3.25])
ax1.set_xticklabels(['Anesthetized', 'Wakefullness', 'Behaving'], rotation=45, ha='right')
ax1.tick_params(axis='x', which='major', pad=0)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'perc_mod_states.pdf'))



# %%
f, ax1 = plt.subplots(figsize=(1.25, 1.75), dpi=dpi)
for i in all_mice[all_mice['state'] == 'anesthetized'].index:
    
    ax1.plot(1, all_mice.loc[i, 'mod_index'],
             color='k', marker='o', ms=3,
             markeredgewidth=0.4, markeredgecolor='w')
for i in all_mice[all_mice['state'] == 'awake'].index:
    task_mod = all_mice.loc[(all_mice['subject'] == all_mice.loc[i, 'subject'])
                            & (all_mice['state'] == 'task'), 'mod_index']
    if len(task_mod) > 0:
        ax1.plot([2, 3],
                 [all_mice.loc[i, 'mod_index'], task_mod.values[0]],
                 marker='o', ms=3, markeredgewidth=0.4, lw=0.75, color='grey',
                 markeredgecolor='w', markerfacecolor='k')
    else:
        ax1.plot(2, all_mice.loc[i, 'mod_index'],
                 color='k', marker='o', ms=3, markeredgewidth=0.4,
                 markeredgecolor='w')
ax1.plot([1, 2, 3], all_mice.groupby('state').mean(numeric_only=True)['mod_index'], marker='_',
         color='tab:red', lw=0, ms=7)

ax1.plot([1, 3], [0.22, 0.22], color='k', lw=0.75)
ax1.plot([1, 1.95], [0.21, 0.21], color='k', lw=0.75)
#ax1.plot([2.05, 3], [90, 90], color='k', lw=0.75)
ax1.text(2, 0.22, '*', fontsize=10, ha='center')

ax1.set(ylabel='Median modulation index', xlabel='', xlim=[0.75, 3.25], yticks=np.arange(-0.3, 0.21, 0.1))
ax1.set_xticklabels(['Anesthetized', 'Wakefullness', 'Behaving'], rotation=45, ha='right')
ax1.tick_params(axis='x', which='major', pad=0)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'mod_index_states.pdf'))


"""
# %%
f, ax1 = plt.subplots(figsize=(1.4, 1.75), dpi=dpi)
for i in all_mice[all_mice['state'] == 'anesthetized'].index:
    if all_mice.loc[i, 'subject'] == 'ZFM-05169':
        jitter = 0.05
    elif all_mice.loc[i, 'subject'] == 'ZFM-05492':
        jitter = -0.05
    else:
        jitter = 0
    ax1.plot(1+jitter, all_mice.loc[i, 'perc_mod'],
             color='k', marker='o', ms=3,
             markeredgewidth=0.4, markeredgecolor='w')
for i in all_mice[all_mice['state'] == 'awake'].index:
    ax1.plot(2, all_mice.loc[i, 'perc_mod'],
             color='k', marker='o', ms=3, markeredgewidth=0.4,
             markeredgecolor='w')
for i in all_mice[all_mice['state'] == 'task'].index:
    ax1.plot(3, all_mice.loc[i, 'perc_mod'],
             color='k', marker='o', ms=3, markeredgewidth=0.4,
             markeredgecolor='w')
ax1.plot([1, 2, 3], all_mice.groupby('state').mean(numeric_only=True)['perc_mod'], marker='_',
         color='tab:red', lw=0, ms=7)
ax1.plot([1, 3], [92, 92], color='k', lw=0.75)
ax1.plot([1, 1.95], [90, 90], color='k', lw=0.75)
ax1.plot([2.05, 3], [90, 90], color='k', lw=0.75)
ax1.text(2, 90, '**', fontsize=10, ha='center')
ax1.set(ylabel='Modulated neurons (%)', xlabel='', yticks=np.arange(0, 101, 20), xlim=[0.75, 3.25])
ax1.set_xticklabels(['Anesthetized', 'Wakefullness', 'Behaving'], rotation=45, ha='right')
ax1.tick_params(axis='x', which='major', pad=0)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'light_mod_summary_states_no_lines.pdf'))
"""


"""
# %% Plot percentage mod neurons
PROPS = {'boxprops':{'facecolor':'none', 'edgecolor':'none'}, 'medianprops':{'color':'none'},
         'whiskerprops':{'color':'none'}, 'capprops':{'color':'none'}}
ORDER = ['anesthetized', 'awake', 'task']
f, ax1 = plt.subplots(1, 1, figsize=(1.4, 1.75), dpi=dpi)

this_cmap = ListedColormap([colors['subject_palette'][i] for i in all_mice['subject_nr'].astype(int)])

f.subplots_adjust(bottom=0.2, left=0.35, right=0.85, top=0.9)
#sns.stripplot(x='sert-cre', y='perc_mod', data=all_mice, order=[1, 0], size=3,
#              palette=[colors['sert'], colors['wt']], ax=ax1, jitter=0.2)
sns.swarmplot(x='state', y='perc_mod', data=all_mice, order=ORDER,
              size=2.5, hue='subject_nr', palette=this_cmap, legend=None, zorder=2, ax=ax1)
sns.boxplot(x='state', y='perc_mod', ax=ax1, data=all_mice, showmeans=True,
            order=ORDER, meanprops={"marker": "_", "markeredgecolor": "red", "markersize": "8"},
            fliersize=0, zorder=1, **PROPS)
ax1.set(ylabel='Modulated neurons (%)', xlabel='', yticks=np.arange(0, 101, 20))
ax1.set_xticklabels(['Anesthetized', 'Wakefullness', 'Behaving'], rotation=45, ha='right')
ax1.tick_params(axis='both', which='major', pad=0)

sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, 'light_mod_summary_states_swarm.pdf'))
"""
