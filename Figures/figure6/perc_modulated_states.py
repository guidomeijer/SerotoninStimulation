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

# Calculate percentage modulated neurons
task_mice = ((task_neurons.groupby(['subject', 'subject_nr']).sum()['opto_modulated']
              / task_neurons.groupby(['subject', 'subject_nr']).size() * 100).to_frame().reset_index())
awake_mice = ((awake_neurons.groupby(['subject', 'subject_nr']).sum()['modulated']
              / awake_neurons.groupby(['subject', 'subject_nr']).size() * 100).to_frame().reset_index())
anes_mice = ((anes_neurons.groupby(['subject', 'subject_nr']).sum()['modulated']
              / anes_neurons.groupby(['subject', 'subject_nr']).size() * 100).to_frame().reset_index())

# Merge dataframes
task_mice = task_mice.rename(columns={0: 'perc_mod'})
task_mice['state'] = 'task'
awake_mice = awake_mice.rename(columns={0: 'perc_mod'})
awake_mice['state'] = 'awake'
anes_mice = anes_mice.rename(columns={0: 'perc_mod'})
anes_mice['state'] = 'anesthetized'
all_mice = pd.concat((task_mice, awake_mice, anes_mice))

# %% Plot percentage mod neurons
colors, dpi = figure_style()
PROPS = {'boxprops':{'facecolor':'none', 'edgecolor':'none'}, 'medianprops':{'color':'none'},
         'whiskerprops':{'color':'none'}, 'capprops':{'color':'none'}}
ORDER = ['anesthetized', 'awake', 'task']
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

this_cmap = ListedColormap([colors['subject_palette'][i] for i in np.sort(all_mice['subject_nr']).astype(int)])

f.subplots_adjust(bottom=0.2, left=0.35, right=0.85, top=0.9)
#sns.stripplot(x='sert-cre', y='perc_mod', data=all_mice, order=[1, 0], size=3,
#              palette=[colors['sert'], colors['wt']], ax=ax1, jitter=0.2)
sns.swarmplot(x='state', y='perc_mod', data=all_mice, order=ORDER,
              size=2.5, color='k', legend=None, zorder=2, ax=ax1)
sns.boxplot(x='state', y='perc_mod', ax=ax1, data=all_mice, showmeans=True,
            order=ORDER, meanprops={"marker": "_", "markeredgecolor": "red", "markersize": "8"},
            fliersize=0, zorder=1, **PROPS)
ax1.set(xticklabels=['Anesthetized', 'Quiet wakefullness', 'Task performing'],
        ylabel='Modulated neurons (%)', xlabel='', yticks=np.arange(0, 101, 20))
ax1.set_xticklabels(['Anesthetized', 'Quiet\nwakefullness', 'Task\nperforming'], rotation=45,
                    ha='right')

sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, 'light_mod_summary_states.pdf'))
