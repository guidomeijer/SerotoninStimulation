#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:45:43 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style, combine_regions, load_subjects

# Settings
MIN_NEURONS_PER_MOUSE = 5
MIN_MOD_NEURONS = 20
MIN_REC = 3

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
mod_idx_df = pd.read_pickle(join(save_path, 'mod_over_time.pickle'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    light_neurons.loc[light_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Merge regions
light_neurons['full_region'] = combine_regions(light_neurons['region'], abbreviate=True)

# Drop root and void
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'root' in j])

# Get modulated neurons
mod_neurons = light_neurons[(light_neurons['sert-cre'] == 0) & (light_neurons['modulated'] == 1)]
mod_neurons = mod_neurons.groupby('full_region').filter(lambda x: len(x) >= MIN_MOD_NEURONS)

# Calculate summary statistics
summary_df = light_neurons[light_neurons['sert-cre'] == 0].groupby(['full_region']).sum()
summary_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 0].groupby(['full_region']).size()
summary_df['modulation_index'] = light_neurons[light_neurons['sert-cre'] == 0].groupby(
    ['full_region']).mean(numeric_only=True)['mod_index']
summary_df = summary_df.reset_index()
summary_df['perc_mod'] =  (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['modulated'] >= MIN_MOD_NEURONS]

# Get ordered regions
ordered_regions = summary_df.sort_values('perc_mod', ascending=False).reset_index()

# Summary statistics per mouse
per_mouse_df = light_neurons[light_neurons['sert-cre'] == 0].groupby(
    ['full_region', 'subject']).sum(numeric_only=True)
per_mouse_df['mod_index'] = light_neurons[light_neurons['sert-cre'] == 0].groupby(
    ['full_region', 'subject']).median(numeric_only=True)['mod_index']
per_mouse_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 0].groupby(['full_region', 'subject']).size()
per_mouse_df['perc_mod'] = (per_mouse_df['modulated'] / per_mouse_df['n_neurons']) * 100
per_mouse_df = per_mouse_df[per_mouse_df['n_neurons'] >= MIN_NEURONS_PER_MOUSE]
per_mouse_df = per_mouse_df.groupby('full_region').filter(lambda x: len(x) >= MIN_REC)
per_mouse_df = per_mouse_df.reset_index()

# Add mouse number
per_mouse_df['subject_nr'] = [subjects.loc[subjects['subject'] == i, 'subject_nr'].values[0]
                              for i in per_mouse_df['subject']]

# Get ordered regions per mouse
ordered_regions_pm = per_mouse_df.groupby('full_region').mean(numeric_only=True).sort_values('perc_mod', ascending=False).reset_index()

# %%
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2.3, 2), dpi=dpi)
sns.barplot(x='perc_mod', y='full_region', data=per_mouse_df,
            order=ordered_regions_pm['full_region'],
            color=[0.6, 0.6, 0.6], ax=ax1, errorbar=None)
sns.swarmplot(x='perc_mod', y='full_region', data=per_mouse_df,
              order=ordered_regions_pm['full_region'],
              color='k', ax=ax1, size=2, legend=None)
#ax1.plot([5, 5], ax1.get_ylim(), color='grey', ls='--', lw=0.75)
ax1.set(xlabel='Modulated neurons (%)', ylabel='', xlim=[0, 82], xticks=np.arange(0, 81, 20))
#ax1.legend(frameon=False, bbox_to_anchor=(0.8, 1.1), prop={'size': 5}, title='Mouse',
#           handletextpad=0.1)

#plt.tight_layout()
plt.subplots_adjust(top=0.9, left=0.35, bottom=0.2, right=0.95)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'perc_light_modulated_neurons_per_region_no-color.pdf'))

