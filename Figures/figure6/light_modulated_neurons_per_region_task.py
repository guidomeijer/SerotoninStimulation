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
from stim_functions import paths, figure_style, load_subjects, combine_regions

# Settings
MIN_NEURONS_PER_MOUSE = 5
MIN_MOD_NEURONS = 20
MIN_REC = 3

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in results
light_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    light_neurons.loc[light_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Merge regions
light_neurons['high_level_region'] = combine_regions(light_neurons['region'])

# Drop root and void
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['high_level_region']) if 'root' in j])

# Get modulated neurons
mod_neurons = light_neurons[(light_neurons['sert-cre'] == 1) & (light_neurons['opto_modulated'] == 1)]
mod_neurons = mod_neurons.groupby('high_level_region').filter(lambda x: len(x) >= MIN_MOD_NEURONS)

# Calculate summary statistics
summary_df = light_neurons[light_neurons['sert-cre'] == 1].groupby(['high_level_region']).sum()
summary_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(['high_level_region']).size()
summary_df['modulation_index'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(
    ['high_level_region']).mean(numeric_only=True)['opto_mod_roc']
summary_df = summary_df.reset_index()
summary_df['perc_mod'] =  (summary_df['opto_modulated'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['opto_modulated'] >= MIN_MOD_NEURONS]

summary_no_df = light_neurons[light_neurons['sert-cre'] == 0].groupby(['high_level_region']).sum()
summary_no_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 0].groupby(['high_level_region']).size()
summary_no_df = summary_no_df.reset_index()
summary_no_df['perc_mod'] =  (summary_no_df['opto_modulated'] / summary_no_df['n_neurons']) * 100
summary_no_df = summary_no_df[summary_no_df['opto_modulated'] >= MIN_MOD_NEURONS]
summary_no_df = pd.concat((summary_no_df, pd.DataFrame(data={
    'high_level_region': summary_df.loc[~summary_df['high_level_region'].isin(summary_no_df['high_level_region']), 'high_level_region'],
    'perc_mod': np.zeros(np.sum(~summary_df['high_level_region'].isin(summary_no_df['high_level_region'])))})))

# Get ordered regions
ordered_regions = summary_df.sort_values('perc_mod', ascending=False).reset_index()

# Summary statistics per mouse
per_mouse_df = light_neurons[light_neurons['sert-cre'] == 1].groupby(
    ['high_level_region', 'subject']).sum(numeric_only=True)
per_mouse_df['opto_mod_roc'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(
    ['high_level_region', 'subject']).median(numeric_only=True)['opto_mod_roc']
per_mouse_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(['high_level_region', 'subject']).size()
per_mouse_df['perc_mod'] = (per_mouse_df['opto_modulated'] / per_mouse_df['n_neurons']) * 100
per_mouse_df = per_mouse_df[per_mouse_df['n_neurons'] >= MIN_NEURONS_PER_MOUSE]
per_mouse_df = per_mouse_df.groupby('high_level_region').filter(lambda x: len(x) >= MIN_REC)
per_mouse_df = per_mouse_df.reset_index()

# Add mouse number
per_mouse_df['subject_nr'] = [subjects.loc[subjects['subject'] == i, 'subject_nr'].values[0]
                              for i in per_mouse_df['subject']]

# Get ordered regions per mouse
ordered_regions_pm = per_mouse_df.groupby('high_level_region').mean(numeric_only=True).sort_values('perc_mod', ascending=False).reset_index()

# %% Plot percentage modulated neurons per region

colors, dpi = figure_style()
this_cmap = [colors['subject_palette'][i] for i in np.unique(per_mouse_df['subject_nr'])]

f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
sns.barplot(x='perc_mod', y='high_level_region', data=per_mouse_df,
            order=ordered_regions_pm['high_level_region'],
            color=[0.6, 0.6, 0.6], ax=ax1, errorbar=None)
sns.swarmplot(x='perc_mod', y='high_level_region', data=per_mouse_df,
              order=ordered_regions_pm['high_level_region'],
              hue='subject_nr', palette=this_cmap, ax=ax1, size=2, legend=None)
ax1.set(xlabel='Modulated neurons (%)', ylabel='', xlim=[0, 60], xticks=np.arange(0, 61, 20))
#ax1.legend(frameon=False, bbox_to_anchor=(0.8, 1.1), prop={'size': 5}, title='Mouse',
#           handletextpad=0.1)

plt.tight_layout()
#plt.subplots_adjust(left=0.35, bottom=0.2, right=0.95)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'perc_light_modulated_neurons_per_region.pdf'))

# %%
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 2), dpi=dpi)
sns.barplot(x='perc_mod', y='high_level_region', data=per_mouse_df,
            order=ordered_regions_pm['high_level_region'],
            color=[0.6, 0.6, 0.6], ax=ax1, errorbar=None)
sns.swarmplot(x='perc_mod', y='high_level_region', data=per_mouse_df,
              order=ordered_regions_pm['high_level_region'],
              color='k', ax=ax1, size=2, legend=None)
ax1.set(xlabel='Modulated neurons (%)', ylabel='', xlim=[0, 80], xticks=np.arange(0, 81, 20))
#ax1.legend(frameon=False, bbox_to_anchor=(0.8, 1.1), prop={'size': 5}, title='Mouse',
#           handletextpad=0.1)

plt.tight_layout()
#plt.subplots_adjust(left=0.35, bottom=0.2, right=0.95)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'perc_light_modulated_neurons_per_region_no-color.pdf'))

# %%

PROPS = {'boxprops':{'facecolor':'none', 'edgecolor':'none'}, 'medianprops':{'color':'none'},
         'whiskerprops':{'color':'none'}, 'capprops':{'color':'none'}}
ORDER = mod_neurons.groupby('high_level_region').mean(numeric_only=True)['opto_mod_roc'].sort_values(
    ascending=False).reset_index()['high_level_region']

f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
sns.stripplot(x='opto_mod_roc', y='high_level_region', ax=ax1, data=mod_neurons, order=ORDER,
              size=2, color='grey', zorder=1)
sns.boxplot(x='opto_mod_roc', y='high_level_region', ax=ax1,
            data=mod_neurons[mod_neurons['high_level_region'] != 'Amygdala'], showmeans=True,
            order=ORDER, meanprops={"marker": "|", "markeredgecolor": "red", "markersize": "8"},
            fliersize=0, zorder=2, **PROPS)
ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', zorder=0)
ax1.set(ylabel='', xlabel='Modulation index', xlim=[-1.05, 1.05], xticklabels=[-1, 0, 1])
#ax1.spines['bottom'].set_position(('data', np.floor(ax1.get_ylim()[0]) - 0.4))
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_modulation_per_neuron_per_region.pdf'))
