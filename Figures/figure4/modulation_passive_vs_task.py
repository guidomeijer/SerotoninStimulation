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
from scipy.stats import ttest_rel
from matplotlib.colors import ListedColormap
from stim_functions import paths, figure_style, load_subjects, combine_regions

# Settings
MIN_NEURONS = 15
MIN_NEURONS_PER_MOUSE = 5
MIN_REC = 2

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])
colors, dpi = figure_style()

# Load in modulation index over time
task_mod_over_time = pd.read_pickle(join(save_path, 'mod_over_time_task.pickle'))
passive_mod_over_time = pd.read_pickle(join(save_path, 'mod_over_time.pickle'))

# Load in modulation index
all_task_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
all_passive_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Add genotype and subject number
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_task_neurons.loc[all_task_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    all_passive_neurons.loc[all_passive_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only sert-cre mice
all_task_neurons = all_task_neurons[all_task_neurons['sert-cre'] == 1]
all_passive_neurons = all_passive_neurons[all_passive_neurons['sert-cre'] == 1]

# Only high level regions
all_task_neurons['full_region'] = combine_regions(all_task_neurons['region'])
all_task_neurons['full_region_name'] = combine_regions(all_task_neurons['region'], abbreviate=False)
all_passive_neurons['full_region'] = combine_regions(all_passive_neurons['region'])
all_passive_neurons['full_region_name'] = combine_regions(all_passive_neurons['region'], abbreviate=False)
all_task_neurons = all_task_neurons[all_task_neurons['full_region'] != 'root']
all_passive_neurons = all_passive_neurons[all_passive_neurons['full_region'] != 'root']

# Merge dataframes
task_neurons = pd.merge(task_mod_over_time, all_task_neurons,
                        on=['pid', 'subject', 'date', 'neuron_id', 'region'])
passive_neurons = pd.merge(passive_mod_over_time, all_passive_neurons,
                           on=['pid', 'subject', 'date', 'neuron_id', 'region'])

# Only modulated neurons
task_neurons = task_neurons[task_neurons['opto_modulated'] == 1]
passive_neurons = passive_neurons[passive_neurons['modulated'] == 1]

# Get max modulation
task_neurons['task_mod_idx'] = [i[np.argmax(np.abs(i))] for i in task_neurons['mod_idx']]
passive_neurons['passive_mod_idx'] = [i[np.argmax(np.abs(i))] for i in passive_neurons['mod_idx']]

# Get modulation index per region
grouped_df = passive_neurons.groupby(['full_region']).median(numeric_only=True)['mod_index'].to_frame()
#grouped_df = passive_neurons.groupby(['full_region']).median(numeric_only=True)['passive_mod_idx'].to_frame()
grouped_df['n_neurons'] = passive_neurons.groupby(['full_region']).size()
#grouped_df['task'] = task_neurons.groupby(['full_region']).median(numeric_only=True)['task_mod_idx']
grouped_df['task'] = task_neurons.groupby(['full_region']).median(numeric_only=True)['opto_mod_roc']
grouped_df = grouped_df.rename(columns={'mod_index': 'passive'}).reset_index()
grouped_df = grouped_df.loc[grouped_df['n_neurons'] >= MIN_NEURONS]

# Get percentage modulated neurons per region
per_passive_df = all_passive_neurons.groupby(['full_region', 'full_region_name', 'subject']).sum(numeric_only=True)
per_passive_df['n_neurons'] = all_passive_neurons.groupby(['full_region', 'full_region_name', 'subject']).size()
per_passive_df['perc_mod'] = (per_passive_df['modulated'] / per_passive_df['n_neurons']) * 100
per_passive_df = per_passive_df[per_passive_df['n_neurons'] >= MIN_NEURONS_PER_MOUSE]
per_passive_df = per_passive_df.groupby('full_region').filter(lambda x: len(x) >= MIN_REC)
per_passive_df = per_passive_df.reset_index()
perc_mod = per_passive_df[['full_region', 'full_region_name', 'perc_mod']].groupby(
    ['full_region', 'full_region_name']).mean().rename(columns={'perc_mod': 'passive_mean'})
perc_mod['passive_sem'] = per_passive_df[['full_region', 'full_region_name', 'perc_mod']].groupby(
    ['full_region', 'full_region_name']).sem()['perc_mod']

per_task_df = all_task_neurons.groupby(['full_region', 'full_region_name', 'subject']).sum(numeric_only=True)
per_task_df['n_neurons'] = all_task_neurons.groupby(['full_region', 'full_region_name', 'subject']).size()
per_task_df['perc_mod'] = (per_task_df['opto_modulated'] / per_task_df['n_neurons']) * 100
per_task_df = per_task_df[per_task_df['n_neurons'] >= MIN_NEURONS_PER_MOUSE]
per_task_df = per_task_df.groupby('full_region').filter(lambda x: len(x) >= MIN_REC)
per_task_df = per_task_df.reset_index()
perc_mod['task_mean'] = per_task_df[['full_region', 'full_region_name', 'perc_mod']].groupby(
    ['full_region', 'full_region_name']).mean()['perc_mod']
perc_mod['task_sem'] = per_passive_df[['full_region', 'full_region_name', 'perc_mod']].groupby(
    ['full_region', 'full_region_name']).sem()['perc_mod']

perc_mod = perc_mod.reset_index()

# Add colormap
perc_mod['color'] = [colors[i] for i in perc_mod['full_region']]

# Do stats
_, p = ttest_rel(perc_mod['task_mean'], perc_mod['passive_mean'])

# %%
colors, dpi = figure_style()

f, ax1 = plt.subplots(1, 1, figsize=(3.6, 2.2), dpi=dpi)
ax1.plot([0, 50], [0, 50], ls='--', color='grey', label='_nolegend_')
ax1.errorbar(perc_mod['passive_mean'], perc_mod['task_mean'],
             xerr=perc_mod['passive_sem'], yerr=perc_mod['task_sem'],
             fmt='none', ecolor=[0.7, 0.7, 0.7], capsize=2, capthick=1, zorder=0)
for _, row in perc_mod.iterrows():
    ax1.scatter(row['passive_mean'], row['task_mean'], color=row['color'], s=20, marker='s', zorder=0)
ax1.text(25, 48, '*', fontsize=12, ha='center', va='center')

ax1.set(yticks=[0, 25, 50], xticks=[0, 25, 50],
        ylim=[0, 50], xlim=[0, 50],
        xlabel='Modulated neurons passive (%)', ylabel='Modulated neurons task (%)')
ax1.legend(labels=perc_mod['full_region_name'], bbox_to_anchor=(1.05, 1.1))

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'perc_mod_passive_vs_task.pdf'))


"""

# Add colormap
grouped_df['color'] = [colors[i] for i in grouped_df['full_region']]

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
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
        ylim=[-0.3, 0.3], xlim=[-0.3, 0.3],
        xlabel='Modulation index passive')
ax1.set_ylabel('Modulation index task', labelpad=0)
sns.despine(offset=0, trim=False)
plt.tight_layout()
plt.savefig(join(fig_path, 'modulation_passive_vs_task.pdf'))

# %%
perc_mod['color'] = [colors[i] for i in perc_mod['full_region']]
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.plot([0, 50], [0, 50], ls='--', color='grey')
# this plots the colored region names
for i in perc_mod.index:
    ax1.text(perc_mod.loc[i, 'passive'],
             perc_mod.loc[i, 'task'],
             perc_mod.loc[i, 'full_region'],
             ha='center', va='center',
             color=perc_mod.loc[i, 'color'], fontsize=6, fontweight='bold')
ax1.set(yticks=[0, 25, 50], xticks=[0, 25, 50],
        ylim=[0, 50], xlim=[0, 50],
        xlabel='Modulated neurons passive (%)', ylabel='Modulated neurons task (%)')
sns.despine(offset=0, trim=False)
plt.tight_layout()
plt.savefig(join(fig_path, 'perc_mod_passive_vs_task.pdf'))
"""