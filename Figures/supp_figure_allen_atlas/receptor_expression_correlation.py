#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from scipy.stats import pearsonr
from stim_functions import paths, figure_style, load_subjects, combine_regions
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas(res_um=25)
colors, dpi = figure_style()

# Paths
f_path, data_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# load structure and expression data set 
receptor_expressions_orig = pd.read_csv(join(data_path, 'receptor_expressions.csv'))
structures_leaves = pd.read_csv(join(data_path, 'structures_leaves.csv'))

# only load the expression values for regions we care about 
receptor_expressions_regions = receptor_expressions_orig[receptor_expressions_orig['structure_id'].isin(structures_leaves['id'])]

# add the beryl names to the dataframe
receptor_expressions_regions['beryl_acronym'] = receptor_expressions_regions['structure_id'].apply(lambda x: structures_leaves['beryl_acronyms'].iloc[np.where(structures_leaves['id'].values==x)[0][0]])

# add the translation for VISa and VISrl (which is now both PTL)
PTL_id = structures_leaves[structures_leaves['parent_structure_id']==22]['id'].values
receptor_expressions_regions.loc[receptor_expressions_regions['structure_id'].isin(PTL_id), 'beryl_acronym'] = 'VISa'

# convert to region names
receptor_expressions_regions['region'] = combine_regions(receptor_expressions_regions['beryl_acronym'])

# Get expression value per region per receptor 
expression_mean = receptor_expressions_regions[['region', 'receptor_name', 'expression_energy']].groupby(
    ['region', 'receptor_name']).mean().reset_index()
htr1a = expression_mean[expression_mean['receptor_name']=='htr1a']
htr2a = expression_mean[expression_mean['receptor_name']=='htr2a']
htr3a = expression_mean[expression_mean['receptor_name']=='htr3a']
htr2c = expression_mean[expression_mean['receptor_name']=='htr2c']

# Load in neural data
ephys_data = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
ephys_data['region'] = combine_regions(ephys_data['region'])
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    ephys_data.loc[ephys_data['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Calculate percentage modulated neurons 
per_mouse_df = ephys_data[ephys_data['sert-cre'] == 1].groupby(['region', 'subject']).sum(numeric_only=True)
per_mouse_df['n_neurons'] = ephys_data[ephys_data['sert-cre'] == 1].groupby(['region', 'subject']).size()
per_mouse_df['perc_mod'] = (per_mouse_df['modulated'] / per_mouse_df['n_neurons']) * 100
per_mouse_df = per_mouse_df.reset_index()
ephys_summary = per_mouse_df[['region', 'perc_mod']].groupby('region').mean()

# Calculate summary per neuron
ephys_summary['mod_index'] = ephys_data[(ephys_data['sert-cre'] == 1) & (ephys_data['modulated'] == 1)][[
    'region', 'mod_index']].groupby('region').mean()['mod_index']

# Drop some regions
ephys_summary = ephys_summary.reset_index()
ephys_summary = ephys_summary[~np.isin(ephys_summary['region'], ['AI', 'ZI', 'root', 'BC'])]

# %% Plot

# htr1a
plot_data = pd.merge(ephys_summary, htr1a, on=['region'])
plot_data['color'] = [colors[i] for i in plot_data['region']]

r, p = pearsonr(plot_data['mod_index'], plot_data['expression_energy'])
print(f'5-HT1a receptor; p = {np.round(p, 2)}')

slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
x_fit = np.linspace(-0.3, 0.3, 100)
y_fit = slope * x_fit + intercept

fig, ax = plt.subplots(figsize=(3.1, 2.3), dpi=dpi)
for _, row in plot_data.iterrows():
    ax.scatter(row['mod_index'], row['expression_energy'], color=row['color'], s=20, zorder=1)
ax.set(xlabel='5-HT modulation index', ylabel='5-HT1A receptor expression', title='5-HT1A',
       xlim=[-0.2, 0.2], xticks=[-0.2, 0, 0.2], xticklabels=[-0.2, 0, 0.2],
       ylim=[0, 3], yticks=np.arange(4))
ax.legend(labels=plot_data['region'], bbox_to_anchor=(1.05, 1.1))
plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'legend.pdf'))

# %% htr1a
plot_data = pd.merge(ephys_summary, htr1a, on=['region'])
plot_data['color'] = [colors[i] for i in plot_data['region']]

fig, axs = plt.subplots(1, 2, figsize=(1.75 * 2, 1.75), dpi=dpi, sharey=True)

r, p = pearsonr(plot_data['mod_index'], plot_data['expression_energy'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
    x_fit = np.linspace(-0.3, 0.3, 100)
    y_fit = slope * x_fit + intercept
    axs[1].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
for _, row in plot_data.iterrows():
    axs[0].scatter(row['mod_index'], row['expression_energy'], color=row['color'], s=20, zorder=1)
axs[0].text(0, 2.9, f'p={np.round(p, 2)}', ha='center', va='center')
axs[0].set(xlabel='5-HT modulation index', ylabel='5-HT1A receptor expression',
       xlim=[-0.2, 0.2], xticks=[-0.2, -0.1, 0, 0.1, 0.2], xticklabels=[-0.2, -0.1, 0, 0.1, 0.2],
       ylim=[0, 3], yticks=np.arange(4))

r, p = pearsonr(plot_data['perc_mod'], plot_data['expression_energy'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
    x_fit = np.linspace(-0.3, 0.3, 100)
    y_fit = slope * x_fit + intercept
    axs[1].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
for _, row in plot_data.iterrows():
    axs[1].scatter(row['perc_mod'], row['expression_energy'], color=row['color'], s=20, zorder=1)
axs[1].text(25, 2.9, f'p={np.round(p, 2)}', ha='center', va='center')
axs[1].set(xlabel='5-HT modulated neurons (%)',
       xlim=[10, 40], xticks=[10, 20, 30, 40],
       ylim=[0, 3], yticks=np.arange(4))

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, '5HT1A_receptor.pdf'))

# %% htr2a
plot_data = pd.merge(ephys_summary, htr2a, on=['region'])
plot_data['color'] = [colors[i] for i in plot_data['region']]

fig, axs = plt.subplots(1, 2, figsize=(1.75 * 2, 1.75), dpi=dpi, sharey=True)

r, p = pearsonr(plot_data['mod_index'], plot_data['expression_energy'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
    x_fit = np.linspace(-0.3, 0.3, 100)
    y_fit = slope * x_fit + intercept
    axs[1].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
for _, row in plot_data.iterrows():
    axs[0].scatter(row['mod_index'], row['expression_energy'], color=row['color'], s=20, zorder=1)
axs[0].text(0, 7.75, f'p={np.round(p, 2)}', ha='center', va='center')
axs[0].set(xlabel='5-HT modulation index', ylabel='5-HT2A receptor expression',
       xlim=[-0.2, 0.2], xticks=[-0.2, -0.1, 0, 0.1, 0.2], xticklabels=[-0.2, -0.1, 0, 0.1, 0.2],
       ylim=[0, 8], yticks=np.arange(9, step=2))

r, p = pearsonr(plot_data['perc_mod'], plot_data['expression_energy'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
    x_fit = np.linspace(-0.3, 0.3, 100)
    y_fit = slope * x_fit + intercept
    axs[1].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
for _, row in plot_data.iterrows():
    axs[1].scatter(row['perc_mod'], row['expression_energy'], color=row['color'], s=20, zorder=1)
axs[1].text(25, 7.75, f'p={np.round(p, 2)}', ha='center', va='center')
axs[1].set(xlabel='5-HT modulated neurons (%)',
       xlim=[10, 40], xticks=[10, 20, 30, 40])

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, '5HT2A_receptor.pdf'))

# %% htr2c
plot_data = pd.merge(ephys_summary, htr2c, on=['region'])
plot_data['color'] = [colors[i] for i in plot_data['region']]

fig, axs = plt.subplots(1, 2, figsize=(1.75 * 2, 1.75), dpi=dpi, sharey=True)

r, p = pearsonr(plot_data['mod_index'], plot_data['expression_energy'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
    x_fit = np.linspace(-0.3, 0.3, 100)
    y_fit = slope * x_fit + intercept
    axs[1].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
for _, row in plot_data.iterrows():
    axs[0].scatter(row['mod_index'], row['expression_energy'], color=row['color'], s=20, zorder=1)
axs[0].text(0, 6.75, f'p={np.round(p, 2)}', ha='center', va='center')
axs[0].set(xlabel='5-HT modulation index', ylabel='5-HT2C receptor expression',
       xlim=[-0.2, 0.2], xticks=[-0.2, -0.1, 0, 0.1, 0.2], xticklabels=[-0.2, -0.1, 0, 0.1, 0.2],
       ylim=[-0.2, 7], yticks=np.arange(8))

r, p = pearsonr(plot_data['perc_mod'], plot_data['expression_energy'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
    x_fit = np.linspace(-0.3, 0.3, 100)
    y_fit = slope * x_fit + intercept
    axs[1].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
for _, row in plot_data.iterrows():
    axs[1].scatter(row['perc_mod'], row['expression_energy'], color=row['color'], s=20, zorder=1)
axs[1].text(25, 6.75, f'p={np.round(p, 2)}', ha='center', va='center')
axs[1].set(xlabel='5-HT modulated neurons (%)',
       xlim=[10, 40], xticks=[10, 20, 30, 40])

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, '5HT2C_receptor.pdf'))

# %% htr3a
plot_data = pd.merge(ephys_summary, htr3a, on=['region'])
plot_data['color'] = [colors[i] for i in plot_data['region']]

fig, axs = plt.subplots(1, 2, figsize=(1.75 * 2, 1.75), dpi=dpi, sharey=True)

r, p = pearsonr(plot_data['mod_index'], plot_data['expression_energy'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
    x_fit = np.linspace(-0.3, 0.3, 100)
    y_fit = slope * x_fit + intercept
    axs[1].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
for _, row in plot_data.iterrows():
    axs[0].scatter(row['mod_index'], row['expression_energy'], color=row['color'], s=20, zorder=1)
axs[0].text(0, 1.15, f'p={np.round(p, 2)}', ha='center', va='center')
axs[0].set(xlabel='5-HT modulation index', ylabel='5-HT3A receptor expression',
       xlim=[-0.2, 0.2], xticks=[-0.2, -0.1, 0, 0.1, 0.2], xticklabels=[-0.2, -0.1, 0, 0.1, 0.2],
       ylim=[-0.05, 1.2], yticks=np.arange(1.3, step=0.2))

r, p = pearsonr(plot_data['perc_mod'], plot_data['expression_energy'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
    x_fit = np.linspace(-0.3, 0.3, 100)
    y_fit = slope * x_fit + intercept
    axs[1].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
for _, row in plot_data.iterrows():
    axs[1].scatter(row['perc_mod'], row['expression_energy'], color=row['color'], s=20, zorder=1)
axs[1].text(25, 1.15, f'p={np.round(p, 2)}', ha='center', va='center')
axs[1].set(xlabel='5-HT modulated neurons (%)',
       xlim=[10, 40], xticks=[10, 20, 30, 40])

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, '5HT3A_receptor.pdf'))