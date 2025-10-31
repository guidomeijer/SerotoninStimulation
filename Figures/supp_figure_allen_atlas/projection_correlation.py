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
from stim_functions import paths, figure_style, load_subjects, combine_regions, remap
colors, dpi = figure_style()

# Paths
f_path, data_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# load structure and expression data set
proj_df = pd.read_csv(join(data_path, 'dr_projection_strength.csv'))
proj_df = proj_df[~np.isin(proj_df['allen_acronym'], ['MMd', 'MMme', 'MMl', 'MMm', 'MMp', 'CUL4, 5'])]
proj_df['region'] = combine_regions(remap(proj_df['allen_acronym']))
proj_summary = proj_df[['region', 'projection_density']].groupby(['region']).mean().reset_index()

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
ephys_summary[['mod_index', 'latency']] = ephys_data[(ephys_data['sert-cre'] == 1) & (ephys_data['modulated'] == 1)][[
    'region', 'mod_index', 'latenzy']].groupby('region').mean()[['mod_index', 'latenzy']]

# Drop some regions
ephys_summary = ephys_summary.reset_index()
ephys_summary = ephys_summary[~np.isin(ephys_summary['region'], ['AI', 'ZI', 'root', 'BC'])]

# Merge for plotting
plot_data = pd.merge(ephys_summary, proj_summary, on=['region'])
plot_data['color'] = [colors[i] for i in plot_data['region']]

# %% Plot
fig, axs = plt.subplots(1, 3, figsize=(2.1 * 3, 2), dpi=dpi, sharey=True)

r, p = pearsonr(plot_data['mod_index'], plot_data['projection_density'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['projection_density'], 1)
    x_fit = np.linspace(-0.3, 0.3, 100)
    y_fit = slope * x_fit + intercept
    axs[1].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
for _, row in plot_data.iterrows():
    axs[0].scatter(row['mod_index'], row['projection_density'], color=row['color'], s=20, zorder=1)
#axs[0].text(0, 0.079, f'p={np.round(p, 2)}', ha='center', va='center')
axs[0].set(xlabel='5-HT modulation index', ylabel='Dorsal raphe projection strength',
       xlim=[-0.2, 0.2], xticks=[-0.2, -0.1, 0, 0.1, 0.2], xticklabels=[-0.2, -0.1, 0, 0.1, 0.2],
       ylim=[-0.005, 0.12], yticks=np.arange(0.13, step=0.02))

r, p = pearsonr(plot_data['perc_mod'], plot_data['projection_density'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['projection_density'], 1)
    x_fit = np.linspace(-0.3, 0.3, 100)
    y_fit = slope * x_fit + intercept
    axs[1].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
for _, row in plot_data.iterrows():
    axs[1].scatter(row['perc_mod'], row['projection_density'], color=row['color'], s=20, zorder=1)
#axs[1].text(25, 0.079, f'p={np.round(p, 2)}', ha='center', va='center')
axs[1].set(xlabel='5-HT modulated neurons (%)',
       xlim=[10, 40], xticks=[10, 20, 30, 40])

r, p = pearsonr(plot_data['latency'], plot_data['projection_density'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['projection_density'], 1)
    x_fit = np.linspace(0.3, 0.6, 100)
    y_fit = slope * x_fit + intercept
    axs[2].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
for _, row in plot_data.iterrows():
    axs[2].scatter(row['latency'], row['projection_density'], color=row['color'], s=20, zorder=1)
#axs[2].text(0.45, 0.079, f'p={np.round(p, 2)}', ha='center', va='center')
axs[2].set(xlabel='5-HT modulation latency (s)',
       xlim=[0.25, 0.6], xticks=[0.3, 0.4, 0.5, 0.6])

axs[2].legend(labels=plot_data['region'], bbox_to_anchor=(1.05, 1.1))

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'dr_projection.pdf'))

