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
import math
from stim_functions import paths, figure_style, load_subjects, combine_regions, remap
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas(res_um=25)

# Paths
f_path, data_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in expression
expr_df = pd.read_csv(join(data_path, 'receptor_expression.csv'))
expr_df = expr_df[~np.isin(expr_df['acronym'], ['MMme', 'CUL4, 5', 'CUL4, 5gr', 'CUL4, 5mo'])]
expr_df['region'] = combine_regions(remap(expr_df['acronym']), abbreviate=False)

# Get expression value per region per receptor
expression_mean = expr_df[['region', 'receptor', 'expression_energy']].groupby(
    ['region', 'receptor']).median().reset_index()

# Load in neural data
ephys_data = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
ephys_data['region'] = combine_regions(ephys_data['region'], abbreviate=False)
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
ephys_summary = ephys_summary[~np.isin(ephys_summary['region'], ['Barrel cortex', 'Insular cortex', 'Zona incerta', 'root'])]

# %% Plot
colors, dpi = figure_style()

plot_data = pd.merge(ephys_summary,
                     expression_mean[expression_mean['receptor'] == '5-HT7'],
                     on=['region'])
plot_data['color'] = [colors[i] for i in plot_data['region']]

fig, ax = plt.subplots(figsize=(3.5, 2), dpi=dpi, sharey=True)

r, p = pearsonr(plot_data['mod_index'], plot_data['expression_energy'])
for _, row in plot_data.iterrows():
    ax.scatter(row['mod_index'], row['expression_energy'], color=row['color'], s=20, zorder=1,
               label=row['region'])
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
    x_fit = np.linspace(plot_data['mod_index'].min(), plot_data['mod_index'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
    if p < 0.001:
        ax.text(0, 0.09, '***', fontsize=12, ha='center', va='center')
    elif p < 0.01:
        ax.text(0, 0.09, '**', fontsize=12, ha='center', va='center')
    else:
        ax.text(0, 0.09, '*', fontsize=12, ha='center', va='center')
ax.set(xlabel='Modulation index', ylabel='5-HT7 receptor expression',
       xlim=[-0.2, 0.2], xticks=[-0.2, -0.1, 0, 0.1, 0.2], xticklabels=[-0.2, -0.1, 0, 0.1, 0.2],
       yticks=[0, 0.02, 0.04, 0.06, 0.08, 0.1], yticklabels=[0, 0.02, 0.04, 0.06, 0.08, 0.1])

ax.legend(labels=plot_data['region'], bbox_to_anchor=(1.05, 1.1))

sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, '5HT7_receptor.pdf'))

