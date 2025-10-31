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
expr_df['region'] = remap(expr_df['acronym'])

# Get expression value per region per receptor
expression_mean = expr_df[['region', 'receptor', 'expression_energy']].groupby(
    ['region', 'receptor']).median().reset_index()

# Load in neural data
ephys_data = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
#ephys_data['region'] = combine_regions(ephys_data['region'])
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

# %% Plot
colors, dpi = figure_style()

"""
# htr1a
plot_data = pd.merge(ephys_summary, expression_mean[expression_mean['receptor'] == '5-HT1a'], on=['region'])
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
"""

# %% receptors

for receptor in np.unique(expression_mean['receptor']):

    plot_data = pd.merge(ephys_summary,
                         expression_mean[expression_mean['receptor'] == receptor],
                         on=['region'])
    #plot_data['color'] = [colors[i] for i in plot_data['region']]

    fig, axs = plt.subplots(1, 3, figsize=(1.5 * 3, 1.5), dpi=dpi, sharey=True)

    r, p = pearsonr(plot_data['mod_index'], plot_data['expression_energy'])
    axs[0].scatter(plot_data['mod_index'], plot_data['expression_energy'], s=15, zorder=1)
    y_max = axs[0].get_ylim()[1]
    if y_max < 0.1:
        use_max = math.ceil(y_max * 10**2) / 10**2
        use_min = -0.005
        yticks = np.round(np.arange(0, y_max+0.01, step=0.02), 3)
    elif y_max < 1:
        use_max = math.ceil(y_max * 10**1) / 10**1
        use_min = -0.02
        if y_max > 0.5:
            yticks = np.round(np.arange(0, y_max+0.1, step=0.2), 2)
        else:
            yticks = np.round(np.arange(0, y_max+0.1, step=0.1), 2)
    else:
        use_max = np.ceil(y_max)
        use_min = 0
        yticks = np.arange(0, y_max+1)
    if p < 0.05:
        slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
        x_fit = np.linspace(plot_data['mod_index'].min(), plot_data['mod_index'].max(), 100)
        y_fit = slope * x_fit + intercept
        axs[0].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
        if p < 0.001:
            axs[0].text(0, y_max*0.9, '***', fontsize=12, ha='center', va='center')
        elif p < 0.01:
            axs[0].text(0, y_max*0.9, '**', fontsize=12, ha='center', va='center')
        else:
            axs[0].text(0, y_max*0.9, '*', fontsize=12, ha='center', va='center')
    axs[0].set(xlabel='Modulation index', ylabel=f'{receptor} expression',
           xlim=[-0.2, 0.2], xticks=[-0.2, -0.1, 0, 0.1, 0.2], xticklabels=[-0.2, -0.1, 0, 0.1, 0.2],
           ylim=[use_min, use_max], yticks=yticks)

    r, p = pearsonr(plot_data['perc_mod'], plot_data['expression_energy'])
    if p < 0.05:
        slope, intercept = np.polyfit(plot_data['perc_mod'], plot_data['expression_energy'], 1)
        x_fit = np.linspace(plot_data['perc_mod'].min(), plot_data['perc_mod'].max(), 100)
        y_fit = slope * x_fit + intercept
        axs[1].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
        if p < 0.001:
            axs[1].text(25, y_max*0.9, '***', fontsize=12, ha='center', va='center')
        elif p < 0.01:
            axs[1].text(25, y_max*0.9, '**', fontsize=12, ha='center', va='center')
        else:
            axs[1].text(25, y_max*0.9, '*', fontsize=12, ha='center', va='center')
    axs[1].scatter(plot_data['perc_mod'], plot_data['expression_energy'], s=15, zorder=1)
    axs[1].set(xlabel='Modulated neurons (%)',
           xlim=[10, 40], xticks=[10, 20, 30, 40])

    r, p = pearsonr(plot_data['latency'], plot_data['expression_energy'])
    if p < 0.05:
        slope, intercept = np.polyfit(plot_data['latency'], plot_data['expression_energy'], 1)
        x_fit = np.linspace(plot_data['latency'].min(), plot_data['latency'].max(), 100)
        y_fit = slope * x_fit + intercept
        axs[2].plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
        if p < 0.001:
            axs[2].text(0.45, y_max*0.9, '***', fontsize=12, ha='center', va='center')
        elif p < 0.01:
            axs[2].text(0.45, y_max*0.9, '**', fontsize=12, ha='center', va='center')
        else:
            axs[2].text(0.45, y_max*0.9, '*', fontsize=12, ha='center', va='center')
    axs[2].scatter(plot_data['latency'], plot_data['expression_energy'], s=15, zorder=1)
    axs[2].set(xlabel='Modulation latency (s)',
               xticks=[0.3, 0.4, 0.5, 0.6])

    plt.tight_layout()
    sns.despine(trim=True)

    #plt.savefig(join(fig_path, f'{receptor}_receptor.pdf'))

