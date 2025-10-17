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
from iblatlas.atlas import BrainRegions
br = BrainRegions()
colors, dpi = figure_style()

# Paths
f_path, data_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Get glutamate co-expression data
glut_df = pd.DataFrame(data={
    'region': ['PVH', 'CeA', 'LHb', 'DLG', 'OB', 'OFC', 'PIR', 'ENT'],
    'perc_glut': [18, 21, 10, 10, 82, 70, 50, 68]})

# Load in neural data
ephys_data = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
ephys_data['region'] = combine_regions(ephys_data['region'])
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    ephys_data.loc[ephys_data['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Create mapping of to fmri regions
acronym_map = {
    'LH': 'LHb', 'LGd': 'DLG', 'MOB': 'OB', 'ORB': 'OFC', 'CEA': 'CeA'
}
regions = np.array(['root'] * ephys_data.shape[0], dtype=object)
for allen_acronym, custom_region in acronym_map.items():
    descendant_acronyms = br.descendants(br.acronym2id(allen_acronym))['acronym']
    mask = np.isin(ephys_data['allen_acronym'], descendant_acronyms)
    regions[mask] = custom_region
ephys_data['region'] = regions

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
ephys_summary = ephys_summary[~np.isin(ephys_summary['region'], ['root'])]

# Merge
plot_data = pd.merge(ephys_summary, glut_df, on=['region'])
plot_data['color'] = sns.color_palette('Set2')[:3]

# %%

fig, axs = plt.subplots(1, 2, figsize=(1.75 * 2, 1.75), dpi=dpi, sharey=True)

for _, row in plot_data.iterrows():
    axs[0].scatter(row['mod_index'], row['perc_glut'], color=row['color'], s=20, zorder=1)
axs[0].set(xlabel='5-HT modulation index', ylabel='vGLUT3 co-expression (%)',
       xlim=[-0.2, 0.2], xticks=[-0.2, -0.1, 0, 0.1, 0.2], xticklabels=[-0.2, -0.1, 0, 0.1, 0.2],
       ylim=[0, 80], yticks=np.arange(90, step=20))

for _, row in plot_data.iterrows():
    axs[1].scatter(row['perc_mod'], row['perc_glut'], color=row['color'], s=20, zorder=1)
axs[1].set(xlabel='5-HT modulated neurons (%)',
       xlim=[10, 60], xticks=[10, 20, 30, 40, 50, 60])

axs[1].legend(labels=plot_data['region'], bbox_to_anchor=(1.05, 1.1))

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'dr_projection.pdf'))

