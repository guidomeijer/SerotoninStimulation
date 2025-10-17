# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:21:10 2025

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import pandas as pd
from scipy.stats import pearsonr
from os.path import join, realpath, dirname, split
from stim_functions import paths, load_subjects, figure_style
from iblatlas.atlas import BrainRegions
br = BrainRegions()

MIN_NEURONS_PER_MOUSE = 3

# Get paths
f_path, data_path = path_dict = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in fmri data
fmri_data = pd.read_csv(join(data_path, 'Hamada', 'tphskk_blueval.csv'), header=None)
fmri_data['roi'] = pd.read_csv(join(data_path, 'Hamada', 'roi.csv'), header=None)
fmri_data = fmri_data.rename(columns={0: 'beta'})

# Load in Neuropixel data
ephys_data = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    ephys_data.loc[ephys_data['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
ephys_data = ephys_data[(ephys_data['sert-cre'] == 1) & (ephys_data['modulated'] == 1)]

# Create mapping of to fmri regions
acronym_map = {
    'AUD': 'A1', 'ACA': 'ACC', 'BST': 'BST', 'CA1': 'CA1', 'CP': 'CPu',
    'DG': 'DG', 'DR': 'DRN', 'FRP': 'FP', 'GPe': 'GPe', 'GPi': 'GPi',
    'LH': 'LHb', 'MOp': 'M1', 'MOs': 'M2', 'ACB': 'NAc', 'ORB': 'OFC',
    'VISa': 'PPA', 'VISam': 'PPA', 'VISrl': 'PPA', 'RT': 'RT', 'SSp': 'S1',
    'SNr': 'SNr', 'SSs': 'SS', 'TEa': 'TAA', 'VISp': 'V1', 'PALv': 'VP',
    'VTA': 'VTA', 'RSP': 'dRSC', 'PL': 'mPFC', 'ILA': 'mPFC'
}
regions = np.array(['root'] * ephys_data.shape[0], dtype=object)
for allen_acronym, custom_region in acronym_map.items():
    descendant_acronyms = br.descendants(br.acronym2id(allen_acronym))['acronym']
    mask = np.isin(ephys_data['allen_acronym'], descendant_acronyms)
    regions[mask] = custom_region
ephys_data['fmri_region'] = regions

# Calculate summaries
per_mouse_df = ephys_data[['fmri_region', 'subject']].groupby(['fmri_region', 'subject']).sum()
per_mouse_df['mod_index'] = ephys_data[['fmri_region', 'subject', 'mod_index']].groupby(
    ['fmri_region', 'subject']).mean()['mod_index']
per_mouse_df['n_neurons'] = ephys_data[['fmri_region', 'subject']].groupby(['fmri_region', 'subject']).size()
per_mouse_df = per_mouse_df[per_mouse_df['n_neurons'] >= MIN_NEURONS_PER_MOUSE]
per_mouse_df = per_mouse_df.reset_index()

ephys_summary = per_mouse_df.groupby('fmri_region')['mod_index'].agg(['mean', 'sem']).rename(
    columns={'mean': 'mod_index_mean', 'sem': 'mod_index_sem'})
fmri_summary = fmri_data.groupby('roi')['beta'].agg(['mean', 'sem']).rename(
    columns={'mean': 'beta_mean', 'sem': 'beta_sem'})

# %% Plot
colors, dpi = figure_style()
plot_data = pd.merge(ephys_summary, fmri_summary, left_index=True, right_index=True)
plot_data['color'] = sns.color_palette('tab20')[:plot_data.shape[0]]

r, p = pearsonr(plot_data['mod_index_mean'], plot_data['beta_mean'])
slope, intercept = np.polyfit(plot_data['mod_index_mean'], plot_data['beta_mean'], 1)
x_fit = np.linspace(-0.3, 0.3, 100)
y_fit = slope * x_fit + intercept

fig, ax = plt.subplots(figsize=(3.1, 2.3), dpi=dpi)
ax.plot(x_fit, y_fit, color='k', lw=1, label='_nolegend_')
ax.plot([0, 0], [-0.5, 2], color='grey', lw=1, ls='--', label='_nolegend_',
        zorder=0)
ax.plot([-0.4, 0.4], [0, 0], color='grey', lw=1, ls='--', label='_nolegend_',
        zorder=0)
ax.errorbar(
    x=plot_data['mod_index_mean'],
    y=plot_data['beta_mean'],
    xerr=plot_data['mod_index_sem'],
    yerr=plot_data['beta_sem'],
    fmt='',           
    linestyle='none',  
    ecolor=[0.7, 0.7, 0.7],
    capsize=2,         
    capthick=1,
    zorder=0
)
for _, row in plot_data.iterrows():
    ax.scatter(row['mod_index_mean'], row['beta_mean'], color=row['color'], s=20, marker='s', zorder=1)
ax.text(0.2, 1.7, '**', fontsize=12, ha='center', va='center')
ax.set(xlabel='5-HT modulation Ephys (modulation index)',
       ylabel='5-HT modulation fMRI (beta value)',
       xlim=[-0.4, 0.3], xticks=[-0.4, -0.2, 0, 0.2, 0.4], xticklabels=[-0.4, -0.2, 0, 0.2, 0.4],
       ylim=[-0.5, 2])
ax.legend(labels=plot_data.index, bbox_to_anchor=(1.05, 1.1))
plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'ephys_vs_fmri.pdf'))