# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:56:02 2023

@author: Guido
"""

import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
from os.path import join, split
import matplotlib.pyplot as plt
from stim_functions import paths, figure_style, load_subjects

N_STATES_SELECT = 'global'  # region or global

# Plotting
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State')

# Load in data
corr_df = pd.read_csv(join(save_path, f'state_correlation_{N_STATES_SELECT}.csv'))
corr_task_df = pd.read_csv(join(save_path, f'state_correlation_{N_STATES_SELECT}_task.csv'))
corr_task_df.loc[corr_task_df['opto'] == 2] = 1

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    corr_df.loc[corr_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    corr_task_df.loc[corr_task_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# %% Plot all region pairs together
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -0.005), 1, 0.02, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=corr_df, x='time', y='r_mean', errorbar='se', hue='sert-cre', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['wt'], colors['sert']])
ax1.set(xlabel='Time from stimulation start (s)', ylim=[-0.001, 0.0075],
        xticks=[-1, 0, 1, 2, 3, 4], yticks=[0, 0.007], yticklabels=[0, 0.007])
        #ylim=[-0.001, 0.005], yticks=[0, 0.005],
        #yticklabels=[0, 0.005])
ax1.set_ylabel('State correlation (r)', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(0.7, 1), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['WT', 'SERT']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'state_correlation_passive_f{N_STATES_SELECT}.jpg'), dpi=600)

# %% Plot region pairs seperately
g = sns.FacetGrid(corr_df[corr_df['sert-cre'] == 1], col='region_pair', col_wrap=5, height=2,
                  ylim=(-0.02, 0.02))
g.map(sns.lineplot, 'time', 'r_mean', color='k', errorbar='se')
plt.savefig(join(fig_path, 'state_correlation_all_region_pair_passive_f{N_STATES_SELECT}.jpg'), dpi=600)

# %% Plot all region pairs together
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -0.1), 1, 0.2, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=corr_task_df[corr_task_df['sert-cre'] == 1],
             x='time', y='r_mean', errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
ax1.set(xlabel='Time from stimulation start (s)', ylim=[-0.002, 0.02],
        xticks=[-1, 0, 1, 2, 3, 4], yticks=[0, 0.02], yticklabels=[0, 0.02])
        #ylim=[-0.001, 0.005], yticks=[0, 0.005],
        #yticklabels=[0, 0.005])
ax1.set_ylabel('State correlation (r)', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.9), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'state_correlation_task_f{N_STATES_SELECT}.jpg'), dpi=600)

# %% Plot all region pairs together
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -0.1), 1, 0.2, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=corr_task_df[corr_task_df['sert-cre'] == 1],
             x='time', y='r_mean', errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
ax1.set(xlabel='Time from stimulation start (s)', ylim=[-0.002, 0.02],
        xticks=[-1, 0, 1, 2, 3, 4], yticks=[0, 0.02], yticklabels=[0, 0.02])
        #ylim=[-0.001, 0.005], yticks=[0, 0.005],
        #yticklabels=[0, 0.005])
ax1.set_ylabel('State correlation (r)', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.9), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'state_correlation_task_f{N_STATES_SELECT}.jpg'), dpi=600)

# %% Plot region pairs seperately
g = sns.FacetGrid(corr_task_df[corr_task_df['sert-cre'] == 1], col='region_pair', col_wrap=5, height=2,
                  ylim=(-0.02, 0.04), hue='opto', hue_order=[0, 1],
                  palette=[colors['no-stim'], colors['stim']])
g.map(sns.lineplot, 'time', 'r_mean', errorbar='se')
plt.savefig(join(fig_path, 'state_correlation_all_region_pair_task_f{N_STATES_SELECT}.jpg'), dpi=600)


