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

N_STATES_SELECT = 'global'
RANDOM_TIMES = 'jitter'

# Plotting
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State')

# Load in data
corr_anes_df = pd.read_csv(join(save_path, 'state_correlation_anesthesia.csv'))
corr_passive_df = pd.read_csv(join(save_path, f'state_correlation_{RANDOM_TIMES}_passive.csv'))
#corr_passive_df = pd.read_csv(join(save_path, f'state_correlation_{RANDOM_TIMES}_passive.csv'))
corr_task_df = pd.read_csv(join(save_path, f'state_correlation_{N_STATES_SELECT}_task.csv'))
corr_task_df = corr_task_df[corr_task_df['opto'] != 2]  # exclude probe trials

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    corr_passive_df.loc[corr_passive_df['subject'] == nickname,
                        'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    corr_task_df.loc[corr_task_df['subject'] == nickname,
                     'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    corr_anes_df.loc[corr_anes_df['subject'] == nickname,
                     'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# %% Anesthesia
f, ax1 = plt.subplots(1, 1, figsize=(1.4, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0.5), 1, 0.3, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=corr_anes_df[corr_anes_df['sert-cre'] == 1], x='time', y='perc',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
ax1.set(xlabel='Time from stimulation (s)', ylim=[0.5, 0.7],
        xticks=[-1, 0, 1, 2, 3, 4], yticks=[0.5, 0.7], yticklabels=[0.5, 0.7])
ax1.set_ylabel('State synchrony (%)', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(1, 1), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'state_synchrony_anesthesia.pdf'), dpi=600)

"""
# %% Plot region pairs seperately
g = sns.FacetGrid(corr_anes_df[corr_anes_df['sert-cre'] == 1], col='region_pair', col_wrap=5, height=2,
                  ylim=(0, 0.5))
g.map(sns.lineplot, 'time', 'r_mean', color='k', errorbar='se')
plt.savefig(
    join(fig_path, 'state_correlation_all_region_pair_anesthesia.jpg'), dpi=600)
"""

# %% Passive mean
f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -0.002), 1, 0.1, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=corr_passive_df[corr_passive_df['sert-cre'] == 1], x='time', y='r_mean',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
ax1.set(xlabel='Time from stimulation (s)', ylim=[-0.002, 0.01],
        xticks=[-1, 0, 1, 2, 3, 4], yticks=[0, 0.01], yticklabels=[0, 0.01])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State correlation (r)', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(1, 0.7), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'state_correlation_{RANDOM_TIMES}_passive_mean.jpg'), dpi=600)
plt.savefig(join(fig_path, f'state_correlation_{RANDOM_TIMES}_passive_mean.pdf'))

# %% Passive abs
f, ax1 = plt.subplots(1, 1, figsize=(2.2, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -0.002), 1, 0.1, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=corr_passive_df[corr_passive_df['sert-cre'] == 1], x='time', y='r_abs',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
#ax1.set(xlabel='Time from stimulation start (s)', ylim=[-0.002, 0.01],
#        xticks=[-1, 0, 1, 2, 3, 4], yticks=[0, 0.01], yticklabels=[0, 0.01])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State correlation (r)', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(1, 0.7), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'state_correlation_{RANDOM_TIMES}_passive_abs.jpg'), dpi=600)
plt.savefig(join(fig_path, f'state_correlation_{RANDOM_TIMES}_passive_abs.pdf'))

# %% Passive max
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 0.6, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=corr_passive_df[corr_passive_df['sert-cre'] == 1], x='time', y='r_max',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
ax1.set(xlabel='Time from stimulation start (s)', ylim=[0.1, 0.6],
        xticks=[-1, 0, 1, 2, 3, 4], yticks=[0.1, 0.6], yticklabels=[0.1, 0.6])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State correlation (r)', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.5), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'state_correlation_{RANDOM_TIMES}_passive_max.jpg'), dpi=600)

# %% Passive min
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -1), 1, 2, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=corr_passive_df[corr_passive_df['sert-cre'] == 1], x='time', y='r_min',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
ax1.set(xlabel='Time from stimulation start (s)', ylim=[-0.5, -0.2],
        xticks=[-1, 0, 1, 2, 3, 4], yticks=[-0.5, -0.2])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State correlation (r)', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.8), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, f'state_correlation_{RANDOM_TIMES}_passive_min.jpg'), dpi=600)

"""
# %% Plot region pairs seperately
g = sns.FacetGrid(corr_passive_df[corr_passive_df['sert-cre'] == 1], col='region_pair', col_wrap=5, height=2,
                  ylim=(-0.02, 0.02))
g.map(sns.lineplot, 'time', 'r_mean', color='k', errorbar='se')
plt.savefig(
    join(fig_path, 'state_correlation_all_region_pair_passive_f{N_STATES_SELECT}.jpg'), dpi=600)
"""

# %% Passive permut
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 0.5, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=corr_passive_df[corr_passive_df['sert-cre'] == 1], x='time', y='r_max',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
ax1.set(xlabel='Time from stimulation start (s)', ylim=[0.1, 0.5],
        xticks=[-1, 0, 1, 2, 3, 4], yticks=[0.1, 0.5], yticklabels=[0.1, 0.5])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State correlation (r)', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.7), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'state_correlation_{RANDOM_TIMES}_passive_permut.jpg'), dpi=600)


# %% Task mean
f, ax1 = plt.subplots(1, 1, figsize=(1.4, 1.75), dpi=dpi)
#ax1.add_patch(Rectangle((0, -0.1), 1, 0.2, color='royalblue', alpha=0.25, lw=0))
ax1.plot([0, 0], [-0.2, 0.2], ls='--', color='grey')
sns.lineplot(data=corr_task_df[corr_task_df['sert-cre'] == 1],
             x='time', y='r_mean', errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
ax1.set(xlabel='Time from trial start (s)', ylim=[-0.002, 0.02],
        xticks=[-1, 0, 1, 2, 3, 4], yticks=[0, 0.02], yticklabels=[0, 0.02])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State correlation (r)', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(0.4, 0.9), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'state_correlation_{RANDOM_TIMES}_task_mean.jpg'), dpi=600)
plt.savefig(join(fig_path, f'state_correlation_{RANDOM_TIMES}_task_mean.pdf'))


# %% Plot all region pairs together
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
# ax1.add_patch(Rectangle((0, -0.1), 1, 0.2, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=corr_task_df[corr_task_df['sert-cre'] == 1],
             x='time', y='r_max', errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
ax1.set(xlabel='Time from trial start (s)', ylim=[0.4, 0.51],
        xticks=[-1, 0, 1, 2, 3, 4], yticks=[0.4, 0.5], yticklabels=[0.4, 0.5])
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.9), prop={'size': 5})
ax1.set_ylabel('State correlation (r)', labelpad=-5)

new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'state_correlation_{RANDOM_TIMES}_task_max.jpg'), dpi=600)

"""
# %% Plot region pairs seperately
g = sns.FacetGrid(corr_task_df[corr_task_df['sert-cre'] == 1], col='region_pair', col_wrap=5, height=2,
                  ylim=(-0.02, 0.04), hue='opto', hue_order=[0, 1],
                  palette=[colors['no-stim'], colors['stim']])
g.map(sns.lineplot, 'time', 'r_mean', errorbar='se')
plt.savefig(
    join(fig_path, 'state_correlation_all_region_pair_task_f{N_STATES_SELECT}.jpg'), dpi=600)
"""
