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
#coact_mean_df = pd.read_csv(join(save_path, f'state_coactivation_mean_{RANDOM_TIMES}_passive.csv'))
#coact_df = pd.read_csv(join(save_path, f'state_coactivation_{RANDOM_TIMES}_passive.csv'))
coact_mean_df = pd.read_csv(join(save_path, f'state_coactivation_mean_task.csv'))
coact_df = pd.read_csv(join(save_path, f'state_coactivation_task.csv'))

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    coact_mean_df.loc[coact_mean_df['subject'] == nickname,
                      'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    coact_df.loc[coact_df['subject'] == nickname,
                 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
   


# %% Passive mean
f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
#ax1.add_patch(Rectangle((0, -0.002), 1, 0.1, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=coact_mean_df[coact_mean_df['sert-cre'] == 1], x='time', y='r_mean',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
#ax1.set(xlabel='Time from stimulation (s)', ylim=[0.058, 0.07],
#        xticks=[-1, 0, 1, 2, 3], yticks=[0.06, 0.07], yticklabels=[0.06, 0.07])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State coactivation', labelpad=0)
g = ax1.legend(title='', bbox_to_anchor=(0.95, 0.7), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'state_coactivation_{RANDOM_TIMES}_passive_mean.jpg'), dpi=600)
plt.savefig(join(fig_path, f'state_coactivation_{RANDOM_TIMES}_passive_mean.pdf'))

# %% Passive mean normalized
f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -0.002), 1, 0.1, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=coact_mean_df[coact_mean_df['sert-cre'] == 1], x='time', y='coact_norm',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
#ax1.set(xlabel='Time from stimulation (s)', ylim=[0.058, 0.07],
#        xticks=[-1, 0, 1, 2, 3], yticks=[0.06, 0.07], yticklabels=[0.06, 0.07])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State coactivation', labelpad=0)
g = ax1.legend(title='', bbox_to_anchor=(0.95, 0.7), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'state_coactivation_{RANDOM_TIMES}_passive_mean_norm.jpg'), dpi=600)
plt.savefig(join(fig_path, f'state_coactivation_{RANDOM_TIMES}_passive_mean_norm.pdf'))

# %% Passive max
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
#ax1.add_patch(Rectangle((0, 0), 1, 0.6, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=coact_mean_df[coact_mean_df['sert-cre'] == 1], x='time', y='r_max',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
#ax1.set(xlabel='Time from stimulation start (s)', ylim=[0.3, 0.5],
#        xticks=[-1, 0, 1, 2, 3], yticks=[0.3, 0.5], yticklabels=[0.3, 0.5])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State coactivation', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.3), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'state_coactivation_{RANDOM_TIMES}_passive_max.jpg'), dpi=600)

# %% Passive over time
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
#ax1.add_patch(Rectangle((0, 0), 1, 0.6, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=coact_df[coact_df['sert-cre'] == 1], x='time', y='coact',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
#ax1.set(xlabel='Time from stimulation start (s)', ylim=[0.3, 0.5],
#        xticks=[-1, 0, 1, 2, 3], yticks=[0.3, 0.5], yticklabels=[0.3, 0.5])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State coactivation', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.3), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'state_coactivation_{RANDOM_TIMES}_passive_time.jpg'), dpi=600)

# %% Passive mean frontal midbrain
f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
#ax1.add_patch(Rectangle((0, -0.002), 1, 0.1, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=coact_mean_df[(coact_mean_df['sert-cre'] == 1) & (coact_mean_df['region_pair'] == 'Frontal cortex-Midbrain')],
             x='time', y='r_mean',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
#ax1.set(xlabel='Time from stimulation (s)', ylim=[0.058, 0.07],
#        xticks=[-1, 0, 1, 2, 3], yticks=[0.05, 0.08], yticklabels=[0.06, 0.07])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State coactivation', labelpad=0)
g = ax1.legend(title='', bbox_to_anchor=(0.95, 0.7), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'state_coactivation_{RANDOM_TIMES}_passive_mean_frontal-midbrain.jpg'), dpi=600)
plt.savefig(join(fig_path, f'state_coactivation_{RANDOM_TIMES}_passive_mean_frontal-midbrain.pdf'))

# %% Plot region pairs seperately
g = sns.FacetGrid(coact_mean_df[coact_mean_df['sert-cre'] == 1], col='region_pair', col_wrap=5, height=2,
                  ylim=(0.04, 0.08), hue='opto', hue_order=[0, 1],
                  palette=[colors['no-stim'], colors['stim']])
g.map(sns.lineplot, 'time', 'r_mean', errorbar='se')
#plt.savefig(
#    join(fig_path, 'state_correlation_all_region_pair_task_f{N_STATES_SELECT}.jpg'), dpi=600)

