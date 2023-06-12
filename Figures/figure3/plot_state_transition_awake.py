#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:00:47 2023
By: Guido Meijer
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
from stim_functions import figure_style, paths, load_subjects
from os.path import join, realpath, dirname, split

BIN_SIZE = 100  # ms
NEURONS = 'all'  # non-sig, sig or all
SERT_CRE = 1
REGION_ORDER = ['Frontal cortex', 'Hippocampus', 'Thalamus', 'Amygdala', 'Sensory cortex',
                'Midbrain', 'Striatum']

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
state_trans_df = pd.read_csv(join(save_path, f'state_trans_{BIN_SIZE}msbins_{NEURONS}.csv'))
state_trans_null_df = pd.read_csv(join(save_path, f'state_trans_null_{BIN_SIZE}msbins_{NEURONS}.csv'))

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    state_trans_df.loc[state_trans_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    state_trans_null_df.loc[state_trans_null_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
state_trans_df = state_trans_df[state_trans_df['sert-cre'] == SERT_CRE]
state_trans_null_df = state_trans_null_df[state_trans_null_df['sert-cre'] == SERT_CRE]

# %%
colors, dpi = figure_style()
f, axs = plt.subplots(1, 7, figsize=(5.5, 1.75), dpi=dpi, sharey=True, sharex=True)
for i, region in enumerate(REGION_ORDER):
    axs[i].add_patch(Rectangle((0, -0.05), 1, 0.1, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=state_trans_null_df[state_trans_null_df['region'] == region], x='time', y='p_trans_bl',
                 color=colors['grey'], errorbar='se', ax=axs[i], err_kws={'lw': 0})
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_trans_bl',
                 color=colors['stim'], errorbar='se', ax=axs[i], err_kws={'lw': 0})
    axs[i].set(title=region, ylim=[-0.052, 0.055], yticks=[-0.05, 0, 0.05], yticklabels=[-5, 0, 5])
    if i == 0:
        axs[i].set(ylabel='State transition probability (%)', xticks=[0, 2])
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].text(1, -0.055, '2s', ha='center', va='top')
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
            
plt.subplots_adjust(left=0.08, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
#plt.tight_layout(rect=(0.05, 0.05, 1, 1))
sns.despine(trim=True)
plt.savefig(join(fig_path, 'state_change_rate_baseline.pdf'))

# %%
f, axs = plt.subplots(1, 7, figsize=(5.5, 1.75), dpi=dpi, sharey=True, sharex=True)
for i, region in enumerate(REGION_ORDER):
    axs[i].add_patch(Rectangle((0, 0), 1, 0.3, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=state_trans_null_df[state_trans_null_df['region'] == region], x='time', y='p_trans',
                 color=colors['grey'], errorbar='se', ax=axs[i], err_kws={'lw': 0})
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_trans',
                 color=colors['stim'], errorbar='se', ax=axs[i], err_kws={'lw': 0})
    axs[i].set(title=region)
    if i == 0:
        axs[i].set(ylabel='State transition probability (%)', xticks=[0, 2])
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].text(1, -0.02, '2s', ha='center', va='top')
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
            
plt.subplots_adjust(left=0.08, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
#plt.tight_layout(rect=(0.05, 0.05, 1, 1))
sns.despine(trim=True)
plt.savefig(join(fig_path, 'state_change_rate.pdf'))


