#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:00:47 2023
By: Guido Meijer
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
import pandas as pd
from matplotlib.patches import Rectangle
from stim_functions import figure_style, paths, load_subjects

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
state_trans_df = pd.read_csv(join(save_path, 'state_trans_anesthesia.csv'))
p_state_df = pd.read_csv(join(save_path, 'updown_states_anesthesia.csv'))
state_trans_null_df = pd.read_csv(join(save_path, 'state_trans_null_anesthesia.csv'))
p_state_null_df = pd.read_csv(join(save_path, 'updown_states_null_anesthesia.csv'))

# Average over mice first
state_trans_df = state_trans_df.groupby(['subject', 'time', 'region']).mean(numeric_only=True).reset_index()
p_state_df = p_state_df.groupby(['subject', 'time', 'region']).mean(numeric_only=True).reset_index()
state_trans_null_df = state_trans_null_df.groupby(['subject', 'time', 'region']).mean(numeric_only=True).reset_index()
p_state_null_df = p_state_null_df.groupby(['subject', 'time', 'region']).mean(numeric_only=True).reset_index()

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    state_trans_df.loc[state_trans_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    p_state_df.loc[p_state_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    state_trans_null_df.loc[state_trans_null_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    p_state_null_df.loc[p_state_null_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
state_trans_df = state_trans_df[state_trans_df['sert-cre'] == 1]
p_state_df = p_state_df[p_state_df['sert-cre'] == 1]
state_trans_null_df = state_trans_null_df[state_trans_null_df['sert-cre'] == 1]
p_state_null_df = p_state_null_df[p_state_null_df['sert-cre'] == 1]


# %% Plot
colors, dpi = figure_style()
f, axs = plt.subplots(1, 7, figsize=(7, 1.75), dpi=dpi, sharey=True)
#axs = np.concatenate(axs)
regions = ['Frontal cortex', 'Amygdala', 'Striatum', 'Sensory cortex', 'Hippocampus',
           'Thalamus', 'Midbrain']
for i, region in enumerate(regions):

    axs[i].add_patch(Rectangle((0, 0), 1, 1, color='royalblue', alpha=0.25, lw=0))
    axs[i].plot([-1, 4], [0.5, 0.5], ls='--', color='grey')
    sns.lineplot(data=p_state_null_df[p_state_null_df['region'] == region], x='time', y='p_down',
                 color=colors['grey'], errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='p_down',
                 color=colors['down-state'], errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    axs[i].set(xlabel='Time (s)', title=region, ylim=[0.3, 1],
               yticks=[0.25, 0.5, 0.75, 1], yticklabels=[25, 50, 75, 100])
    if i == 0:
        axs[i].set_ylabel('Down state probability (%)', labelpad=1)
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.01, -0.01], color='k', lw=0.5)
        axs[i].text(1, -0.03, '2s', ha='center', va='top')
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
    
plt.subplots_adjust(left=0.06, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
#plt.tight_layout(h_pad=-10, w_pad=1.08)
plt.savefig(join(fig_path, 'p_down_state_anesthesia.pdf'))

# %%
f, axs = plt.subplots(1, 7, figsize=(7, 1.75), dpi=dpi)
for i, region in enumerate(regions):
    axs[i].add_patch(Rectangle((0, 0), 1, 1, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_down_state_change',
                 color=colors['suppressed'], errorbar='se', ax=axs[i], label='To down', err_kws={'lw': 0})
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_up_state_change',
                 color=colors['enhanced'], errorbar='se', ax=axs[i], label='To up', err_kws={'lw': 0})
    axs[i].set(ylabel='P(state change)', xlabel='Time (s)', title=region, ylim=[-0.012, 0.2],
               yticks=[0, 0.25], yticklabels=[0, 25], xticks=[-1, 0, 1, 2, 3, 4])
    if i == 0:
        axs[i].set_ylabel('Probability of state change (%)', labelpad=5)
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.01, -0.01], color='k', lw=0.5)
        axs[i].text(1, -0.02, '2s', ha='center', va='top')
        axs[i].legend(prop={'size': 6})
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
        axs[i].get_legend().set_visible(False)
plt.subplots_adjust(left=0.06, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
plt.savefig(join(fig_path, 'p_updown_state_change_anesthesia.pdf'))
