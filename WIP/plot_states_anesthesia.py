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
from stim_functions import figure_style, paths, load_subjects
from os.path import join

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State')

# Load in data
state_trans_df = pd.read_csv(join(save_path, 'state_trans_anesthesia.csv'))
p_state_df = pd.read_csv(join(save_path, 'updown_states_anesthesia.csv'))

# Average over mice first
state_trans_df = state_trans_df.groupby(['subject', 'time', 'region']).mean(numeric_only=True).reset_index()
p_state_df = p_state_df.groupby(['subject', 'time', 'region']).mean(numeric_only=True).reset_index()

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    state_trans_df.loc[state_trans_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    p_state_df.loc[p_state_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
state_trans_df = state_trans_df[state_trans_df['sert-cre'] == 1]
p_state_df = p_state_df[p_state_df['sert-cre'] == 1]

# Get up state probability
p_state_df['p_up'] = 1 - p_state_df['p_down']

# %% Plot
colors, dpi = figure_style()
f, axs = plt.subplots(1, 7, figsize=(5.5, 1.75), dpi=dpi, sharey=True)
#axs = np.concatenate(axs)
regions = ['Amygdala', 'Frontal cortex', 'Striatum', 'Sensory cortex', 'Thalamus',
           'Hippocampus', 'Midbrain']
for i, region in enumerate(regions):

    axs[i].add_patch(Rectangle((0, 0), 1, 1, color='royalblue', alpha=0.25, lw=0))
    axs[i].plot([-1, 4], [0.5, 0.5], ls='--', color='grey')
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='p_down',
                 color=colors['suppressed'], errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='p_up',
                 color=colors['enhanced'], errorbar='se', ax=axs[i])
    axs[i].set(xlabel='Time (s)', title=region, ylim=[-0.02, 1],
               yticks=[0, 0.5, 1], yticklabels=[0, 50, 100])
    if i == 0:
        axs[i].set(ylabel='Down state probability (%)')
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.01, -0.01], color='k', lw=0.5)
        axs[i].text(1, -0.03, '2s', ha='center', va='top')
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
    
plt.subplots_adjust(left=0.08, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
#plt.tight_layout(h_pad=-10, w_pad=1.08)
plt.savefig(join(fig_path, 'p_state_anesthesia.jpg'), dpi=600)

# %% Plot
colors, dpi = figure_style()
f, axs = plt.subplots(1, 7, figsize=(5.5, 1.75), dpi=dpi, sharey=True)
#axs = np.concatenate(axs)
regions = ['Amygdala', 'Frontal cortex', 'Striatum', 'Sensory cortex', 'Thalamus',
           'Hippocampus', 'Midbrain']
for i, region in enumerate(regions):

    axs[i].add_patch(Rectangle((0, 0), 1, 1, color='royalblue', alpha=0.25, lw=0))
    axs[i].plot([-1, 4], [0.5, 0.5], ls='--', color='grey')
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='p_down',
                 color='k', errorbar='se', err_kws={'lw': 0}, ax=axs[i])
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
    
plt.subplots_adjust(left=0.08, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
#plt.tight_layout(h_pad=-10, w_pad=1.08)
plt.savefig(join(fig_path, 'p_down_state_anesthesia.jpg'), dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(7, 3.5), dpi=dpi)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(state_trans_df['region'])):

    axs[i].add_patch(Rectangle((0, -4), 1, 5, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_state_change',
                 color='k', errorbar='se', ax=axs[i])
    axs[i].set(ylabel='P(state change)', xlabel='Time (s)', title=region, ylim=[0, 0.2],
               yticks=[0, 0.2], xticks=[-1, 0, 1, 2, 3, 4])
axs[-1].axis('off')
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'p_state_change_anesthesia.jpg'), dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(7, 3.5), dpi=dpi)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(state_trans_df['region'])):

    axs[i].add_patch(Rectangle((0, -4), 1, 5, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_down_state_change',
                 color=colors['suppressed'], errorbar='se', ax=axs[i], label='Down', err_kws={'lw': 0})
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_up_state_change',
                 color=colors['enhanced'], errorbar='se', ax=axs[i], label='Up', err_kws={'lw': 0})
    axs[i].set(ylabel='P(state change)', xlabel='Time (s)', title=region, ylim=[0, 0.2],
               yticks=[0, 0.25], xticks=[-1, 0, 1, 2, 3, 4])
    if i > 0:
        axs[i].get_legend().remove()
axs[-1].axis('off')
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'p_state_change_anesthesia.jpg'), dpi=600)

