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
p_state_df = pd.read_csv(join(save_path, 'updown_states_anesthesia.csv'))

# Average over mice first
p_state_df = p_state_df.groupby(['subject', 'time', 'region', 'opto']).mean(numeric_only=True).reset_index()

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    p_state_df.loc[p_state_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
p_state_df = p_state_df[p_state_df['sert-cre'] == 1]

# %% Plot
colors, dpi = figure_style()
f, axs = plt.subplots(1, 7, figsize=(5.2, 1.75), dpi=dpi, sharey=True)
#axs = np.concatenate(axs)
regions = ['Frontal cortex', 'Striatum', 'Amygdala', 'Sensory cortex', 'Hippocampus',
           'Thalamus', 'Midbrain']
titles = ['Frontal cortex', 'Striatum', 'Amygdala', 'Sensory ctx', 'Hippocampus',
           'Thalamus', 'Midbrain']
for i, region in enumerate(regions):

    axs[i].add_patch(Rectangle((0, -0.25), 1, 0.75, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='p_down_bl', hue='opto',
                 hue_order=[1, 0], palette=[colors['stim'], colors['no-stim']], errorbar='se',
                 err_kws={'lw': 0}, ax=axs[i])
    axs[i].set(xlabel='Time (s)', title=titles[i], ylim=[-0.1, 0.3],
               yticks=[-0.1, 0, 0.3], yticklabels=[-10, 0, 30])
    n_sub = len(np.unique(p_state_df.loc[p_state_df['region'] == region, 'subject']))
    if i == 0:
        axs[i].text(2, -0.22, f'n={n_sub} mice', ha='center', va='center')
        axs[i].set_ylabel(u'Δ down state probability (%)', labelpad=0)
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.26, -0.26], color='k', lw=0.5)
        axs[i].text(1, -0.275, '2s', ha='center', va='top')
        axs[i].get_legend().set_visible(False)
    elif i == len(regions)-1:
        axs[i].text(3, -0.22, f'n={n_sub}', ha='center', va='center')
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
        leg = axs[i].legend(prop={'size': 6}, frameon=True)
        leg.get_frame().set_linewidth(0.0)
    else:
        axs[i].text(3, -0.22, f'n={n_sub}', ha='center', va='center')
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
        axs[i].get_legend().set_visible(False)
    
plt.subplots_adjust(left=0.11, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
#plt.tight_layout(h_pad=-10, w_pad=1.08)
plt.savefig(join(fig_path, 'p_down_state_anesthesia.pdf'))

# %%
f, axs = plt.subplots(1, 7, figsize=(5.2, 1.75), dpi=dpi, sharey=True)
#axs = np.concatenate(axs)
regions = ['Frontal cortex', 'Amygdala', 'Striatum', 'Sensory cortex', 'Hippocampus',
           'Thalamus', 'Midbrain']
titles = ['Frontal cortex', 'Amygdala', 'Striatum', 'Sensory ctx', 'Hippocampus',
           'Thalamus', 'Midbrain']
for i, region in enumerate(regions):

    axs[i].add_patch(Rectangle((0, -0.25), 1, 0.75, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='p_down', hue='opto',
                 hue_order=[1, 0], palette=[colors['stim'], colors['no-stim']], errorbar='se',
                 err_kws={'lw': 0}, ax=axs[i])
    #axs[i].set(xlabel='Time (s)', title=titles[i], ylim=[-0.1, 0.3],
    #           yticks=[-0.1, 0, 0.3], yticklabels=[-10, 0, 30])
    n_sub = len(np.unique(p_state_df.loc[p_state_df['region'] == region, 'subject']))
    if i == 0:
        axs[i].text(2, -0.22, f'n={n_sub} mice', ha='center', va='center')
        axs[i].set_ylabel(u'Δ down state probability (%)', labelpad=0)
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.26, -0.26], color='k', lw=0.5)
        axs[i].text(1, -0.275, '2s', ha='center', va='top')
        axs[i].get_legend().set_visible(False)
    elif i == len(regions)-1:
        axs[i].text(3, -0.22, f'n={n_sub}', ha='center', va='center')
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
        leg = axs[i].legend(prop={'size': 6}, frameon=True)
        leg.get_frame().set_linewidth(0.0)
    else:
        axs[i].text(3, -0.22, f'n={n_sub}', ha='center', va='center')
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
        axs[i].get_legend().set_visible(False)
    
plt.subplots_adjust(left=0.11, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
#plt.tight_layout(h_pad=-10, w_pad=1.08)
plt.savefig(join(fig_path, 'p_down_state_anesthesia_noblsub.pdf'))

"""
# %%
f, axs = plt.subplots(1, 7, figsize=(5, 1.75), dpi=dpi)
for i, region in enumerate(regions):
    axs[i].add_patch(Rectangle((0, 0), 1, 1, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_down_state_change',
                 color=colors['down-state'], errorbar='se', ax=axs[i], label='To down', err_kws={'lw': 0})
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='p_up_state_change',
                 color=colors['up-state'], errorbar='se', ax=axs[i], label='To up', err_kws={'lw': 0})
    axs[i].set(ylabel='P(state change)', xlabel='Time (s)', title=titles[i], ylim=[-0.012, 0.2],
               yticks=[0, 0.25], yticklabels=[0, 25], xticks=[-1, 0, 1, 2, 3, 4])
    if i == 0:
        axs[i].set_ylabel('Probability of state change (%)', labelpad=1)
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.01, -0.01], color='k', lw=0.5)
        axs[i].text(1, -0.02, '2s', ha='center', va='top')
        axs[i].get_legend().set_visible(False)
    elif i == len(regions)-1:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
        leg = axs[i].legend(prop={'size': 6}, frameon=True)
        leg.get_frame().set_linewidth(0.0)
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
        axs[i].get_legend().set_visible(False)

plt.subplots_adjust(left=0.1, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
plt.savefig(join(fig_path, 'p_updown_state_change_anesthesia.pdf'))
"""