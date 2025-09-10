#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os import path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import load_trials, paths, figure_style, load_subjects

# Settings
subjects = load_subjects()

# Paths
f_path, save_path = paths()
fig_path = path.join(f_path, path.split(path.dirname(path.realpath(__file__)))[-1])

# Load in all trials 
all_trials = pd.read_csv(path.join(save_path, 'all_trials.csv'))

switch_df = pd.DataFrame()
for i, nickname in enumerate(np.unique(all_trials['subject'])):
    print(f'Subject {nickname} ({i} of {np.unique(all_trials["subject"]).shape[0]})')

    # Get trials DataFrame
    trials = all_trials[all_trials['subject'] == nickname].copy()
    trials = trials.drop(columns=['subject'])
    trials = trials.reset_index(drop=True)

    # Make array of after block switch trials
    trials['block_switch'] = np.zeros(trials.shape[0])
    trial_blocks = (trials['probabilityLeft'] == 0.2).astype(int)
    block_trans = np.append(np.array(np.where(np.diff(trial_blocks) != 0)) + 1, [trial_blocks.shape[0]])

    for t, trans in enumerate(block_trans[:-1]):
        r_choice = trials.loc[(trials.index.values < block_trans[t+1])
                              & (trials.index.values >= block_trans[t]), 'right_choice'].reset_index(drop=True)
        if trials.loc[trans, 'probabilityLeft'] == 0.8:
            to_prior_choice = np.logical_not(r_choice).astype(int)
        else:
            to_prior_choice = r_choice.copy()
        switch_df = pd.concat((switch_df, pd.DataFrame(data={
            'right_choice': r_choice, 'trial': r_choice.index.values,
            'opto': trials.loc[trans, 'laser_stimulation'], 'to_prior_choice': to_prior_choice,
            'switch_to': trials.loc[trans, 'probabilityLeft'], 'subject': nickname,
            'sert-cre': subjects.loc[i, 'sert-cre']})), ignore_index=True)

# %%

per_animal_df = switch_df.groupby(['subject', 'trial', 'opto']).mean().reset_index()

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(x='trial', y='to_prior_choice', data=per_animal_df,
             hue='opto', errorbar='se', hue_order=[1, 0], err_kws={'lw': 0},
             palette=[colors['stim'], colors['no-stim']], ax=ax1)
ax1.text(7.5, 0.52, f'n = {len(np.unique(per_animal_df["subject"]))} mice')
ax1.set(xlim=[0, 20], ylabel='Fraction of choices towards\nthe side with the high prior', xlabel='Trials since prior switch',
        ylim=[0.4, 0.8])
leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['5-HT', 'No 5-HT']
ax1.legend(leg_handles, leg_labels, prop={'size': 6}, loc='lower right', frameon=False)

sns.despine(trim=True)
plt.tight_layout(h_pad=1.8)
plt.savefig(path.join(fig_path, 'block_switch.pdf'))

