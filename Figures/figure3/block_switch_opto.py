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
from stim_functions import (load_trials, paths, behavioral_criterion,
                            figure_style, query_opto_sessions, load_subjects)
from one.api import ONE
one = ONE()

# Settings
PLOT_SINGLE_ANIMALS = False
subjects = load_subjects()

# Paths
f_path, save_path = paths()
fig_path = path.join(f_path, path.split(path.dirname(path.realpath(__file__)))[-1])

switch_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'{nickname} ({i} of {subjects.shape[0]})')

    # Query sessions
    eids = query_opto_sessions(nickname, include_ephys=True, one=one)

    # Apply behavioral criterion
    eids = behavioral_criterion(eids, verbose=False, one=one)
    if len(eids) == 0:
        continue

    # Get trials DataFrame
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, one=one)
            these_trials['session'] = ses_count
            trials = pd.concat((trials, these_trials), ignore_index=True)
            ses_count = ses_count + 1
        except:
            pass
    if len(trials) == 0:
        continue

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
    
    if PLOT_SINGLE_ANIMALS:
        colors, dpi = figure_style()
        f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
        sns.lineplot(x='trial', y='to_prior_choice', data=switch_df[switch_df['subject'] == nickname],
                     hue='opto', errorbar='se', hue_order=[1, 0],
                     palette=[colors['stim'], colors['no-stim']])
        ax1.set(xlim=[0, 20], ylabel='Frac. choices to biased side', xlabel='Trials since switch',
                title=f'{nickname}, SERT: {subjects.loc[i, "sert-cre"]}')
        leg_handles, _ = ax1.get_legend_handles_labels()
        leg_labels = ['Opto', 'No opto']
        ax1.legend(leg_handles, leg_labels, prop={'size': 5}, loc='lower left', frameon=False)
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(path.join(fig_path, f'block_switch_{nickname}.jpg'), dpi=300)

# %%

per_animal_df = switch_df.groupby(['subject', 'trial', 'opto']).mean().reset_index()

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(x='trial', y='to_prior_choice', data=per_animal_df[per_animal_df['sert-cre'] == 1],
             hue='opto', errorbar='se', hue_order=[1, 0], err_kws={'lw': 0},
             palette=[colors['stim'], colors['no-stim']], ax=ax1)
ax1.set(xlim=[0, 20], ylabel='Fraction of choices towards\nthe biased side', xlabel='Trials since bias switch',
        ylim=[0.4, 0.8])
leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['5-HT', 'No 5-HT']
ax1.legend(leg_handles, leg_labels, prop={'size': 6}, loc='lower right', frameon=False)

sns.despine(trim=True)
plt.tight_layout(h_pad=1.8)
plt.savefig(path.join(fig_path, 'block_switch.pdf'))

