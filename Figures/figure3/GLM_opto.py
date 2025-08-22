#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:53:33 2021
By: Guido Meijer
"""

import numpy as np
from os import path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stim_functions import (paths, behavioral_criterion, load_trials, figure_style,
                            query_opto_sessions, load_subjects, fit_glm, init_one)
one = init_one()

# Settings
MIN_SES = 2
subjects = load_subjects()
f_path, save_path = paths()
fig_path = path.join(f_path, path.split(path.dirname(path.realpath(__file__)))[-1])
colors, dpi = figure_style()

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    
    # Only use sert-cre animals
    if subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0] == 0:
        continue    
    print(f'{nickname}')

    # Query sessions
    eids = query_opto_sessions(nickname, include_ephys=True, one=one)
    eids = behavioral_criterion(eids, verbose=False, one=one)
    if len(eids) < MIN_SES:
        continue

    # Get trials DataFrame
    trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, invert_choice=True, one=one)
        except:
            continue
        these_trials = these_trials.rename(columns={'feedbackType': 'trial_feedback_type'})
        these_trials['block_id'] = (these_trials['probabilityLeft'] == 0.2).astype(int)
        these_trials['stimulus_side'] = these_trials['stim_side'].copy()
        these_trials.loc[these_trials['signed_contrast'] == 0, 'stimulus_side'] = 0
        these_trials['contrast'] = these_trials['signed_contrast'].abs() * 100
        these_trials['previous_choice'] = these_trials['choice'].shift(periods=1)
        trials = pd.concat((trials, these_trials), ignore_index=True)

    # Remove no-go trials
    trials = trials[trials['choice'] != 0]
    
    # Fit GLM
    params_all = fit_glm(trials)
    
    # Add to dataframe
    results_df = pd.concat((results_df, params_all), ignore_index=True)
    results_df.loc[results_df.shape[0]-1, 'subject'] = nickname
    results_df.loc[results_df.shape[0]-1, 'sert-cre'] = subjects.loc[i, 'sert-cre']
   
# %% Plot
long_df = pd.melt(results_df.loc[results_df['sert-cre'] == 1,
                                 ['100_1', '100_0', '25_1', '25_0', '12_1', '12_0', '6_1', '6_0',
                                  'previous_choice_0', 'previous_choice_1', 'prior_0', 'prior_1']])
long_df['pair'] = [i[:-2] for i in long_df['variable']]
long_df['5HT'] = [i[-1] for i in long_df['variable']]

f, ax1 = plt.subplots(1, 1, figsize=(2.2, 2), dpi=dpi)
sns.barplot(data=long_df, x='pair', y='value', hue='5HT', hue_order=['1', '0'],
            palette=[colors['stim'], colors['no-stim']], zorder=1, linewidth=0)


#ax1.text(2.5, 2.5, f'n = {np.sum(results_df["sert-cre"] == 1)} mice')
#ax1.plot(ax1.get_xlim(), [0, 0], color='grey', zorder=0)
ax1.set(ylabel='Weight', xticks=np.arange(6), xlabel='', yticks=[0, 1, 2, 3], xlim=[-0.75, 5.75])
ax1.set_xticklabels(['100%', '25%', '12.5%', '6.25%', 'Prev. choice', 'Prior'],
                    rotation=40, ha='right')
handles, previous_labels = ax1.get_legend_handles_labels()
ax1.legend(title='', handles=handles, labels=['5-HT', 'no 5-HT'])

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'GLM_5HT_split.pdf'))