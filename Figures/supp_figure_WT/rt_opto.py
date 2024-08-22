#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:09:11 2023
By: Guido Meijer
"""
import numpy as np
import pandas as pd
from os.path import join, realpath, dirname, split
from scipy.stats import ttest_rel, ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import (load_subjects, query_opto_sessions, behavioral_criterion,
                            load_trials, figure_style, paths)
from one.api import ONE
one = ONE()

# Settings
MIN_TRIALS = 400

# Query which subjects to use and create eid list per subject
subjects = load_subjects()

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

rt_perc_df = pd.DataFrame()
for i, subject in enumerate(subjects['subject']):

    # Only use sert-cre animals
    if subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0] == 1:
        continue
    
    # Query sessions
    eids = query_opto_sessions(subject, include_ephys=True, one=one)
    eids = behavioral_criterion(eids, min_perf=0.7, verbose=False, one=one)

    # Loop over sessions
    trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, one=one)
        except Exception as err:
            print(err)
            continue
        if np.sum(these_trials['laser_probability'] == 0.5) > 0:
            continue
        these_trials['trial'] = these_trials.index.values
        these_trials['session'] = j
        trials = pd.concat((trials, these_trials), ignore_index=True)
    if trials.shape[0] < MIN_TRIALS:
        continue
    print(f'{subject}: {trials.shape[0]} trials')
        
    # Remove probe trials
    #trials = trials[~((trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1))]
    #trials = trials[~((trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0))]

    trials['abs_contrast'] = np.abs(trials['signed_contrast'])
    this_rt_per_contrast = trials[['abs_contrast', 'reaction_times', 'laser_stimulation']].groupby(
        ['abs_contrast', 'laser_stimulation']).median()
    this_rt_per_contrast = this_rt_per_contrast.reset_index()
    this_rt_per_contrast['subject'] = subject
    this_rt_perc = []
    for c, contr in enumerate(np.unique(this_rt_per_contrast['abs_contrast'])):
        rt_opto = this_rt_per_contrast.loc[
            (this_rt_per_contrast['abs_contrast'] == contr) & (this_rt_per_contrast['laser_stimulation'] == 1),
            'reaction_times'].values[0]
        rt_no_opto = this_rt_per_contrast.loc[
            (this_rt_per_contrast['abs_contrast'] == contr) & (this_rt_per_contrast['laser_stimulation'] == 0),
            'reaction_times'].values[0]
        perc_rt = ((rt_opto - rt_no_opto) / rt_no_opto) * 100
        ratio_rt = (rt_opto - rt_no_opto) / (rt_opto + rt_no_opto)
        diff_rt = rt_opto - rt_no_opto
        this_rt_perc.append({'perc_rt': perc_rt, 'ratio_rt': ratio_rt, 'diff_rt': diff_rt,
                             'contrast': contr, 'subject': subject})
    rt_perc_df = pd.concat((rt_perc_df, pd.DataFrame(this_rt_perc)))
        
# %%
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.barplot(data=rt_perc_df, x='contrast', y='perc_rt', errorbar='se', ax=ax1)
ax1.set(ylabel='5-HT induced reaction time\nincrease (%)', xticks=np.arange(5),
        xticklabels=[0, 6.25, 12.5, 25, 100], xlabel='Stimulus contrast (%)')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'reaction_time_per_contrast.pdf'))
