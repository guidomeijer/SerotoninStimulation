#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:09:11 2023
By: Guido Meijer
"""
import numpy as np
import pandas as pd
from os.path import join, realpath, dirname, split
from scipy.stats import ttest_rel, ttest_1samp, zscore
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import (load_subjects, query_opto_sessions, behavioral_criterion,
                            load_trials, figure_style, paths)
from one.api import ONE
one = ONE()
colors, dpi = figure_style()

# Query which subjects to use and create eid list per subject
subjects = load_subjects()

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

rt_perc_df, all_trials_rt = pd.DataFrame(), pd.DataFrame()
for i, subject in enumerate(subjects['subject']):

    # Only use sert-cre animals
    if subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0] == 0:
        continue
    
    # Query sessions
    eids = query_opto_sessions(subject, include_ephys=True, one=one)
    eids = behavioral_criterion(eids, verbose=False, one=one)
    if len(eids) < 2:
        continue

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

    print(f'{subject}: {trials.shape[0]} trials')
        
    # Remove probe trials
    #trials = trials[~((trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1))]
    #trials = trials[~((trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0))]

    trials['abs_contrast'] = np.abs(trials['signed_contrast'])
    this_rt_per_contrast = trials[['abs_contrast', 'time_to_choice', 'laser_stimulation']].groupby(
        ['abs_contrast', 'laser_stimulation']).median()
    this_rt_per_contrast = this_rt_per_contrast.reset_index()
    this_rt_per_contrast['subject'] = subject
    this_rt_perc = []
    for c, contr in enumerate(np.unique(this_rt_per_contrast['abs_contrast'])):
        rt_opto = this_rt_per_contrast.loc[
            (this_rt_per_contrast['abs_contrast'] == contr) & (this_rt_per_contrast['laser_stimulation'] == 1),
            'time_to_choice'].values[0]
        rt_no_opto = this_rt_per_contrast.loc[
            (this_rt_per_contrast['abs_contrast'] == contr) & (this_rt_per_contrast['laser_stimulation'] == 0),
            'time_to_choice'].values[0]
        perc_rt = ((rt_opto - rt_no_opto) / rt_no_opto) * 100
        ratio_rt = (rt_opto - rt_no_opto) / (rt_opto + rt_no_opto)
        diff_rt = rt_opto - rt_no_opto
        this_rt_perc.append({'perc_rt': perc_rt, 'ratio_rt': ratio_rt, 'diff_rt': diff_rt,
                             'contrast': contr, 'subject': subject})
    rt_perc_df = pd.concat((rt_perc_df, pd.DataFrame(this_rt_perc)))
    
    # Add to dataframe
    trials['rt_zscored'] = zscore(trials['time_to_choice'])
    all_trials_rt = pd.concat((all_trials_rt, trials))
    
    
    # %% Plot animal
  
    f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)

    sns.histplot(data=trials, x='time_to_choice', hue='laser_stimulation', binwidth=0.05, ax=ax1,
                 palette=[colors['no-stim'], colors['stim']])
    ax1.legend(labels=['5-HT', 'No 5-HT'], bbox_to_anchor=(0.4, 1), prop={'size': 5})
    ax1.set(xlim=[0, 2], xlabel='Reaction time (s)', xticks=[0, 0.5, 1, 1.5, 2],
            xticklabels=[0, 0.5, 1, 1.5, 2])
    #ax1.set(xscale='log')

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, f'{subject}_reaction_times.pdf'))
    
    
