# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:54:26 2025

By Guido Meijer
"""

import numpy as np
from os import path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from stim_functions import (load_subjects, query_opto_sessions, behavioral_criterion, init_one,
                            load_trials, figure_style, paths)
fig_path, save_path = paths()
one = init_one()
subjects = load_subjects()
colors, dpi = figure_style()

BINS = 5
MIN_TRIALS = 150

trial_bins, all_trials = pd.DataFrame(), pd.DataFrame()
for i, subject in enumerate(subjects['subject']):
    print(f'{subject} ({i} of {subjects.shape[0]})')

    # Query sessions
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    if sert_cre == 0:
        continue
    eids = query_opto_sessions(subject, include_ephys=True, one=one)
    #eids = behavioral_criterion(eids, verbose=False, one=one)
    if len(eids) == 0:
        continue
        
    # Loop over sessions
    for j, eid in enumerate(eids):
        try:
            trials_df = load_trials(eid, laser_stimulation=True)
        except Exception:
            continue
        if np.sum(trials_df['laser_probability'] == 0.5) > 0:
            continue
        if trials_df.shape[0] < MIN_TRIALS:
            continue
        
        # Clean up RTs
        trials_df.loc[trials_df['reaction_times'] <= 0, 'reaction_times'] = 0.01
        trials_df = trials_df[~np.isnan(trials_df['reaction_times'])]
        trials_df = trials_df.reset_index()     
        
        # Log transform and then z-score reaction times per contrast
        trials_df['rt_zscore'] = np.log10(trials_df['reaction_times'])
        trials_df['abs_contrast'] = np.abs(trials_df['signed_contrast'])
        for ii, this_contrast in enumerate(np.unique(trials_df['abs_contrast'])):
            trials_df.loc[trials_df['abs_contrast'] == this_contrast, 'rt_zscore'] = zscore(
                trials_df.loc[trials_df['abs_contrast'] == this_contrast, 'rt_zscore'])    
        
        # Remove probe trials
        trials_df.loc[(trials_df['laser_probability'] == 0.25)
                      & (trials_df['laser_stimulation'] == 1), 'laser_stimulation'] = 0
        trials_df.loc[(trials_df['laser_probability'] == 0.75)
                      & (trials_df['laser_stimulation'] == 0), 'laser_stimulation'] = 1
        
        # Get opto block switch points
        trials_df['opto_block_switch'] = np.concatenate((
            [False], np.diff(trials_df['laser_stimulation']) != 0))
        opto_block_switch_ind = np.concatenate((
            [0], trials_df[trials_df['opto_block_switch']].index.values))
        
        # Loop over opto blocks
        for ii, this_block_start in enumerate(opto_block_switch_ind[:-1]):
            trials_df.loc[this_block_start:opto_block_switch_ind[ii+1] - 1, 'trial_bins'] = pd.qcut(
                np.arange((opto_block_switch_ind[ii+1] - this_block_start)), q=BINS, labels=np.arange(BINS))
        
        # Get mean RT per trial quantile
        bin_means = trials_df[['laser_stimulation', 'trial_bins', 'rt_zscore']].groupby(
            ['laser_stimulation', 'trial_bins'], observed=True).mean().reset_index()
        bin_means['subject'] = subject
        bin_means['eid'] = eid
        
        # Add to df
        trial_bins = pd.concat((trial_bins, bin_means))
        all_trials = pd.concat((all_trials, trials_df))
        
# Get means over sessions per animal
animal_means = trial_bins.groupby(['subject', 'laser_stimulation', 'trial_bins'], observed=True).mean(
    numeric_only=True)
all_trials = all_trials.reset_index()

# %% Plot

f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
sns.lineplot(all_trials, x='trial_bins', y='rt_zscore', hue='laser_stimulation', errorbar='se',
             ax=ax1, err_kws={'lw': 0}, marker='o', hue_order=[0, 1],
             palette=[colors['no-stim'], colors['stim']])
ax1.set(ylabel='Z-scored reaction time (std)', xlabel='Equidistant trial bins in block')
ax1.legend(bbox_to_anchor=(1, 1), title='5-HT')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'Extra plots', 'RT_trial_bins.jpg'), dpi=dpi)        
