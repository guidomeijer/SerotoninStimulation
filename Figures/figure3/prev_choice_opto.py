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
SINGLE_TRIALS = [2, 10]
TRIAL_BINS = np.arange(-15, 41, 15)
trial_bin_size = np.unique(np.diff(TRIAL_BINS))[0]
trial_bin_labels = TRIAL_BINS[:-1] + (np.diff(TRIAL_BINS) / 2)

# Query which subjects to use and create eid list per subject
subjects = load_subjects()

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

p_repeat_df, p_repeat_bins_df = pd.DataFrame(), pd.DataFrame()
p_repeat_probe_df, p_repeat_single_df = pd.DataFrame(), pd.DataFrame()
p_repeat_contrast_df = pd.DataFrame()
for i, subject in enumerate(subjects['subject']):

    # Query sessions
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
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

    # Get repeated choices
    trials['repeat_choice'] = np.concatenate((
        [False], trials['choice'].values[:-1] == trials['choice'].values[1:])).astype(int)

    # Get P(repeat choice) centered at probe trials
    trials['probe_trial'] = (((trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1))
                             | ((trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0)))
    this_probe_df = pd.DataFrame()
    for s in np.unique(trials['session']):
        this_ses = trials[trials['session'] == s].reset_index(drop=True)
        opto_probe_ind = this_ses[this_ses['probe_trial']].index
        for b, trial_ind in enumerate(opto_probe_ind):
            
            # Single trials
            these_repeats = this_ses.loc[trial_ind-SINGLE_TRIALS[0]:trial_ind+SINGLE_TRIALS[-1],
                                         'repeat_choice'].values
            this_ses['rel_trial'] = this_ses['trial'] - trial_ind
            these_trials = this_ses.loc[trial_ind-SINGLE_TRIALS[0]:trial_ind+SINGLE_TRIALS[-1],
                                        'rel_trial']
            this_probe_df = pd.concat((this_probe_df, pd.DataFrame(data={
                'repeat_choice': these_repeats, 'trial': these_trials,
                'opto': this_ses.loc[trial_ind, 'laser_stimulation']})))
                 
    # Get P(choice) depending on stim contrast
    trials['unsigned_contrast'] = np.abs(trials['signed_contrast'])
    rep_contrast = (trials.groupby(['unsigned_contrast', 'laser_stimulation']).sum()['repeat_choice']
                    / trials.groupby(['unsigned_contrast', 'laser_stimulation']).size()).reset_index()
    rep_contrast = rep_contrast.rename(columns={0: 'repeat_choice'})
    rep_contrast['subject'] = subject
    rep_contrast['sert-cre'] = sert_cre    
            
    # Remove probe trials
    #trials.loc[(trials['laser_probability'] == 0.25) | (trials['laser_probability'] == 0.75), 'repeat_choice'] = np.nan
    trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1), 'laser_stimulation'] = 0
    trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0), 'laser_stimulation'] = 1

    # Get P(repeat choice) centered at stimulation block switches
    this_block_df, this_block_single_df = pd.DataFrame(), pd.DataFrame()
    all_blocks = 0
    for s in np.unique(trials['session']):
        this_ses = trials[trials['session'] == s].reset_index(drop=True)    
        this_ses['opto_block_switch'] = np.concatenate(([False], np.diff(this_ses['laser_stimulation']) != 0))
        opto_block_switch_ind = this_ses[this_ses['opto_block_switch']].index
        for b, trial_ind in enumerate(opto_block_switch_ind):
            all_blocks += 1
            
            # Single trials
            these_repeats = this_ses.loc[trial_ind-SINGLE_TRIALS[0]:trial_ind+SINGLE_TRIALS[-1],
                                         'repeat_choice'].values
            this_ses['rel_trial'] = this_ses['trial'] - trial_ind
            these_trials = this_ses.loc[trial_ind-SINGLE_TRIALS[0]:trial_ind+SINGLE_TRIALS[-1],
                                        'rel_trial']
            this_block_single_df = pd.concat((this_block_single_df, pd.DataFrame(data={
                'repeat_choice': these_repeats, 'trial': these_trials,
                'opto': this_ses.loc[trial_ind, 'laser_stimulation']})))
                        
            # Binned trials
            these_p_rep = np.empty(len(TRIAL_BINS)-1)
            these_p_rep[:] = np.nan
            for tt, this_edge in enumerate(TRIAL_BINS[:-1]):
                if this_ses[trial_ind+this_edge:trial_ind+TRIAL_BINS[tt+1]].shape[0] == trial_bin_size:
                    these_repeats = this_ses.loc[trial_ind+this_edge:(trial_ind+TRIAL_BINS[tt+1])-1, 'repeat_choice'].values
                    these_p_rep[tt] = np.sum(these_repeats) / these_repeats.shape[0]
                        
            this_block_df = pd.concat((this_block_df, pd.DataFrame(data={
                'repeat_choice': these_p_rep, 'trial_bin': trial_bin_labels,
                'opto_switch': all_blocks, 'trial_ind': np.arange(len(trial_bin_labels)),
                'opto': this_ses.loc[trial_ind, 'laser_stimulation']})), ignore_index=True)


    # Get P(repeat choice)
    no_stim_trials = trials[trials['laser_stimulation'] == 0]
    stim_trials = trials[trials['laser_stimulation'] == 1]
    p_no_stim = (np.sum(no_stim_trials['choice'].values[:-1] == no_stim_trials['choice'].values[1:])
                 / (no_stim_trials.shape[0]-1))
    p_stim = (np.sum(stim_trials['choice'].values[:-1] == stim_trials['choice'].values[1:])
              / (stim_trials.shape[0]-1))

    # Add to dataframe
    this_repeats = this_probe_df[this_probe_df['opto'] == 1].groupby('trial').mean()['repeat_choice'].values * 100
    p_repeat_probe_df = pd.concat((p_repeat_probe_df, pd.DataFrame(data={
        'p_repeat': this_repeats, 'p_repeat_bl': this_repeats - np.mean(this_repeats[:SINGLE_TRIALS[0]]),
        'trial': np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[1]+1), 'sert-cre': sert_cre, 'subject': subject,
        'opto': 1})))
    this_repeats = this_probe_df[this_probe_df['opto'] == 0].groupby('trial').mean()['repeat_choice'].values * 100
    p_repeat_probe_df = pd.concat((p_repeat_probe_df, pd.DataFrame(data={
        'p_repeat': this_repeats, 'p_repeat_bl': this_repeats - np.mean(this_repeats[:SINGLE_TRIALS[0]]),
        'trial': np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[1]+1), 'sert-cre': sert_cre, 'subject': subject,
        'opto': 0})))
    
    this_repeats = this_block_single_df[this_block_single_df['opto'] == 1].groupby('trial').mean()['repeat_choice'].values * 100
    p_repeat_single_df = pd.concat((p_repeat_single_df, pd.DataFrame(data={
        'p_repeat': this_repeats, 'p_repeat_bl': this_repeats - np.mean(this_repeats[:SINGLE_TRIALS[0]]),
        'trial': np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[1]+1), 'sert-cre': sert_cre, 'subject': subject,
        'opto': 1})))
    this_repeats = this_block_single_df[this_block_single_df['opto'] == 0].groupby('trial').mean()['repeat_choice'].values * 100
    p_repeat_single_df = pd.concat((p_repeat_single_df, pd.DataFrame(data={
        'p_repeat': this_repeats, 'p_repeat_bl': this_repeats - np.mean(this_repeats[:SINGLE_TRIALS[0]]),
        'trial': np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[1]+1), 'sert-cre': sert_cre, 'subject': subject,
        'opto': 0})))
        
    this_repeats = this_block_df[this_block_df['opto'] == 1].groupby('trial_ind').mean(numeric_only=True)['repeat_choice'] * 100
    p_repeat_bins_df = pd.concat((p_repeat_bins_df, pd.DataFrame(data={
        'p_repeat': this_repeats,
        'p_repeat_bl': this_repeats - np.mean(this_repeats.values[:np.sum(trial_bin_labels < 0)]),
        'trial': trial_bin_labels, 'subject': subject,
        'sert-cre': sert_cre,
        'opto': 1})))
    this_repeats = this_block_df[this_block_df['opto'] == 0].groupby('trial_ind').mean(numeric_only=True)['repeat_choice'] * 100
    p_repeat_bins_df = pd.concat((p_repeat_bins_df, pd.DataFrame(data={
        'p_repeat': this_repeats,
        'p_repeat_bl': this_repeats - np.mean(this_repeats.values[:np.sum(trial_bin_labels < 0)]),
        'trial': trial_bin_labels, 'subject': subject,
        'sert-cre': sert_cre,
        'opto': 0})))
    
    p_repeat_df = pd.concat((p_repeat_df, pd.DataFrame(index=[p_repeat_df.shape[0]+1], data={
        'p_no_stim': p_no_stim, 'p_stim': p_stim, 'subject': subject,
        'sert-cre': sert_cre})))
    
    p_repeat_contrast_df = pd.concat((p_repeat_contrast_df, rep_contrast))



# %%
color, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(1.2, 2), dpi=dpi, sharex=True)

p_repeat_bins_df = p_repeat_bins_df[p_repeat_bins_df['sert-cre'] == 1]
p = ttest_rel(p_repeat_bins_df.loc[(p_repeat_bins_df['trial'] == 7.5) & (p_repeat_bins_df['opto'] == 0), 'p_repeat'],
              p_repeat_bins_df.loc[(p_repeat_bins_df['trial'] == 7.5) & (p_repeat_bins_df['opto'] == 1), 'p_repeat'])[1]
print(f'block p value: {p}')
for i, subject in enumerate(np.unique(p_repeat_bins_df['subject'])):
    this_df = p_repeat_bins_df[p_repeat_bins_df['subject'] == subject]
    ax1.plot([1, 2], [this_df.loc[(this_df['trial'] == 7.5) & (this_df['opto'] == 0), 'p_repeat'],
                      this_df.loc[(this_df['trial'] == 7.5) & (this_df['opto'] == 1), 'p_repeat']],
             color='k', marker='o', ms=3.5, markeredgewidth=0.5, markeredgecolor='w')
ax1.text(1.5, 78, '*', fontsize=10, ha='center')
ax1.set(xticks=[1, 2], xlim=[0.8, 2.2], ylim=[65, 80], title='Block of 15 trials')

p_repeat_probe_df = p_repeat_probe_df[p_repeat_probe_df['sert-cre'] == 1]
p = ttest_rel(p_repeat_probe_df.loc[(p_repeat_probe_df['trial'] == 0) & (p_repeat_probe_df['opto'] == 0), 'p_repeat'],
              p_repeat_probe_df.loc[(p_repeat_probe_df['trial'] == 0) & (p_repeat_probe_df['opto'] == 1), 'p_repeat'])[1]
print(f'probe trial p value: {p}')
for i, subject in enumerate(np.unique(p_repeat_probe_df['subject'])):
    this_df = p_repeat_probe_df[p_repeat_probe_df['subject'] == subject]
    ax2.plot([1, 2], [this_df.loc[(this_df['trial'] == 0) & (this_df['opto'] == 0), 'p_repeat'],
                      this_df.loc[(this_df['trial'] == 0) & (this_df['opto'] == 1), 'p_repeat']],
             color='k', marker='o', ms=3.5, markeredgewidth=0.5, markeredgecolor='w')
ax2.text(1.5, 78, '*', fontsize=10, ha='center')
ax2.set(xticks=[1, 2], xticklabels=['No stim', 'Stim'], title='Single probe trial',
        xlim=[0.8, 2.2], ylim=[65, 80])
f.text(0.04, 0.5, 'P[repeat choice] (%)', va='center', rotation='vertical')

#plt.tight_layout()
plt.subplots_adjust(left=0.32, right=0.96, hspace=0.4)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'prev_choice_opto.pdf'))