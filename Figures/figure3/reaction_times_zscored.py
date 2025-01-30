# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:03:56 2024

By Guido Meijer
"""

import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt
from stim_functions import (paths, query_opto_sessions, load_trials, init_one,
                            figure_style, load_subjects, behavioral_criterion)

# Settings
WIN_STARTS = np.arange(-20, 50)
WIN_SIZE = 10
PLOT_SESSIONS = False
trial_win_labels = WIN_STARTS + (WIN_SIZE/2)

# Paths
f_path, save_path = paths()
fig_path = path.join(f_path, path.split(path.dirname(path.realpath(__file__)))[-1])

# Initialize
one = init_one()
_, data_path = paths()
subjects = load_subjects()

# Loop over subjects
rt_df, block_df, block_end_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for i, subject in enumerate(subjects['subject']):
    print(f'{subject} ({i} of {subjects.shape[0]})')

    # Query sessions
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    if sert_cre == 0:
        continue
    eids = query_opto_sessions(subject, include_ephys=True, one=one)
    eids = behavioral_criterion(eids, verbose=False, one=one)
    if len(eids) == 0:
        continue
        
    # Loop over sessions
    rt_opto_median, rt_no_opto_median = [], []
    for j, eid in enumerate(eids):
        try:
            trials_df = load_trials(eid, laser_stimulation=True)
        except Exception:
            continue
        if np.sum(trials_df['laser_probability'] == 0.5) > 0:
            continue
           
        # Get reaction times 
        #trials_df['rt'] = trials_df['firstMovement_times'] - trials_df['goCue_times']
        trials_df['rt'] = trials_df['feedback_times'] - trials_df['goCue_times']
        trials_df = trials_df[~np.isnan(trials_df['rt'])]
        
        # Log transform and then z-score reaction times per contrast
        trials_df['rt'] = np.log10(trials_df['rt'])
        trials_df['abs_contrast'] = np.abs(trials_df['signed_contrast'])
        for ii, this_contrast in enumerate(np.unique(trials_df['abs_contrast'])):
            trials_df.loc[trials_df['abs_contrast'] == this_contrast, 'rt'] = zscore(
                trials_df.loc[trials_df['abs_contrast'] == this_contrast, 'rt'])
                    
       
        # Remove probe trials
        trials_df.loc[(trials_df['laser_probability'] == 0.25)
                      & (trials_df['laser_stimulation'] == 1), 'laser_stimulation'] = 0
        trials_df.loc[(trials_df['laser_probability'] == 0.75)
                      & (trials_df['laser_stimulation'] == 0), 'laser_stimulation'] = 1
    
        # Get states centered at opto block switches
        this_block_df, this_end_df = pd.DataFrame(), pd.DataFrame()
        all_blocks = 0
        trials_df['opto_block_switch'] = np.concatenate((
            [False], np.diff(trials_df['laser_stimulation']) != 0))
        opto_block_switch_ind = trials_df[trials_df['opto_block_switch']].index
        for b, trial_ind in enumerate(opto_block_switch_ind):
            all_blocks += 1
            
            # Binned trials
            these_rts, these_trial_bins = [], []
            for tt, this_start in enumerate(WIN_STARTS-1):
                trial_win = trials_df[trial_ind+this_start:trial_ind+this_start+WIN_SIZE]
                if trial_win.shape[0] == WIN_SIZE:
                    these_rts.append(trial_win['rt'].mean())
                    these_trial_bins.append(trial_win_labels[tt])

            this_block_df = pd.concat((this_block_df, pd.DataFrame(data={
                'rt': np.array(these_rts), 'trial_bin': np.array(these_trial_bins),
                'opto_switch': all_blocks,
                'opto': trials_df.loc[trial_ind, 'laser_stimulation']})), ignore_index=True)
            
                    
        # Get reaction time medians
        rt_opto_median.append(trials_df.loc[trials_df['laser_stimulation'] == 1, 'rt'].median())
        rt_no_opto_median.append(trials_df.loc[trials_df['laser_stimulation'] == 0, 'rt'].median())
             
    # Add to dataframe
    rt_df = pd.concat((rt_df, pd.DataFrame(index=[rt_df.shape[0]], data={
        'subject': subject, 'sert-cre': sert_cre,
        'rt_opto': np.mean(rt_opto_median), 'rt_no_opto': np.mean(rt_no_opto_median)})))

    this_rt = this_block_df[this_block_df['opto'] == 1].groupby('trial_bin').mean(numeric_only=True)['rt']
    block_df = pd.concat((block_df, pd.DataFrame(data={
        'rt': this_rt,
        'rt_bl': this_rt - np.mean(this_rt.values[:np.sum(trial_win_labels < -5)]),
        'trial': trial_win_labels, 'subject': subject,
        'sert-cre': sert_cre,
        'opto': 1})))
    this_rt = this_block_df[this_block_df['opto'] == 0].groupby('trial_bin').mean(numeric_only=True)['rt']
    block_df = pd.concat((block_df, pd.DataFrame(data={
        'rt': this_rt,
        'rt_bl': this_rt - np.mean(this_rt.values[:np.sum(trial_win_labels < -5)]),
        'trial': trial_win_labels, 'subject': subject,
        'sert-cre': sert_cre,
        'opto': 0})))
    
 
  
# %% Plot
    
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)

sns.lineplot(data=block_df, x='trial', y='rt_bl', hue='opto', ax=ax1,
             hue_order=[1, 0], palette=[colors['stim'], colors['no-stim']],
             errorbar='se', err_kws={'lw': 0})
ax1.plot(ax1.get_xlim(), [0, 0], ls='--', color='grey', lw=0.5)
ax1.set(ylabel='Baseline subtr. reaction time (std)', 
        xticks=[-20, 0, 20, 40, 60],
        xlabel='Trials since stim. block switch')

leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['Start', 'End']
leg = ax1.legend(leg_handles, leg_labels, prop={'size': 5}, bbox_to_anchor=[0.9, 0.5], frameon=False,
                 title='5-HT block')
leg.get_title().set_fontsize('6')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'reaction_time_opto_block_bl.pdf'))

# %%
f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)

sns.lineplot(data=block_df, x='trial', y='rt', hue='opto', ax=ax1,
             hue_order=[1, 0], palette=[colors['stim'], colors['no-stim']],
             errorbar='se', err_kws={'lw': 0})
ax1.set(ylabel='Z-scored reaction time (std)',
        xticks=[-20, 0, 20, 40, 60],
        xlabel='Trials since stim. block switch')

leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['Start', 'End']
leg = ax1.legend(leg_handles, leg_labels, prop={'size': 5}, bbox_to_anchor=[0.9, 0.5], frameon=False,
                 title='5-HT block')
leg.get_title().set_fontsize('6')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'reaction_time_opto_block.pdf'))


# %%

f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
for i in rt_df.index:
    ax1.plot([0, 1], [rt_df.loc[i, 'rt_opto'], rt_df.loc[i, 'rt_opto']], marker='o', color='k',
             markersize=2)
ax1.set(xticks=[0, 1], xticklabels=['5-HT', 'No 5-HT'], ylabel='Median reaction time (s)',
        yticks=np.arange(0, 1.5, 0.2), xlim=[-0.2, 1.2])
ax1.text(0.5, 1.3, 'n.s.', ha='center', va='center')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'reaction_time_median.pdf'))

