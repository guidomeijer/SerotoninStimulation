#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join, realpath, dirname, split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from stim_functions import (load_trials, plot_psychometric, paths, behavioral_criterion,
                            fit_psychfunc, figure_style, query_opto_sessions, get_bias,
                            load_subjects)
from one.api import ONE
one = ONE()

# Settings
MIN_TRIALS = 400
PLOT_SINGLE_ANIMALS = False
subjects = load_subjects()
colors, dpi = figure_style()

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

bias_df, lapse_df, psy_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Subject {nickname} ({i} of {subjects.shape[0]})')
    
    # Only use sert-cre animals
    if subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0] == 0:
        continue

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
        these_trials = load_trials(eid, laser_stimulation=True, one=one)
        these_trials['session'] = ses_count
        trials = pd.concat((trials, these_trials), ignore_index=True)
        ses_count = ses_count + 1

    if len(trials) < MIN_TRIALS:
        continue

    # Get bias from fitted curves
    bias_fit_stim = get_bias(trials.loc[(trials['laser_stimulation'] == 1) & (trials['probe_trial'] == 0)])
    bias_fit_no_stim = get_bias(trials.loc[(trials['laser_stimulation'] == 0) & (trials['probe_trial'] == 0)])

    # Get bias 
    trials_select = (trials['laser_stimulation'] == 1) & (trials['probe_trial'] == 0) & (trials['signed_contrast'] == 0) & (trials['probabilityLeft'] == 0.2)
    bias_r_opto = (trials.loc[trials_select, 'correct'].sum() / trials.loc[trials_select, 'correct'].shape[0]) * 100
    trials_select = (trials['laser_stimulation'] == 0) & (trials['probe_trial'] == 0) & (trials['signed_contrast'] == 0) & (trials['probabilityLeft'] == 0.2)
    bias_r_no_opto = (trials.loc[trials_select, 'correct'].sum() / trials.loc[trials_select, 'correct'].shape[0]) * 100
    trials_select = (trials['laser_stimulation'] == 1) & (trials['probe_trial'] == 0) & (trials['signed_contrast'] == 0) & (trials['probabilityLeft'] == 0.8)
    bias_l_opto = (trials.loc[trials_select, 'correct'].sum() / trials.loc[trials_select, 'correct'].shape[0]) * 100
    trials_select = (trials['laser_stimulation'] == 0) & (trials['probe_trial'] == 0) & (trials['signed_contrast'] == 0) & (trials['probabilityLeft'] == 0.8)
    bias_l_no_opto = (trials.loc[trials_select, 'correct'].sum() / trials.loc[trials_select, 'correct'].shape[0]) * 100
    bias_opto = bias_r_opto - bias_l_opto
    bias_no_opto = bias_r_no_opto - bias_l_no_opto

    # Get performance
    perf_opto = (trials.loc[trials['laser_stimulation'] == 1, 'correct'].sum()
                 / trials.loc[trials['laser_stimulation'] == 1, 'correct'].shape[0]) * 100
    perf_no_opto = (trials.loc[trials['laser_stimulation'] == 0, 'correct'].sum()
                    / trials.loc[trials['laser_stimulation'] == 0, 'correct'].shape[0]) * 100

    # Get RT/
    rt_no_stim = trials[trials['laser_stimulation'] == 0].median()['reaction_times']
    rt_stim = trials[trials['laser_stimulation'] == 1].median()['reaction_times']
    rt_catch_no_stim = trials[(trials['laser_stimulation'] == 0)
                              & (trials['laser_probability'] == 0.25)].median()['reaction_times']
    rt_catch_stim = trials[(trials['laser_stimulation'] == 1)
                           & (trials['laser_probability'] == 0.25)].median()['reaction_times']
    bias_df = pd.concat((bias_df, pd.DataFrame(index=[bias_df.shape[0] + 1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'rt_no_stim': rt_no_stim,
        'rt_stim': rt_stim, 'rt_catch_no_stim': rt_catch_no_stim, 'rt_catch_stim': rt_catch_stim,
        'bias_fit_stim': bias_fit_stim, 'bias_fit_no_stim': bias_fit_no_stim,
        'perf_stim': perf_opto, 'perf_no_stim': perf_no_opto,
        'bias_opto': bias_opto, 'bias_no_opto': bias_no_opto})))

    # Get fit parameters
    these_trials = trials[(trials['probabilityLeft'] == 0.8) & (trials['laser_stimulation'] == 0)
                          & (trials['laser_probability'] != 0.75)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = pd.concat((psy_df, pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 0, 'prob_left': 0.8,
        'bias': pars[0], 'slope': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]})))
    these_trials = trials[(trials['probabilityLeft'] == 0.2) & (trials['laser_stimulation'] == 0)
                          & (trials['laser_probability'] != 0.75)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = pd.concat((psy_df, pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 0, 'prob_left': 0.2,
        'bias': pars[0], 'slope': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]})))
    these_trials = trials[(trials['probabilityLeft'] == 0.8) & (trials['laser_stimulation'] == 1)
                          & (trials['laser_probability'] != 0.25)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = pd.concat((psy_df, pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 1, 'prob_left': 0.8,
        'bias': pars[0], 'slope': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]})))
    these_trials = trials[(trials['probabilityLeft'] == 0.2) & (trials['laser_stimulation'] == 1)
                          & (trials['laser_probability'] != 0.25)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = pd.concat((psy_df, pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'opto_stim': 1, 'prob_left': 0.2,
        'bias': pars[0], 'slope': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]})))

    # Plot
    if PLOT_SINGLE_ANIMALS:
       
        f, ax1 = plt.subplots(figsize=(2, 2), dpi=dpi)

        # plot_psychometric(trials[trials['probabilityLeft'] == 0.5], ax=ax1, color='k')
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 0)], ax=ax1,
                          color=colors['left-no-stim'])
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 1)], ax=ax1,
                          color=colors['left-stim'])
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['laser_stimulation'] == 0)], ax=ax1,
                          color=colors['right-no-stim'])
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['laser_stimulation'] == 1)], ax=ax1,
                          color=colors['right-stim'])

        sns.despine(trim=True)
        #plt.tight_layout()
        plt.subplots_adjust(left=0.21, right=0.98, bottom=0.2, top=0.95)
        
        plt.savefig(join(fig_path, '%s_opto_behavior_psycurve.pdf' % nickname))
        plt.close(f)
        
psy_avg_block_df = psy_df.groupby(['subject', 'opto_stim']).mean()
psy_avg_block_df['lapse_both'] = psy_avg_block_df.loc[:, 'lapse_l':'lapse_r'].mean(axis=1)
psy_avg_block_df = psy_avg_block_df.reset_index()


# %% Plot

# Get percentage increase
#perc_bias = ((psy_avg_block_df.loc[psy_avg_block_df['opto_stim'] == 1, 'bias'].values
#               - psy_avg_block_df.loc[psy_avg_block_df['opto_stim'] == 0, 'bias'].values)
#              / psy_avg_block_df.loc[psy_avg_block_df['opto_stim'] == 1, 'bias'].values) * 100
perc_bias = ((bias_df['bias_fit_stim'] - bias_df['bias_fit_no_stim']) / bias_df['bias_fit_stim']).values * 100
#perc_bias = ((bias_df['bias_opto'] - bias_df['bias_no_opto']) / bias_df['bias_opto']).values * 100
perc_rt = ((bias_df['rt_stim'] - bias_df['rt_no_stim']) / bias_df['rt_stim']).values * 100
perc_lapse = ((psy_avg_block_df.loc[psy_avg_block_df['opto_stim'] == 1, 'lapse_both'].values
               - psy_avg_block_df.loc[psy_avg_block_df['opto_stim'] == 0, 'lapse_both'].values)
              / psy_avg_block_df.loc[psy_avg_block_df['opto_stim'] == 1, 'lapse_both'].values) * 100
perc_slope = ((psy_avg_block_df.loc[psy_avg_block_df['opto_stim'] == 1, 'slope'].values
                   - psy_avg_block_df.loc[psy_avg_block_df['opto_stim'] == 0, 'slope'].values)
                  / psy_avg_block_df.loc[psy_avg_block_df['opto_stim'] == 1, 'slope'].values) * 100
perc_perf = ((bias_df['perf_stim'] - bias_df['perf_no_stim']) / bias_df['perf_stim']).values * 100
perc_df = pd.DataFrame(data={'Performance': perc_perf, 'slope': perc_slope, 'Bias': perc_bias,
                             'Lapse rate': perc_lapse})

# Do statistics
p_bias = stats.ttest_1samp(perc_bias, 0)[1]
p_rt = stats.ttest_1samp(perc_rt, 0)[1]
p_lapse = stats.ttest_1samp(perc_lapse, 0)[1]
p_slope = stats.ttest_1samp(perc_slope, 0)[1]
p_perf = stats.ttest_1samp(perc_perf, 0)[1]

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 2), dpi=dpi)
sns.swarmplot(data=pd.melt(perc_df), x='variable', y='value', color='k', size=3)
ax1.plot(ax1.get_xlim(), [0, 0], ls='--', color='grey')
ax1.text(1, 35, '*', ha='center', va='center', fontsize=12)
ax1.set(ylabel='5-HT induced change (%)', xlabel='', yticks=[-40, -20, 0, 20, 40])
ax1.set_xticklabels(ax1.get_xmajorticklabels(), rotation=40, ha='right')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'summary_psycurve.pdf'))

# %%
"""
# %% Plot

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(2.2, 2), sharex=True, dpi=dpi)
for i, subject in enumerate(bias_df['subject']):   
    ax1.plot([1, 2], [bias_df.loc[bias_df['subject'] == subject, 'bias_fit_no_stim'],
                      bias_df.loc[bias_df['subject'] == subject, 'bias_fit_stim']],
             color='k', marker='o', ms=3.5, markeredgewidth=0.5, markeredgecolor='w')
#ax1.tick_params(axis='both', which='both', length=0)
ax1.set(xlabel='', xticks=[], ylim=[0, 0.4], xlim=[0.8, 2.2], yticks=[0, 0.4],
        yticklabels=[0, 0.4])
ax1.set_ylabel('Bias', labelpad=-5)

for i, subject in enumerate(psy_avg_block_df['subject']):
    ax2.plot([1, 2], 
             [psy_avg_block_df.loc[(psy_avg_block_df['subject'] == subject) & (psy_avg_block_df['opto_stim'] == 0), 'slope'],
              psy_avg_block_df.loc[(psy_avg_block_df['subject'] == subject) & (psy_avg_block_df['opto_stim'] == 1), 'slope']],
             color='k', marker='o', ms=3.5, markeredgewidth=0.5, markeredgecolor='w')
ax2.text(1.5, 35, '*', ha='center', fontsize=10)
ax2.set(xlabel='', xticks=[1, 2], xticklabels=['No stim.', 'Stim.'], ylabel='slope (σ)',
        ylim=[10, 40], yticks=[10, 20, 30, 40], xlim=[0.8, 2.2])


for i, subject in enumerate(psy_avg_block_df['subject']):
    ax3.plot([1, 2], psy_avg_block_df.loc[(psy_avg_block_df['subject'] == subject), 'lapse_both'],
             color='k', marker='o', ms=3.5, markeredgewidth=0.5, markeredgecolor='w')
ax3.set(xlabel='', xticks=[], yticks=[0, 0.3], yticklabels=[0, 0.3])
ax3.set_ylabel('Lapse rate', labelpad=-5)

for i, subject in enumerate(bias_df['subject']):
    ax4.plot([1, 2], [bias_df.loc[bias_df['subject'] == subject, 'rt_no_stim'],
                      bias_df.loc[bias_df['subject'] == subject, 'rt_stim']],
             color='k', marker='o', ms=3.5, markeredgewidth=0.5, markeredgecolor='w')
ax4.set(xlabel='', xticks=[1, 2], xticklabels=['No stim.', 'Stim.'], ylabel='Reaction time (s)',
        yscale='log', yticks=[0.1, 1], yticklabels=[0.1, 1])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'summary_psycurve.pdf'))
"""