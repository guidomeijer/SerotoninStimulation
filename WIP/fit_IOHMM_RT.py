# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:03:56 2024

By Guido Meijer
"""

import numpy as np
from os import path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IOHMM import UnSupervisedIOHMM
from IOHMM import OLS, CrossEntropyMNL
from stim_functions import (paths, query_opto_sessions, load_trials, init_one,
                            figure_style, load_subjects, behavioral_criterion)

# Settings
SINGLE_TRIALS = [2, 10]
WIN_STARTS = np.arange(-20, 70) 
WIN_SIZE = 15
trial_win_labels = WIN_STARTS + (WIN_SIZE/2)

# Paths
f_path, save_path = paths()
fig_path = path.join(f_path, path.split(path.dirname(path.realpath(__file__)))[-1])

# Initialize
one = init_one()
colors, dpi = figure_style()
_, data_path = paths()
subjects = load_subjects()

# Initialize IOHMM with two states
SHMM = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6)
SHMM.set_models(model_emissions = [OLS()], 
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))
SHMM.set_inputs(covariates_initial = [],
                covariates_transition = [],
                covariates_emissions = [[]])
SHMM.set_outputs([['rt']])

# Loop over subjects
state_df, block_df, block_single_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
probe_df = pd.DataFrame()
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
    eng_stim, eng_nostim, switch_stim, switch_nostim = [], [], [], []
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
            
        # Start training
        SHMM.set_data([trials_df])
        SHMM.train()
        
        # Get posterior probabilities and most likely state per trial
        post_prob = np.exp(SHMM.log_gammas[0])
        predicted_states = np.array([np.argmax(i, axis=0) for i in post_prob])
        
        # Determine the engaged state as the one with the lowest RT
        # engaged state = 1, disengaged state = 0
        if (np.median(trials_df['rt'].values[predicted_states == 0])
                < np.median(trials_df['rt'].values[predicted_states == 1])):
            predicted_states = np.where((predicted_states==0) | (predicted_states==1),
                                        predicted_states^1, predicted_states)
            trials_df['p_engaged'] = post_prob[:, 0]
        else:
            trials_df['p_engaged'] = post_prob[:, 1]
        trials_df['state'] = predicted_states
    
        # Get state transitions
        trials_df['state_switch'] = np.concatenate((
            [0], (np.diff(predicted_states) != 0).astype(int)))
        
        # Get P(state) centered at probe trials
        trials_df['probe_trial'] = (((
            trials_df['laser_probability'] == 0.25) & (trials_df['laser_stimulation'] == 1))
            | ((trials_df['laser_probability'] == 0.75) & (trials_df['laser_stimulation'] == 0)))
        this_probe_df = pd.DataFrame()
        opto_probe_ind = trials_df[trials_df['probe_trial']].index
        for b, trial_ind in enumerate(opto_probe_ind):
            
            # Single trials
            these_states = trials_df.loc[trial_ind-SINGLE_TRIALS[0]:trial_ind+SINGLE_TRIALS[-1],
                                         'p_engaged'].values
            trials_df['rel_trial'] = trials_df.index.values - trial_ind
            these_trials = trials_df.loc[trial_ind-SINGLE_TRIALS[0]:trial_ind+SINGLE_TRIALS[-1],
                                         'rel_trial']
            this_probe_df = pd.concat((this_probe_df, pd.DataFrame(data={
                'state': these_states, 'trial': these_trials,
                'opto': trials_df.loc[trial_ind, 'laser_stimulation']})))
        
        # Remove probe trials
        trials_df.loc[(trials_df['laser_probability'] == 0.25)
                      & (trials_df['laser_stimulation'] == 1), 'laser_stimulation'] = 0
        trials_df.loc[(trials_df['laser_probability'] == 0.75)
                      & (trials_df['laser_stimulation'] == 0), 'laser_stimulation'] = 1
        
        # Get precentage of disengaged trials for stimulated vs non-stimulated blocks
        eng_stim.append(trials_df.loc[trials_df['laser_stimulation'] == 1, 'p_engaged'].mean() * 100)
        eng_nostim.append(trials_df.loc[trials_df['laser_stimulation'] == 0, 'p_engaged'].mean() * 100)
        
        # Get percentage of state switches for stimulated vs non-stimulated blocks
        switch_stim.append((np.sum(trials_df.loc[trials_df['laser_stimulation'] == 1, 'state_switch'])
                            / np.sum(trials_df['laser_stimulation'] == 1)) * 100)
        switch_nostim.append((np.sum(trials_df.loc[trials_df['laser_stimulation'] == 0, 'state_switch'])
                              / np.sum(trials_df['laser_stimulation'] == 0)) * 100)

        # Get states centered at opto block switches
        this_block_df, this_block_single_df = pd.DataFrame(), pd.DataFrame()
        all_blocks = 0
        trials_df['opto_block_switch'] = np.concatenate((
            [False], np.diff(trials_df['laser_stimulation']) != 0))
        opto_block_switch_ind = trials_df[trials_df['opto_block_switch']].index
        for b, trial_ind in enumerate(opto_block_switch_ind):
            all_blocks += 1
            
            # Single trials
            these_states = trials_df.loc[trial_ind-SINGLE_TRIALS[0]:trial_ind+SINGLE_TRIALS[-1],
                                         'p_engaged'].values
            trials_df['rel_trial'] = trials_df.index.values - trial_ind
            these_trials = trials_df.loc[trial_ind-SINGLE_TRIALS[0]:trial_ind+SINGLE_TRIALS[-1],
                                        'rel_trial']
            this_block_single_df = pd.concat((this_block_single_df, pd.DataFrame(data={
                'state': these_states, 'trial': these_trials,
                'opto': trials_df.loc[trial_ind, 'laser_stimulation']})))
                        
            # Binned trials
            these_p_state, these_trial_bins = [], []
            for tt, this_start in enumerate(WIN_STARTS-1):
                trial_win = trials_df[trial_ind+this_start:trial_ind+this_start+WIN_SIZE]
                if trial_win.shape[0] == WIN_SIZE:
                    these_p_state.append(trial_win['p_engaged'].mean())
                    these_trial_bins.append(trial_win_labels[tt])
                    
                
            
            
            """
            these_p_state = np.empty(len(TRIAL_BINS)-1)
            these_p_state[:] = np.nan
            for tt, this_edge in enumerate(TRIAL_BINS[:-1]):
                if trials_df[trial_ind+(this_edge):trial_ind+(TRIAL_BINS[tt+1])].shape[0] == trial_bin_size:
                    these_states = trials_df.loc[trial_ind+this_edge:(trial_ind+TRIAL_BINS[tt+1])-1, 'p_engaged'].values
                    these_p_state[tt] = np.mean(these_states)
            """
            this_block_df = pd.concat((this_block_df, pd.DataFrame(data={
                'state': np.array(these_p_state), 'trial_bin': np.array(these_trial_bins),
                'opto_switch': all_blocks,
                'opto': trials_df.loc[trial_ind, 'laser_stimulation']})), ignore_index=True)
        
    # Get mean over sessions
    eng_stim = np.mean(eng_stim)
    eng_nostim = np.mean(eng_nostim)
    switch_stim = np.mean(switch_stim)
    switch_nostim = np.mean(switch_nostim)
        
    # Add to dataframe
    state_df = pd.concat((state_df, pd.DataFrame(index=[state_df.shape[0]], data={
        'subject': subject, 'sert-cre': sert_cre,
        'eng_stim': eng_stim, 'eng_nostim': eng_nostim,
        'switch_stim': switch_stim, 'switch_nostim': switch_nostim})))

    this_state = this_probe_df[this_probe_df['opto'] == 1].groupby('trial').mean()['state'].values * 100
    probe_df = pd.concat((probe_df, pd.DataFrame(data={
        'p_state': this_state, 'p_state_bl': this_state - np.mean(this_state[:SINGLE_TRIALS[0]]),
        'trial': np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[1]+1), 'sert-cre': sert_cre, 'subject': subject,
        'opto': 1})))
    this_state = this_probe_df[this_probe_df['opto'] == 0].groupby('trial').mean()['state'].values * 100
    probe_df = pd.concat((probe_df, pd.DataFrame(data={
        'p_state': this_state, 'p_state_bl': this_state - np.mean(this_state[:SINGLE_TRIALS[0]]),
        'trial': np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[1]+1), 'sert-cre': sert_cre, 'subject': subject,
        'opto': 0})))
    
    this_state = this_block_single_df[this_block_single_df['opto'] == 1].groupby('trial').mean()['state'].values * 100
    block_single_df = pd.concat((block_single_df, pd.DataFrame(data={
        'p_state': this_state, 'p_state_bl': this_state - np.mean(this_state[:SINGLE_TRIALS[0]]),
        'trial': np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[1]+1), 'sert-cre': sert_cre, 'subject': subject,
        'opto': 1})))
    this_state = this_block_single_df[this_block_single_df['opto'] == 0].groupby('trial').mean()['state'].values * 100
    block_single_df = pd.concat((block_single_df, pd.DataFrame(data={
        'p_state': this_state, 'p_state_bl': this_state - np.mean(this_state[:SINGLE_TRIALS[0]]),
        'trial': np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[1]+1), 'sert-cre': sert_cre, 'subject': subject,
        'opto': 0})))
        
    this_state = this_block_df[this_block_df['opto'] == 1].groupby('trial_bin').mean(numeric_only=True)['state'] * 100
    block_df = pd.concat((block_df, pd.DataFrame(data={
        'p_state': this_state,
        'p_state_bl': this_state - np.mean(this_state.values[:np.sum(trial_win_labels < -5)]),
        'trial': trial_win_labels, 'subject': subject,
        'sert-cre': sert_cre,
        'opto': 1})))
    this_state = this_block_df[this_block_df['opto'] == 0].groupby('trial_bin').mean(numeric_only=True)['state'] * 100
    block_df = pd.concat((block_df, pd.DataFrame(data={
        'p_state': this_state,
        'p_state_bl': this_state - np.mean(this_state.values[:np.sum(trial_win_labels < -5)]),
        'trial': trial_win_labels, 'subject': subject,
        'sert-cre': sert_cre,
        'opto': 0})))
    
#stats.ttest_rel(state_df['switch_stim'].values[~np.isnan(state_df['switch_stim'].values)],
#                state_df['switch_nostim'].values[~np.isnan(state_df['switch_nostim'].values)])
    
# %% Plot
    
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

sns.lineplot(data=block_df, x='trial', y='p_state_bl', hue='opto', ax=ax1,
             hue_order=[1, 0], palette=[colors['stim'], colors['no-stim']],
             errorbar='se', err_kws={'lw': 0})
ax1.plot(ax1.get_xlim(), [0, 0], ls='--', color='grey', lw=0.5)
ax1.set(ylabel='P(state)', yticks=[-15, -10, -5, 0, 5, 10], xticks=[-20, 0, 20, 40, 60, 80],
        xlabel='Trials since 5-HT start')

leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['5-HT', 'No 5-HT']
ax1.legend(leg_handles, leg_labels, prop={'size': 5}, bbox_to_anchor=[0.52, 1.1], frameon=False)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'reaction_time_HMM.pdf'))