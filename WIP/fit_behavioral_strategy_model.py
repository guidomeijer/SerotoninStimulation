# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:35:43 2024

By Guido Meijer
"""


import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stim_functions import paths, remap, query_ephys_sessions, load_trials, init_one, figure_style
one = init_one()
colors, dpi = figure_style()
_, data_path = paths()

import stim_strategymodels
from Functions.set_Beta_prior import set_priors
from Functions.update_strategy_posterior_probability import update_strategy_posterior_probability
from Functions.Summaries_of_Beta_distribution import summaries_of_Beta_Distribution
from Functions.plotSessionStructure import plotSessionStructure
from Functions.interpolate_null_trials import interpolate_null_trials


# SETTINGS
STRATEGIES = ['go_cued', 'repeat_choice', 'win_stay_lose_shift', 'integration_window',
              'lose_shift', 'win_stay', 'lose_shift_cued', 'win_stay_cued']   
PRIOR_TYPE = 'Uniform' 
DECAY_RATE = 0.9

# Define priors
[alpha0, beta0] = set_priors(PRIOR_TYPE)

# Query sessions
rec = query_ephys_sessions(n_trials=400, one=one)

all_trials_df = pd.DataFrame()
for i, subject in enumerate(np.unique(rec['subject'])):
    print(f'\nStarting {subject}')
    
    # Load in all trials of all sessions of this mouse and put in one dataframe
    all_trials = []
    for j, eid in enumerate(np.unique(rec.loc[rec['subject'] == subject, 'eid'])):
        these_trials = load_trials(eid, laser_stimulation=True)
        these_trials['session'] = j
        all_trials.append(these_trials)
    trials_df = pd.concat(all_trials).reset_index()
    
    # Disregard stim catch trials
    trials_df.loc[(trials_df['laser_probability'] == 0.25) & (trials_df['laser_stimulation'] == 1),
                  'laser_stimulation'] = 0
    trials_df.loc[(trials_df['laser_probability'] == 0.75) & (trials_df['laser_stimulation'] == 0),
                  'laser_stimulation'] = 1
    
    # Restructure trials dataframe to what the model wants
    model_trials_df = pd.DataFrame(data={
        'TrialIndex': trials_df.index,
        'SessionIndex': trials_df['session'],
        'TargetRule': trials_df['laser_stimulation'].replace({1: 'stim on', 0: 'stim off'}),
        'Choice': trials_df['choice'].replace({-1: 'right', 1: 'left'}),
        'CuePosition': trials_df['stim_side'].replace({-1: 'left', 1: 'right'}),
        'Reward': trials_df['correct'].replace({0: 'no', 1: 'yes'}),
        'RuleChangeTrials': np.concatenate(([0], (np.diff(trials_df['laser_stimulation']) != 0).astype(int))),
        'NewSessionTrials': np.concatenate(([0], (np.diff(trials_df['session']) != 0).astype(int)))
        })
    
    # Initialize variables
    Output_collection = {} 
    event_totals = {}  
    for index_strategy in range(len(STRATEGIES)):
        Output_collection[STRATEGIES[index_strategy]] =  pd.DataFrame(columns=[
            'Alpha', 'Beta', 'MAPprobability', 'Precision','Alpha_interpolated', 'Beta_interpolated',
            'MAPprobability_interpolated', 'Precision_interpolated'])  
        event_totals[STRATEGIES[index_strategy]] = {}
        event_totals[STRATEGIES[index_strategy]]['success_total'] = 0
        event_totals[STRATEGIES[index_strategy]]['failure_total'] = 0
        
    # Fit model
    for trial in range(model_trials_df.shape[0]):
        
        rows_of_data = model_trials_df.iloc[0:trial+1]    
        
        for index_strategy in range(len(STRATEGIES)):
            
           # run current strategy model on data up to current trial 
           strategy_fcn = getattr(stim_strategymodels, STRATEGIES[index_strategy])  
           trial_type = strategy_fcn(rows_of_data)   
         
           # update probability of strategy
           [event_totals[STRATEGIES[index_strategy]]['success_total'],
            event_totals[STRATEGIES[index_strategy]]['failure_total'],
            Alpha, Beta] = update_strategy_posterior_probability(
                trial_type, DECAY_RATE, event_totals[STRATEGIES[index_strategy]]['success_total'],
                event_totals[STRATEGIES[index_strategy]]['failure_total'],
                alpha0, beta0)
                   
           MAPprobability = summaries_of_Beta_Distribution(Alpha, Beta, 'MAP')
           precision = summaries_of_Beta_Distribution(Alpha, Beta, 'precision')
          
           this_trials_data= {'Alpha': Alpha, 'Beta': Beta, 'MAPprobability': MAPprobability,
                              'Precision': precision}
           if trial > 0:
               previous_trials_data = Output_collection[STRATEGIES[index_strategy]].iloc[trial-1] 
           else:
               previous_trials_data = Output_collection[STRATEGIES[index_strategy]]
           new_row_of_data = interpolate_null_trials(this_trials_data, previous_trials_data,
                                                     alpha0, beta0)
           
           # store results  - dynamically-defined dataframe...
           new_df = pd.DataFrame([new_row_of_data])  
           Output_collection[STRATEGIES[index_strategy]] = pd.concat(
               [Output_collection[STRATEGIES[index_strategy]], new_df], ignore_index=True)
           
    # Add to dataframe 
    trials_df['subject'] = subject
    for strategy in STRATEGIES:
        trials_df[strategy] = Output_collection[strategy].MAPprobability_interpolated
    
    # Add trial number of stimulated block
    trial_blocks = (trials_df['laser_stimulation'] == 0).astype(int)
    block_trans = np.concatenate(([0], np.where(np.diff(trial_blocks) != 0)[0] + 1,
                                  [trial_blocks.shape[0]]))
    for ii, trans_trial in enumerate(block_trans[:-1]):
        trials_df.loc[trans_trial:block_trans[ii + 1] - 1, 'rel_trial'] = np.arange(
            0, block_trans[ii + 1] - trans_trial)
    
    # Add to overall dataframe
    all_trials_df = pd.concat((all_trials_df, trials_df))
    
    # Save to disk
    all_trials_df.to_csv(join(data_path, 'behavioral_strategy.csv'), index=False)
    
    
    #%% plot results
    """
    f, ax1 = plt.subplots(1, 1, figsize=(3, 1.75), dpi=dpi)
    #ax1.plot(Output_collection['go_cued'].MAPprobability, linewidth=0.75, color='blue')
    #ax1.plot(Output_collection['sticky'].MAPprobability, linewidth=0.75, color='magenta')
    
    
    ax1.plot(Output_collection['lose_shift_cued'].MAPprobability_interpolated,
             linewidth=0.5, color=(1, 0.1, 0.6), label='Lose shift cued')
    ax1.plot(Output_collection['lose_shift_spatial'].MAPprobability_interpolated,
             linewidth=0.5, color=(0.8, 0.6, 0.5), label='Lose shift spatial')  
    ax1.plot(Output_collection['win_stay_spatial'].MAPprobability_interpolated,
             linewidth=0.5, color=(0.4,0.8,0.5), label='Win stay spatial')  
    ax1.plot(Output_collection['win_stay_cued'].MAPprobability_interpolated,
             linewidth=0.5, color=(0.4,0.8,0.5), label='Win stay cued')
    ax1.legend()
    
    
    no_Trials = np.size(model_trials_df.TrialIndex)

    # plotting time series of MAPprobability for Rule Strategies
    plt.figure(figsize=(10, 5))
    plt.plot(Output_collection['go_cued'].MAPprobability, linewidth=0.75)  # plots the time series
    plt.plot(Output_collection['sticky'].MAPprobability, linewidth=0.75, color=(0.4, 0.8, 0.5))  # plots the time series
    plt.plot(Output_collection['win_stay_cued'].MAPprobability, linewidth=0.75, color=(0.8,0.6,0.5))  # plots the time series
    plt.axis([0, no_Trials, 0, 1.25])  # establishes axis limits 
    plt.xlabel('Trials'), plt.ylabel('P(Strategy)')  # labelling the axis
    plt.axhline(y=0.5, color='darkgrey', linewidth=0.75, label="Chance")  # shows the line at which Chance is exceeded

    plotSessionStructure(model_trials_df)
    plt.show()       

    # plotting Precision for the same three strategies (precision identical for go_left and go_right)
    plt.figure(figsize=(10, 5))
    plt.plot(Output_collection['sticky'].Precision, linewidth=0.75)  # plots the time series
    plt.plot(Output_collection['go_cued'].Precision, linewidth=0.75, color=(0.8,0.6,0.5))  # plots the time series
    plt.xlabel('Trials'), plt.ylabel('Precision')  # labelling the axis
    plt.show()      
           

    # plotting MAP probability for some exploratory strategies - use interpolated values
    plt.figure(figsize=(10, 5))
    plt.plot(Output_collection['lose_shift_cued'].MAPprobability_interpolated, linewidth=0.75, color=(1, 0.1, 0.6))  # plots the time series
    plt.plot(Output_collection['lose_shift_spatial'].MAPprobability_interpolated, linewidth=0.75, color=(0.8, 0.6, 0.5))  # plots the time series
    plt.plot(Output_collection['win_stay_spatial'].MAPprobability_interpolated, linewidth=0.75, color=(0.4,0.8,0.5))  # plots the time series
    plt.axis([0, no_Trials, 0, 1.25])  # establishes axis limits 
    plt.xlabel('Trials'), plt.ylabel('P(Strategy)')  # labelling the axis
    plt.axhline(y=0.5, color='darkgrey', linewidth=0.75, label="Chance")  # shows the line at which Chance is exceeded

    plotSessionStructure(model_trials_df)
    plt.show()  
    
    asd
    """
    