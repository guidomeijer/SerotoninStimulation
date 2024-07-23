# -*- coding: utf-8 -*-
"""
Strategy models module

A set of functions that define each strategy model
Each function takes a Pandas dataframe, where each row is the data for one trial, 
up to the current trial

The column names in the dataframe are used to find the values 

Created on Tue Aug 23 12:42:13 2022

@author: Mark Humphries
"""


### template for strategy models
# def <strategy_name>(rows)
#    nTrials = len(rows)
#    trial_type = "null"   # default trial type unless assigned "success" or "failure"
#    if <conditions for strategy model to be success>
#       trial_type = "success"
#    elif <further success conditions if needed>
#       trial_type = "success"
#    else
#       trial_type = "failure"
#
#    return trial_type


def go_cued(rows):
    # checks if the subject chose the cued option on this trial
    nTrials = len(rows)
     # "at" selects the value at the row/column location in the dataframe
    if rows.at[nTrials-1,'Choice'] == rows.at[nTrials-1,'CuePosition']:      # check the current trial's choice
        trial_type = "success"
    else:
        trial_type = "failure"
    return trial_type


def repeat_choice(rows):
    # checks if the subject made the same choice on this trial as the previous one
    nTrials = len(rows)
     # "at" selects the value at the row/column location in the dataframe
    if nTrials == 1:
         trial_type = "null" # undefined on first trial
    elif nTrials > 1 and rows.at[nTrials-1,'Choice'] == rows.at[nTrials-2,'Choice']:      # check the current trial's choice
        trial_type = "success"
    else:
        trial_type = "failure"
    return trial_type


def integration_window(rows):
    # checks if the subject made the choice that they most made in the last 10 trials
    nWindow = 12
    nTrials = len(rows)
    if nTrials <= nWindow:
         trial_type = "null" # undefined on first 10 trials
    elif rows.at[nTrials-1,'Choice'] == rows.loc[nTrials-(nWindow+1):nTrials-2, 'Choice'].value_counts().index[0]:
        trial_type = "success"
    else:
        trial_type = "failure"
    return trial_type


def win_stay_lose_shift(rows):
    # checks if the subject made the same spatial choice on this trial after being rewarded on the previous one
    trial_type = "null"   # default is that trial does not meet criterion for win-stay
    
    nTrials = len(rows)
     # "at" selects the value at the row/column location in the dataframe
     # check that the previous trial was rewarded ('win')
    if nTrials > 1:
        if ((rows.at[nTrials-2,'Reward'] == "yes") and (rows.at[nTrials-1,'Choice'] == rows.at[nTrials-2,'Choice'])
            or (rows.at[nTrials-2,'Reward'] == "no") and (rows.at[nTrials-1,'Choice'] != rows.at[nTrials-2,'Choice'])): 
            trial_type = "success"
        else:
            trial_type = "failure"
            
    return trial_type
    

def lose_shift(rows):
    # checks if the subject shifted spatial choice on this trial after not being rewarded on the previous one
    trial_type = "null"   # default is that trial does not meet criterion for lose-shift
    
    nTrials = len(rows)
     # "at" selects the value at the row/column location in the dataframe
     # check that the previous trial was not rewarded ('lose')
    if nTrials > 1 and rows.at[nTrials-2,'Reward'] == "no":   
        # now check if the subject shifted their spatial choice
        if rows.at[nTrials-1,'Choice'] != rows.at[nTrials-2,'Choice']: 
            trial_type = "success"
        else:
            trial_type = "failure"
            
    return trial_type


def win_stay(rows):
    # checks if the subject made the same spatial choice on this trial after being rewarded on the previous one
    trial_type = "null"   # default is that trial does not meet criterion for win-stay
    
    nTrials = len(rows)
     # "at" selects the value at the row/column location in the dataframe
     # check that the previous trial was rewarded ('win')
    if nTrials > 1 and rows.at[nTrials-2,'Reward'] == "yes":   
        # now check if the subject stayed with the same spatial choice
        if rows.at[nTrials-1,'Choice'] == rows.at[nTrials-2,'Choice']: 
            trial_type = "success"
        else:
            trial_type = "failure"
            
    return trial_type


def win_stay_cued(rows):
    # checks if the subject made the same cue-driven choice on this trial after being rewarded on the previous one
    trial_type = "null"   # default is that this trial does not meet criterion for win-stay
    
    nTrials = len(rows)
     # "at" selects the value at the row/column location in the dataframe
     # check that the previous trial was rewarded ('win')
    if nTrials > 1 and rows.at[nTrials-2,'Reward'] == "yes":   
        # now check if the subject stayed with the cued-based choice
        if rows.at[nTrials-2,'Choice'] == rows.at[nTrials-2,'CuePosition'] and rows.at[nTrials-1,'Choice'] == rows.at[nTrials-1,'CuePosition']:
            trial_type = "success"  # made the same cued choice
        elif rows.at[nTrials-2,'Choice'] != rows.at[nTrials-2,'CuePosition'] and rows.at[nTrials-1,'Choice'] != rows.at[nTrials-1,'CuePosition']:
            trial_type = "success" # made the same uncued choice
        else:
            trial_type = "failure"
            
    return trial_type


def lose_shift_cued(rows):
    # checks if the subject shifted their cue-based choice on this trial after not being rewarded on the previous one
    trial_type = "null"   # default is that this trial does not meet criterion for win-stay
    
    nTrials = len(rows)
     # "at" selects the value at the row/column location in the dataframe
     # check that the previous trial was not rewarded ('lose')
    if nTrials > 1 and rows.at[nTrials-2,'Reward'] == "no":   
        # now check if the subject shifted their cued-based choice
        if rows.at[nTrials-2,'Choice'] == rows.at[nTrials-2,'CuePosition'] and rows.at[nTrials-1,'Choice'] != rows.at[nTrials-1,'CuePosition']:
            trial_type = "success"  # shifted from cued to uncued choice
        elif rows.at[nTrials-2,'Choice'] != rows.at[nTrials-2,'CuePosition'] and rows.at[nTrials-1,'Choice'] == rows.at[nTrials-1,'CuePosition']:
            trial_type = "success" # shifted from uncued to cued chpice
        else:
            trial_type = "failure"
            
    return trial_type