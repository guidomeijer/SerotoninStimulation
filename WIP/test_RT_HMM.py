# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:51:24 2024 by Guido Meijer
"""

from __future__ import  division
import json
import numpy as np
import pandas as pd
from os.path import join
from IOHMM import UnSupervisedIOHMM
from IOHMM import OLS, DiscreteMNL, CrossEntropyMNL

# Settings
PATH = r'C:\Users\guido\Data\Flatiron\mainenlab\Subjects\ZFM-04811\2022-11-09\001\alf'

# Initialize IOHMM with two states
SHMM = UnSupervisedIOHMM(num_states=2, max_EM_iter=200, EM_tol=1e-6)
SHMM.set_models(model_emissions = [OLS()], 
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))
SHMM.set_inputs(covariates_initial = [], covariates_transition = [], covariates_emissions = [[]])
SHMM.set_outputs([['rt']])

# Load in trials
trials = pd.read_parquet(join(PATH, '_ibl_trials.table.pqt'))
trials['rt'] = trials['firstMovement_times'] - trials['goCue_times']
trials = trials[~np.isnan(trials['rt'])]

# Set trial dataframe to IOHMM
SHMM.set_data([trials])

# Start training
SHMM.train()

# Get posterior probabilities and most likely state per trial
post_prob = np.exp(SHMM.log_gammas[0])
predicted_states = np.array([np.argmax(i, axis=0) for i in post_prob])

# Determine the engaged state as the one with the lowest RT
# engaged state = 0, disengaged state = 1
if (np.median(trials['rt'].values[predicted_states == 0])
        > np.median(trials['rt'].values[predicted_states == 1])):
    predicted_states = np.where((predicted_states==0) | (predicted_states==1),
                                predicted_states^1, predicted_states)
trials['state'] = predicted_states

# Get state transitions
trials['state_trans'] = np.concatenate(([0], (np.diff(predicted_states) != 0).astype(int)))



