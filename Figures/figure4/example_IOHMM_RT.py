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
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from IOHMM import UnSupervisedIOHMM
from IOHMM import OLS, CrossEntropyMNL
from stim_functions import (paths, query_opto_sessions, load_trials, init_one,
                            figure_style, load_subjects, behavioral_criterion)

# Settings
EID = '998b3667-bd20-4d4d-b840-953b868ea90b'
TRIALS = [750, 900]

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

# Load trials
trials_df = load_trials(EID, laser_stimulation=True)
   
# Get reaction times 
trials_df['rt'] = trials_df['feedback_times'] - trials_df['goCue_times']
trials_df = trials_df[~np.isnan(trials_df['rt'])]

# Log transform and then z-score reaction times per contrast
trials_df['rt'] = np.log10(trials_df['rt'])
trials_df['abs_contrast'] = np.abs(trials_df['signed_contrast'])
for ii, this_contrast in enumerate(np.unique(trials_df['abs_contrast'])):
    trials_df.loc[trials_df['abs_contrast'] == this_contrast, 'rt'] = zscore(
        trials_df.loc[trials_df['abs_contrast'] == this_contrast, 'rt'])
            
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

# Remove probe trials
trials_df.loc[(trials_df['laser_probability'] == 0.25)
              & (trials_df['laser_stimulation'] == 1), 'laser_stimulation'] = 0
trials_df.loc[(trials_df['laser_probability'] == 0.75)
              & (trials_df['laser_stimulation'] == 0), 'laser_stimulation'] = 1

        
# %% Plot session
f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
ax1.scatter(np.arange(trials_df['rt'].shape[0]), trials_df['rt'], color='k', s=2, zorder=1)
ax1.add_patch(Rectangle((800, -1), 49, 4, color='royalblue', alpha=0.25, lw=0))
ax2 = ax1.twinx()
ax2.plot(np.arange(trials_df['rt'].shape[0]), trials_df['p_engaged'], alpha=0.5, color='tab:red')
ax1.set(ylim=[-1, 3], xlabel='Trials', ylabel='Z-scored reaction time (std)', xlim=TRIALS)
ax2.set(yticks=[0, 1], yticklabels=[0, 1])
ax2.set_ylabel('P(engaged)', rotation=270, color='tab:red')

sns.despine(trim=True, right=False)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'example_session_IOHMM.pdf'))

    
            
