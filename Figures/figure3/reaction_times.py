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
from stim_functions import (paths, query_opto_sessions, load_trials, init_one,
                            figure_style, load_subjects, behavioral_criterion)

# Settings
MIN_SES = 2

# Paths
f_path, save_path = paths()
fig_path = path.join(f_path, path.split(path.dirname(path.realpath(__file__)))[-1])

# Initialize
one = init_one()
_, data_path = paths()
subjects = load_subjects()
colors, dpi = figure_style()

# Loop over subjects
rt_df = pd.DataFrame()
for i, subject in enumerate(subjects['subject']):
    print(f'{subject} ({i} of {subjects.shape[0]})')

    # Query sessions
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    if sert_cre == 0:
        continue
    eids = query_opto_sessions(subject, include_ephys=True, one=one)
    eids = behavioral_criterion(eids, verbose=False, one=one)
    if len(eids) < MIN_SES:
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
       
        # Remove probe trials
        trials_df.loc[(trials_df['laser_probability'] == 0.25)
                      & (trials_df['laser_stimulation'] == 1), 'laser_stimulation'] = 0
        trials_df.loc[(trials_df['laser_probability'] == 0.75)
                      & (trials_df['laser_stimulation'] == 0), 'laser_stimulation'] = 1
                                
        # Get reaction time medians
        rt_opto_median.append(trials_df.loc[trials_df['laser_stimulation'] == 1, 'rt'].median())
        rt_no_opto_median.append(trials_df.loc[trials_df['laser_stimulation'] == 0, 'rt'].median())
             
    # Add to dataframe
    rt_df = pd.concat((rt_df, pd.DataFrame(index=[rt_df.shape[0]], data={
        'subject': subject, 'sert-cre': sert_cre,
        'rt_opto': np.mean(rt_opto_median), 'rt_no_opto': np.mean(rt_no_opto_median)})))

# %%

f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1.75), dpi=dpi)
for i in rt_df.index:
    ax1.plot([0, 1], [rt_df.loc[i, 'rt_opto'], rt_df.loc[i, 'rt_opto']], marker='o', color='k',
             markersize=2)
ax1.set(xticks=[0, 1], xticklabels=['5-HT', 'No 5-HT'], ylabel='Median reaction time (s)',
        yticks=np.arange(0.3, 0.61, 0.1), xlim=[-0.2, 1.2])
ax1.text(0.5, 0.58, 'n.s.', ha='center', va='center')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'reaction_time_median.pdf'))

