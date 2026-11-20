# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:04:34 2025

By Guido Meijer
"""

import pandas as pd
from os.path import join
from stim_functions import (paths, behavioral_criterion, load_trials, 
                            query_opto_sessions, load_subjects, init_one)
one = init_one()
subjects = load_subjects()
_, save_path = paths()
MIN_SES = 2

all_trials = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    
    # Only use sert-cre animals
    if subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0] == 0:
        continue    
    print(f'{nickname}')

    # Query sessions
    eids = query_opto_sessions(nickname, include_ephys=True, one=one)
    eids = behavioral_criterion(eids, verbose=False, one=one)
    if len(eids) < MIN_SES:
        continue

    # Get trials DataFrame
    subject_trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, one=one)
            subject_trials = pd.concat((subject_trials, these_trials))
        except:
            continue
    subject_trials['subject'] = nickname
    all_trials = pd.concat((all_trials, subject_trials), ignore_index=True)
    
# Save 
all_trials.to_csv(join(save_path, 'all_trials.csv'), index=False)