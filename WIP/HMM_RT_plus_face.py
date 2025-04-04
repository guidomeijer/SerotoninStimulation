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
import ssm
from stim_functions import (paths, query_opto_sessions, load_trials, init_one,
                            figure_style, load_subjects, behavioral_criterion, get_dlc_XYs,
                            get_pupil_diameter)
np.random.seed(0)

# Settings
WIN_SEC = 0.5
SINGLE_TRIALS = [5, 30]
WIN_STARTS = np.arange(-20, 70) 
WIN_SIZE = 15
PLOT_SESSIONS = False
trial_win_labels = WIN_STARTS + (WIN_SIZE/2)

# Paths
f_path, save_path = paths()
fig_path = path.join(f_path, path.split(path.dirname(path.realpath(__file__)))[-1])

# Initialize
one = init_one()
colors, dpi = figure_style()
_, data_path = paths()
subjects = load_subjects()

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
        trials_df['reaction_times'] = trials_df['feedback_times'] - trials_df['goCue_times']
        trials_df = trials_df[~np.isnan(trials_df['reaction_times'])]
        trials_df['rt_log'] = np.log10(trials_df['reaction_times']) + 2 # Log transform
        
        # Get DLC points
        try:
            video_times, XYs = get_dlc_XYs(one, eid)
        except Exception:
            print('No DLC found, skipping')
            continue
        if video_times is None:
            continue
        if np.abs(video_times.shape[0] - XYs['pupil_left_r'].shape[0]) > 10000:
            print('Timestamp mismatch, skipping..')
            continue
        
        # Get pupil diameter, whisking, and sniffing
        pupil = get_pupil_diameter(XYs)
        whisking = one.load_dataset(eid, dataset='leftCamera.ROIMotionEnergy.npy')
        sniffing = np.sqrt(np.sum(np.diff(XYs['nose_tip'], axis=0)**2, axis=1))
        sniffing = np.concatenate((sniffing, [0]))
        
        # Remove dropped frames
        if video_times.shape[0] > pupil.shape[0]:
            video_times = video_times[:pupil.shape[0]]
        elif pupil.shape[0] > video_times.shape[0]:
            pupil = pupil[:video_times.shape[0]]
            whisking = whisking[:video_times.shape[0]]
            sniffing = sniffing[:video_times.shape[0]]
        
        # Get mean of facial features for each trial onset
        for ii, trial_onset in enumerate(trials_df['goCue_times']):
            trials_df.loc[ii, 'pupil'] = np.nanmean(pupil[
                (video_times >= trial_onset) & (video_times <= trial_onset + WIN_SEC)])
            trials_df.loc[ii, 'whisking'] = np.nanmean(whisking[
                (video_times >= trial_onset) & (video_times <= trial_onset + WIN_SEC)])
            trials_df.loc[ii, 'sniffing'] = np.nanmean(sniffing[
                (video_times >= trial_onset) & (video_times <= trial_onset + WIN_SEC)])
            
        # Drop trials without pupil size
        trials_df = trials_df[~np.isnan(trials_df['pupil']) & ~np.isnan(trials_df['reaction_times'])]
            
        # Fit HMM
        input_arr = trials_df[['rt_log', 'pupil', 'whisking', 'sniffing']].to_numpy()       
        simple_hmm = ssm.HMM(3, input_arr.shape[1], observations='gaussian')
        lls = simple_hmm.fit(input_arr, method='em', transitions='sticky')
        predicted_states = simple_hmm.most_likely_states(input_arr)
        post_prob = simple_hmm.filter(input_arr)
        
        # Get mean per state
        mean_rt = [np.mean(trials_df['reaction_times'].values[predicted_states == 0]),
                   np.mean(trials_df['reaction_times'].values[predicted_states == 1]),
                   np.mean(trials_df['reaction_times'].values[predicted_states == 2])]
        mean_whisking = [np.mean(trials_df['whisking'].values[predicted_states == 0]),
                         np.mean(trials_df['whisking'].values[predicted_states == 1]),
                         np.mean(trials_df['whisking'].values[predicted_states == 2])]
        
        # Determine the engaged state as the one with the lowest RT
        # From the two remaining states the high whisking one is the exploratory state
        # engaged state = 0, exploratory state = 1, disengaged state = 2
        low_rt_state = np.argmin(mean_rt)
        mean_whisking[low_rt_state] = 0
        whisk_state = np.argmax(mean_whisking)
        diseng_state = 3 - (low_rt_state + whisk_state) # disengaged state is the remaining state
        state_ind = np.array([low_rt_state, whisk_state, diseng_state])
        state_name = ['engaged', 'exploratory', 'disengaged']
        final_states = np.ones(predicted_states.shape[0]).astype(int) * 10
        for kk in range(state_ind.shape[0]):
            final_states[predicted_states == kk] = state_ind[kk]
            trials_df[f'p_{state_name[kk]}'] = post_prob[:, state_ind[kk]]
        trials_df['state'] = final_states
    
        # Get state transitions
        trials_df['state_switch'] = np.concatenate((
            [0], (np.diff(predicted_states) != 0).astype(int)))
        
        # Save to disk
        date = str(one.get_details(eid)['date'])
        trials_df.to_csv(path.join(save_path, 'HMM', 'RTplusFace', f'{subject}_{date}.csv'))

