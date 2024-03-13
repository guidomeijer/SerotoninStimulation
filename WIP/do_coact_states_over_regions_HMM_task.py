# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:11:41 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from scipy.stats import pearsonr
from os.path import join, split
import matplotlib.pyplot as plt
from stim_functions import paths, figure_style, N_STATES

# Settings
CMAP = 'Set2'
RANDOM_TIMES = 'jitter'  # spont (spontaneous) or jitter (jittered times during stim period)
PRE_TIME = 1
POST_TIME = 4
PLOT = False
ORIG_BIN_SIZE = 0.1  # original bin size
PRE_TIME = 1
POST_TIME = 3
N_STATES_SELECT = 'global'

# Plotting
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()

# Get all files
all_files = glob(join(save_path, 'HMM', 'Task', f'{N_STATES_SELECT}', 'prob_mat', '*.npy'))

# Get all recordings
all_rec = np.unique([split(i)[1][:20] for i in all_files])

# Create time axis
time_ax = np.arange(-PRE_TIME + ORIG_BIN_SIZE/2, (POST_TIME +
                    ORIG_BIN_SIZE)-ORIG_BIN_SIZE/2, ORIG_BIN_SIZE)

corr_df, coact_df = pd.DataFrame(), pd.DataFrame()
for i, this_rec in enumerate(all_rec):
    print(f'Processing recording {i} of {len(all_rec)}')

    # Get all brain regions simultaneously recorded in this recording session
    rec_region_paths = glob(
        join(save_path, 'HMM', 'Task', f'{N_STATES_SELECT}', 'state_mat', f'{this_rec[:20]}*'))
    rec_region = dict()
    for ii in range(len(rec_region_paths)):
        rec_region[split(rec_region_paths[ii])[1][29:-4]] = np.load(join(rec_region_paths[ii]))

    # Get recording data
    subject = split(rec_region_paths[0])[1][:9]
    date = split(rec_region_paths[0])[1][10:20]

    # Correlate each area with each other area
    for r1, region1 in enumerate(rec_region.keys()):
        for r2, region2 in enumerate(list(rec_region.keys())[r1:]):
            if region1 == region2:
                continue

            # Loop over timebins
            coact_mats = np.empty((N_STATES, N_STATES, time_ax.shape[0]))
            coact_mats_null = np.empty((N_STATES, N_STATES, time_ax.shape[0]))
            coact_mean, coact_max = np.empty(time_ax.shape[0]), np.empty(time_ax.shape[0])
            coact_mean_null, coact_max_null = np.empty(time_ax.shape[0]), np.empty(time_ax.shape[0])
            
            for tb, bin_center in enumerate(time_ax):

                # Get coactivation of states
                for state1 in range(N_STATES):
                    for state2 in range(N_STATES):

                        # Calculate jaccard similarity 
                        reg1_states = rec_region[region1][rec_region[region1].shape[0]//2:, tb]
                        reg2_states = rec_region[region2][rec_region[region2].shape[0]//2:, tb]
                        intersection = np.logical_and(reg1_states == state1,
                                                      reg2_states == state2)
                        union = np.logical_or(reg1_states == state1,
                                              reg2_states == state2)
                        if union.sum() == 0:
                            coact_mats[state1, state2, tb] = 0
                        else:
                            coact_mats[state1, state2, tb] = intersection.sum() / float(union.sum())
                            
                        # No stim condition
                        reg1_states = rec_region[region1][:rec_region[region1].shape[0]//2, tb]
                        reg2_states = rec_region[region2][:rec_region[region2].shape[0]//2, tb]
                        intersection = np.logical_and(reg1_states == state1,
                                                      reg2_states == state2)
                        union = np.logical_or(reg1_states == state1,
                                              reg2_states == state2)
                        if union.sum() == 0:
                            coact_mats_null[state1, state2, tb] = 0
                        else:
                            coact_mats_null[state1, state2, tb] = intersection.sum() / float(union.sum())
                        
                # Get mean over entire correlation matrix
                coact_mean[tb] = np.mean(coact_mats[:, :, tb])
                coact_mean_null[tb] = np.mean(coact_mats_null[:, :, tb])
                
                # Get max
                coact_max[tb] = np.max(coact_mats[:, :, tb])
                coact_max_null[tb] = np.max(coact_mats_null[:, :, tb])


            # Add to dataframe
            for state1 in range(N_STATES):
                for state2 in range(N_STATES):
                    coact_df = pd.concat((coact_df, pd.DataFrame(data={
                        'time': time_ax, 'region1': region1, 'region2': region2, 'state1': state1,
                        'state2': state2, 'statepair': f'{state1}-{state2}',
                        'coact': coact_mats[state1, state2, :],
                        'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                        'subject': subject, 'date': date, 'opto': 1})))
                    coact_df = pd.concat((coact_df, pd.DataFrame(data={
                        'time': time_ax, 'region1': region1, 'region2': region2, 'state1': state1,
                        'state2': state2, 'statepair': f'{state1}-{state2}',
                        'coact': coact_mats_null[state1, state2, :],
                        'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                        'subject': subject, 'date': date, 'opto': 0})))
        
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'time': time_ax, 'r_mean': coact_mean, 'r_max': coact_max,
                'region1': region1, 'region2': region2, 'opto': 1,
                'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                'subject': subject, 'date': date})))
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'time': time_ax, 'r_mean': coact_mean_null, 'r_max': coact_max_null,
                'region1': region1, 'region2': region2, 'opto': 0,
                'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                'subject': subject, 'date': date})))

    # Save output
    corr_df.to_csv(join(save_path, 'state_coactivation_mean_task.csv'))
    coact_df.to_csv(join(save_path, 'state_coactivation_task.csv'))
