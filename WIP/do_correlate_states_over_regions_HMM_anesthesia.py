# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:11:41 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from itertools import permutations
from scipy.stats import pearsonr
from os.path import join, split
import matplotlib.pyplot as plt
from stim_functions import paths, figure_style

# Settings
CMAP = 'Set2'
PRE_TIME = 1
POST_TIME = 4
PLOT = True
ORIG_BIN_SIZE = 0.2  # original bin size
PRE_TIME = 1
POST_TIME = 4

# Plotting
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Correlation matrices', 'Anesthesia')

# Get all files
all_files = glob(join(save_path, 'HMM', 'Anesthesia', 'prob_mat', '*.npy'))

# Get all recordings
all_rec = np.unique([split(i)[1][:28] for i in all_files])

# Create time axis
time_ax = np.arange(-PRE_TIME + ORIG_BIN_SIZE/2, (POST_TIME +
                    ORIG_BIN_SIZE)-ORIG_BIN_SIZE, ORIG_BIN_SIZE)

corr_df = pd.DataFrame()
for i, this_rec in enumerate(all_rec):
    print(f'Processing recording {i} of {len(all_rec)}')

    # Get all brain regions simultaneously recorded in this recording session
    rec_region_paths = glob(
        join(save_path, 'HMM', 'Anesthesia', 'state_mat', f'{this_rec[:20]}*'))
    rec_region = dict()
    for ii in range(len(rec_region_paths)):
        rec_region[split(rec_region_paths[ii])[1][29:-4]] = np.load(join(rec_region_paths[ii]))

    # Get recording data
    subject = split(rec_region_paths[0])[1][:9]
    date = split(rec_region_paths[0])[1][10:20]
    probe = split(rec_region_paths[0])[1][21:28]

    # Correlate each area with each other area
    for r1, region1 in enumerate(rec_region.keys()):
        for r2, region2 in enumerate(list(rec_region.keys())[r1:]):
            if region1 == region2:
                continue

            # Loop over timebins
            reg_1 = rec_region[region1]
            reg_2 = rec_region[region2]
            corr_opto, corr_null = np.empty(time_ax.shape[0]), np.empty(time_ax.shape[0])
            dist_opto, dist_null = np.empty(time_ax.shape[0]), np.empty(time_ax.shape[0])
            perc_opto, perc_null = np.empty(time_ax.shape[0]), np.empty(time_ax.shape[0])
            for tb, bin_center in enumerate(time_ax):
                
                corr_opto[tb] = pearsonr(reg_1[int(reg_1.shape[0]/2):, tb],
                                         reg_2[int(reg_2.shape[0]/2):, tb])[0]
                dist_opto[tb] = np.linalg.norm(reg_1[int(reg_1.shape[0]/2):, tb]
                                               - reg_2[int(reg_2.shape[0]/2):, tb])
                perc_opto[tb] = (np.sum(reg_1[int(reg_1.shape[0]/2):, tb] == reg_2[int(reg_2.shape[0]/2):, tb]) 
                                 / int(reg_1.shape[0]/2))
                corr_null[tb] = pearsonr(reg_1[:int(reg_1.shape[0]/2), tb],
                                         reg_2[:int(reg_2.shape[0]/2), tb])[0]
                dist_null[tb] = np.linalg.norm(reg_1[:int(reg_1.shape[0]/2), tb]
                                               - reg_2[:int(reg_2.shape[0]/2), tb])
                perc_null[tb] = (np.sum(reg_1[:int(reg_1.shape[0]/2), tb] == reg_2[:int(reg_2.shape[0]/2), tb]) 
                                 / int(reg_1.shape[0]/2))
                
            # Add to dataframe
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'time': time_ax, 'r': corr_opto, 'dist': dist_opto, 'perc': perc_opto,
                'region1': region1, 'region2': region2, 'opto': 1,
                'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                'subject': subject, 'date': date, 'probe': probe})))
            
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'time': time_ax, 'r': corr_null, 'dist': dist_null, 'perc': perc_null,
                'region1': region1, 'region2': region2, 'opto': 0,
                'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                'subject': subject, 'date': date, 'probe': probe})))

    # Save output
    corr_df.to_csv(join(save_path, 'state_correlation_anesthesia.csv'))
