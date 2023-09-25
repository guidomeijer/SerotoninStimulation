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
RANDOM_TIMES = 'jitter'  # spont (spontaneous) or jitter (jittered times during stim period)
PRE_TIME = 1
POST_TIME = 4
PLOT = False
ORIG_BIN_SIZE = 0.1  # original bin size
BIN_SIZE = 0.3  # binning to apply for this analysis
BIN_SHIFT = 0.1
PRE_TIME = 1
POST_TIME = 4
N_STATES_SELECT = 'global'

# Plotting
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Correlation matrices', f'{RANDOM_TIMES}')

# Get all files
all_files = glob(join(save_path, 'HMM', 'Passive', f'{RANDOM_TIMES}', 'prob_mat', '*.npy'))

# Get all recordings
all_rec = np.unique([split(i)[1][:28] for i in all_files])

# Create time axis
bin_centers = np.arange(-PRE_TIME + BIN_SIZE/2, (POST_TIME - BIN_SIZE/2) + BIN_SHIFT, BIN_SHIFT)
time_ax = np.arange(-PRE_TIME + ORIG_BIN_SIZE/2, (POST_TIME +
                    ORIG_BIN_SIZE)-ORIG_BIN_SIZE/2, ORIG_BIN_SIZE)

corr_df = pd.DataFrame()
for i, this_rec in enumerate(all_rec):
    print(f'Processing recording {i} of {len(all_rec)}')

    # Get all brain regions simultaneously recorded in this recording session
    rec_region_paths = glob(
        join(save_path, 'HMM', 'Passive', f'{RANDOM_TIMES}', 'prob_mat', f'{this_rec[:20]}*'))
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
            corr_mats = np.empty((rec_region[region1].shape[2], rec_region[region2].shape[2],
                                  bin_centers.shape[0]))
            corr_mean, corr_max = np.empty(bin_centers.shape[0]), np.empty(bin_centers.shape[0])
            corr_min = np.empty(bin_centers.shape[0])
            corr_permut = np.empty(bin_centers.shape[0])
            for tb, bin_center in enumerate(bin_centers):

                # Correlate each state with each other state
                this_corr_mat = np.empty(
                    (rec_region[region1].shape[2], rec_region[region2].shape[2]))
                for state1 in range(rec_region[region1].shape[2]):
                    for state2 in range(rec_region[region2].shape[2]):

                        # Bin state probabilities
                        r1s1 = np.mean(rec_region[region1][
                            rec_region[region1].shape[0]//2:,
                            (time_ax >= bin_center - BIN_SIZE/2) & (time_ax <= bin_center + BIN_SIZE/2),
                            state1],
                            axis=1)
                        r2s2 = np.mean(rec_region[region2][
                            rec_region[region2].shape[0]//2:,
                            (time_ax >= bin_center - BIN_SIZE/2) & (time_ax <= bin_center + BIN_SIZE/2),
                            state2],
                            axis=1)

                        # Correlate
                        corr_mats[state1, state2, tb] = pearsonr(r1s1, r2s2)[0]

                # Get mean over entire correlation matrix
                corr_mean[tb] = np.mean(corr_mats[:, :, tb])

                # Get highest state pairs
                n_states = np.max([rec_region[region1].shape[2], rec_region[region2].shape[2]])
                corr_max[tb] = np.mean(np.sort(np.concatenate(corr_mats[:, :, tb]))[-n_states:])
                corr_min[tb] = np.mean(np.sort(np.concatenate(corr_mats[:, :, tb]))[:n_states])

            # Add to dataframe
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'time': bin_centers, 'r_mean': corr_mean, 'r_max': corr_max, 'r_min': corr_min,
                'region1': region1, 'region2': region2, 'opto': 1,
                'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                'subject': subject, 'date': date, 'probe': probe})))

            # Plot an example correlation matrix
            if PLOT:
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
                pos = ax1.imshow(corr_mats[:, :, 15], clim=[-0.4, 0.4], cmap='coolwarm')
                f.colorbar(pos, ax=ax1)
                ax1.set(ylabel=f'States in {region1}', xlabel=f'States in {region2}',
                        title=f'Time = {np.round(time_ax[15], 2)}s',
                        yticks=np.arange(rec_region[region1].shape[2]),
                        yticklabels=np.arange(1, rec_region[region1].shape[2]+1),
                        xticks=np.arange(rec_region[region2].shape[2]),
                        xticklabels=np.arange(1, rec_region[region2].shape[2]+1))

                ax2.plot(bin_centers, corr_mean, color='k')
                ax2.set(ylabel='Correlation (r)', xlabel='Time from stimulation start (s)',
                        xticks=[-1, 0, 1, 2, 3, 4])

                sns.despine(trim=True)
                plt.tight_layout()
                plt.savefig(join(fig_path, f'{subject}_{date}_{region1}-{region2}.jpg'), dpi=600)
                plt.close(f)

    # Do the same for the random onset times null condition
    # Get all brain regions simultaneously recorded in this recording session
    rec_region_paths = glob(
        join(save_path, 'HMM', 'Passive', f'{RANDOM_TIMES}', f'{this_rec[:20]}*'))
    rec_region = dict()
    for ii in range(len(rec_region_paths)):
        rec_region[split(rec_region_paths[ii])[1][29:-4]] = np.load(join(rec_region_paths[ii]))

    # Correlate each area with each other area
    for r1, region1 in enumerate(rec_region.keys()):
        for r2, region2 in enumerate(list(rec_region.keys())[r1:]):
            if region1 == region2:
                continue

            # Loop over timebins
            corr_mats = np.empty((rec_region[region1].shape[2], rec_region[region2].shape[2],
                                  bin_centers.shape[0]))
            corr_mean, corr_max = np.empty(bin_centers.shape[0]), np.empty(bin_centers.shape[0])
            corr_min = np.empty(bin_centers.shape[0])
            corr_permut = np.empty(bin_centers.shape[0])
            for tb, bin_center in enumerate(bin_centers):

                # Correlate each state with each other state
                this_corr_mat = np.empty(
                    (rec_region[region1].shape[2], rec_region[region2].shape[2]))
                for state1 in range(rec_region[region1].shape[2]):
                    for state2 in range(rec_region[region2].shape[2]):

                        # Bin state probabilities
                        r1s1 = np.mean(rec_region[region1][
                            :rec_region[region1].shape[0]//2,
                            (time_ax >= bin_center - BIN_SIZE/2) & (time_ax <= bin_center + BIN_SIZE/2),
                            state1],
                            axis=1)
                        r2s2 = np.mean(rec_region[region2][
                            :rec_region[region2].shape[0]//2,
                            (time_ax >= bin_center - BIN_SIZE/2) & (time_ax <= bin_center + BIN_SIZE/2),
                            state2],
                            axis=1)

                        # Correlate
                        corr_mats[state1, state2, tb] = pearsonr(r1s1, r2s2)[0]

                # Get mean over entire correlation matrix
                corr_mean[tb] = np.mean(corr_mats[:, :, tb])

                # Get highest state pairs
                n_states = np.max([rec_region[region1].shape[2], rec_region[region2].shape[2]])
                corr_max[tb] = np.mean(np.sort(np.concatenate(corr_mats[:, :, tb]))[-n_states*2:])
                corr_min[tb] = np.mean(np.sort(np.concatenate(corr_mats[:, :, tb]))[:n_states])

            # Add to dataframe
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'time': bin_centers, 'r_mean': corr_mean, 'r_max': corr_max, 'r_min': corr_min,
                'region1': region1, 'region2': region2, 'opto': 0,
                'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                'subject': subject, 'date': date, 'probe': probe})))

            # Plot an example correlation matrix
            if PLOT:
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
                pos = ax1.imshow(corr_mats[:, :, 15], clim=[-0.4, 0.4], cmap='coolwarm')
                f.colorbar(pos, ax=ax1)
                ax1.set(ylabel=f'States in {region1}', xlabel=f'States in {region2}',
                        title=f'Time = {np.round(time_ax[15], 2)}s',
                        yticks=np.arange(rec_region[region1].shape[2]),
                        yticklabels=np.arange(1, rec_region[region1].shape[2]+1),
                        xticks=np.arange(rec_region[region2].shape[2]),
                        xticklabels=np.arange(1, rec_region[region2].shape[2]+1))

                ax2.plot(bin_centers, corr_mean, color='k')
                ax2.set(ylabel='Correlation (r)', xlabel='Time from stimulation start (s)',
                        xticks=[-1, 0, 1, 2, 3, 4])

                sns.despine(trim=True)
                plt.tight_layout()
                plt.savefig(
                    join(fig_path, f'{subject}_{date}_{region1}-{region2}_null.jpg'), dpi=600)
                plt.close(f)

    # Save output
    corr_df.to_csv(join(save_path, f'state_correlation_{RANDOM_TIMES}_passive.csv'))
