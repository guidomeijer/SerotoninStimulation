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
PLOT = False
BIN_SIZE = 0.1
PRE_TIME = 1
POST_TIME = 4

# Plotting
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Correlation matrices')

# Get all files
all_files = glob(join(save_path, 'HMM', '*.npy'))

# Get all recordings
all_rec = np.unique([split(i)[1][:28] for i in all_files])

# Create time axis
time_ax = np.arange(-PRE_TIME + BIN_SIZE/2, (POST_TIME+BIN_SIZE)-BIN_SIZE/2, BIN_SIZE)

corr_df = pd.DataFrame()
for i, this_rec in enumerate(all_rec):
    print(f'Processing recording {i} of {len(all_rec)}')
    
    # Get all brain regions simultaneously recorded in this recording session
    rec_region_paths = glob(join(save_path, 'HMM', f'{this_rec[:20]}*'))
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
                                  rec_region[region1].shape[1]))
            corr_mean = np.empty(rec_region[region1].shape[1])
            corr_permut = np.empty(rec_region[region1].shape[1])
            for tb in range(rec_region[region1].shape[1]):
                
                # Correlate each state with each other state
                this_corr_mat = np.empty((rec_region[region1].shape[2], rec_region[region2].shape[2]))
                for state1 in range(rec_region[region1].shape[2]):
                    for state2 in range(rec_region[region2].shape[2]):
                        corr_mats[state1, state2, tb] = pearsonr(rec_region[region1][:, tb, state1],
                                                                 rec_region[region2][:, tb, state2])[0]
                corr_mean[tb] = np.mean(corr_mats[:, :, tb])
            
                # From the region with the most states, drop the states with the lowest correlation
                # This is to make the matrix square
                this_mat = corr_mats[:, :, tb]
                lowest_states = np.argsort(np.max(this_mat, axis=np.argmin(this_mat.shape)))
                this_mat = np.delete(this_mat,
                                     np.argsort(lowest_states)[:np.abs(this_mat.shape[0] - this_mat.shape[1])],
                                     np.argmax(this_mat.shape))
                
                # Permute matrix to get maximal correlation along the diagonal
                all_perm = list(permutations(np.arange(this_mat.shape[0])))
                permut_diag = np.empty(len(all_perm))
                for permut in range(len(all_perm)):
                    permut_diag[permut] = np.sum(np.diag(this_mat[:, all_perm[permut]]))
                permut_mat = this_mat[:, all_perm[np.argmax(permut_diag)]]
                corr_permut[tb] = np.mean(np.diag(permut_mat))
                
            # Add to dataframe
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'time': time_ax, 'r_mean': corr_mean, 'r_permut': corr_permut,
                'region1': region1, 'region2': region2,
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
                
                ax2.plot(time_ax, corr_mean, color='k')
                ax2.set(ylabel='Correlation (r)', xlabel='Time from stimulation start (s)',
                        xticks=[-1, 0, 1, 2, 3, 4])
                
                sns.despine(trim=True)
                plt.tight_layout()
                plt.savefig(join(fig_path, f'{subject}_{date}_{region1}-{region2}.jpg'), dpi=600)
                plt.close(f)
                
    # Save output
    corr_df.to_csv(join(save_path, 'state_correlation.csv'))
                
                             
                