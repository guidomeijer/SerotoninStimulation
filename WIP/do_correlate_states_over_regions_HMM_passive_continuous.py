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
from stim_functions import paths, figure_style, init_one, load_passive_opto_times
one = init_one()

# Settings
CMAP = 'Set2'
PRE_TIME = 1
POST_TIME = 4
PLOT = False
ORIG_BIN_SIZE = 0.1  # original bin size
BIN_SIZE = 0.3  # binning to apply for this analysis
BIN_SHIFT = 0.1
PRE_TIME = 1
POST_TIME = 4
N_STATES_SELECT = 'global'
RANDOM_TIMES = 'spont'

# Plotting
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Correlation matrices', f'continuous_{RANDOM_TIMES}')

# Get all files
all_files = glob(join(save_path, 'HMM', 'PassiveContinuous', '*prob.npy'))

# Get all recordings
all_rec = np.unique([split(i)[1][:20] for i in all_files])

# Create time axis
bin_centers = np.arange(-PRE_TIME + BIN_SIZE/2, (POST_TIME - BIN_SIZE/2) + BIN_SHIFT, BIN_SHIFT)

corr_df = pd.DataFrame()
for i, this_rec in enumerate(all_rec):
    print(f'Processing recording {i} of {len(all_rec)}')

    # Get session info
    subject = this_rec[:9]
    date = this_rec[10:20]
    eid = one.search(subject=subject, date_range=date)[0]
    
    # Get opto onset times and random times
    opto_times, _ = load_passive_opto_times(eid, one=one)
    if RANDOM_TIMES == 'jitter':
        random_times = np.sort(np.random.uniform(opto_times[0]-PRE_TIME, opto_times[-1]+POST_TIME,
                                                 size=opto_times.shape[0]))
    elif RANDOM_TIMES == 'spont':
        random_times = np.sort(np.random.uniform(opto_times[0]-290, opto_times[0]-10,
                                                 size=opto_times.shape[0]*2))
        
    # Skip long opto sessions for now
    if np.mean(np.diff(opto_times)) > 20:
        continue

    # Get all brain regions simultaneously recorded in this recording session
    rec_region_paths = glob(
        join(save_path, 'HMM', 'PassiveContinuous', f'{this_rec}*prob.npy'))
    rec_region, rec_region_null = dict(), dict()
    for ii, this_region in enumerate(rec_region_paths):
        
        # Load in HMM output
        states = np.load(this_region)
        time_ax = np.load(this_region[:-8] + 'time.npy')
        rel_time = np.arange(-PRE_TIME + ORIG_BIN_SIZE/2, (POST_TIME - ORIG_BIN_SIZE/2) + ORIG_BIN_SIZE,
                             ORIG_BIN_SIZE)

        # Stimulation onset times
        prob_mat = []
        for t, this_time in enumerate(opto_times):
            prob_mat.append(states[(time_ax >= this_time - PRE_TIME)
                                   & (time_ax <= this_time + POST_TIME)])
        prob_mat = np.transpose(np.dstack(prob_mat), (2, 0, 1))
        rec_region[split(rec_region_paths[ii])[1][29:-9]] = prob_mat

        # Random onset times
        prob_mat_null = []
        for t, this_time in enumerate(random_times):
            prob_mat_null.append(states[(time_ax >= this_time - PRE_TIME)
                                        & (time_ax <= this_time + POST_TIME)])
        prob_mat_null = np.transpose(np.dstack(prob_mat_null), (2, 0, 1))
        rec_region_null[split(rec_region_paths[ii])[1][29:-9]] = prob_mat_null

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
            corr_mats_null = np.empty((rec_region_null[region1].shape[2], rec_region_null[region2].shape[2],
                                       bin_centers.shape[0]))
            corr_mean_null, corr_max_null = np.empty(bin_centers.shape[0]), np.empty(bin_centers.shape[0])
            corr_min_null = np.empty(bin_centers.shape[0])
            for tb, bin_center in enumerate(bin_centers):

                # Correlate each state with each other state
                for state1 in range(rec_region[region1].shape[2]):
                    for state2 in range(rec_region[region2].shape[2]):

                        # Opto time onsets
                        r1s1 = np.mean(rec_region[region1][:,
                            (rel_time >= bin_center - BIN_SIZE/2) & (rel_time <= bin_center + BIN_SIZE/2),
                            state1],
                            axis=1)
                        r2s2 = np.mean(rec_region[region2][:,
                            (rel_time >= bin_center - BIN_SIZE/2) & (rel_time <= bin_center + BIN_SIZE/2),
                            state2],
                            axis=1)
                        corr_mats[state1, state2, tb] = pearsonr(r1s1, r2s2)[0]
                        
                        # Random time onsets
                        r1s1_null = np.mean(rec_region_null[region1][:,
                            (rel_time >= bin_center - BIN_SIZE/2) & (rel_time <= bin_center + BIN_SIZE/2),
                            state1],
                            axis=1)
                        r2s2_null = np.mean(rec_region_null[region2][:,
                            (rel_time >= bin_center - BIN_SIZE/2) & (rel_time <= bin_center + BIN_SIZE/2),
                            state2],
                            axis=1)
                        corr_mats_null[state1, state2, tb] = pearsonr(r1s1_null, r2s2_null)[0]

                # Get mean, max and min correlations
                corr_mean[tb] = np.mean(corr_mats[:, :, tb])
                n_states = np.max([rec_region[region1].shape[2], rec_region[region2].shape[2]])
                corr_max[tb] = np.mean(np.sort(np.concatenate(corr_mats[:, :, tb]))[-n_states:])
                corr_min[tb] = np.mean(np.sort(np.concatenate(corr_mats[:, :, tb]))[:n_states])
                
                corr_mean_null[tb] = np.mean(corr_mats_null[:, :, tb])
                n_states = np.max([rec_region_null[region1].shape[2], rec_region_null[region2].shape[2]])
                corr_max_null[tb] = np.mean(np.sort(np.concatenate(corr_mats_null[:, :, tb]))[-n_states:])
                corr_min_null[tb] = np.mean(np.sort(np.concatenate(corr_mats_null[:, :, tb]))[:n_states])
                
            # Add to dataframe
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'time': bin_centers, 'r_mean': corr_mean, 'r_max': corr_max, 'r_min': corr_min,
                'region1': region1, 'region2': region2, 'opto': 1,
                'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                'subject': subject, 'date': date})))
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'time': bin_centers, 'r_mean': corr_mean_null, 'r_max': corr_max_null, 'r_min': corr_min_null,
                'region1': region1, 'region2': region2, 'opto': 0,
                'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                'subject': subject, 'date': date})))

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

    # Save output
    corr_df.to_csv(join(save_path, f'state_correlation_{RANDOM_TIMES}_passive_cont.csv'))
