#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from stim_functions import (paths, figure_style, load_passive_opto_times, load_subjects,
                            query_ephys_sessions, init_one, get_dlc_XYs, smooth_interpolate_signal_sg)
one = init_one()

# Settings
OVERWRITE = False
TIME_BINS = np.arange(-0.5, 4.1, 0.1)
BIN_SIZE = 0.1  # seconds
BASELINE = [0.5, 0]  # seconds
fig_path, save_path = paths()
fig_path = join(fig_path, 'Extra plots', 'Whisking')

# Query and load data
subjects = load_subjects()

# Query ephys sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    results_df = pd.DataFrame()
else:
    results_df = pd.read_csv(join(save_path, 'whisking_passive.csv'))

for k, nickname in enumerate(np.unique(rec['subject'])):
    print(f'Processing {nickname}..')

    # Get eids
    eids = np.unique(rec.loc[rec['subject'] == nickname, 'eid'])

    whisking_df = pd.DataFrame()
    for i, eid in enumerate(eids):

        # Skip if session is already done
        if not OVERWRITE:
            if eid in results_df['eid'].values:
                print('Already processed, skipping')
                continue

        # Get session details
        ses_details = one.get_details(eid)
        date = ses_details['start_time'][:10]
        if nickname not in subjects['subject'].values:
            continue
        expression = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
        print(f'Starting {nickname}, {date}')

        # Load in laser pulse times
        try:
            opto_train_times, _ = load_passive_opto_times(eid, one=one)
        except:
            continue

        # Load in camera timestamps and DLC output
        try:
            video_times, XYs = get_dlc_XYs(one, eid)
        except:
            print('Could not load video and/or DLC data')
            continue
        if video_times is None:
            continue

        # If the difference between timestamps and video frames is too large, skip
        if np.abs(video_times.shape[0] - XYs['pupil_left_r'].shape[0]) > 10000:
            print('Timestamp mismatch, skipping..')
            continue

        # Get whisking
        roi_motion = one.load_dataset(eid, dataset='leftCamera.ROIMotionEnergy.npy')
        whisking = smooth_interpolate_signal_sg(roi_motion)
        
        # Assume frames were dropped at the end
        if video_times.shape[0] > whisking.shape[0]:
            video_times = video_times[:whisking.shape[0]]
        elif whisking.shape[0] > video_times.shape[0]:
            whisking = whisking[:video_times.shape[0]]

        # Calculate percentage change
        whisking_perc = ((whisking - np.percentile(whisking[~np.isnan(whisking)], 2))
                         / np.percentile(whisking[~np.isnan(whisking)], 2)) * 100

        # Get trial triggered baseline subtracted pupil diameter
        for t, trial_start in enumerate(opto_train_times):
            this_whisking = np.array([np.nan] * TIME_BINS.shape[0])
            baseline_subtracted = np.array([np.nan] * TIME_BINS.shape[0])
            baseline = np.nanmean(whisking_perc[(video_times > (trial_start - BASELINE[0]))
                                                  & (video_times < (trial_start - BASELINE[1]))])
            for b, time_bin in enumerate(TIME_BINS):
                this_whisking[b] = np.nanmean(whisking_perc[
                    (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                    & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))])
                baseline_subtracted[b] = np.nanmean(whisking_perc[
                    (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                    & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))]) - baseline
            whisking_df = pd.concat((whisking_df, pd.DataFrame(data={
                'diameter': this_whisking, 'baseline_subtracted': baseline_subtracted, 'eid': eid,
                'subject': nickname, 'trial': t, 'time': TIME_BINS, 'expression': expression,
                'date': date})))

        # Add to overal dataframe
        results_df = pd.concat((results_df, pd.DataFrame(data={
            'diameter': whisking_df[whisking_df['eid'] == eid].groupby('time').mean(numeric_only=True)['diameter'],
            'baseline_subtracted': whisking_df[whisking_df['eid'] == eid].groupby('time').mean(numeric_only=True)['baseline_subtracted'],
            'time': TIME_BINS, 'subject': nickname, 'expression': expression, 'eid': eid})), ignore_index=True)

    # Save output
    results_df.to_csv(join(save_path, 'whisking_passive.csv'))

    # Plot this animal
    colors, dpi = figure_style()
    plot_ses = results_df[results_df['subject'] == nickname].reset_index(drop=True)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi)
    sns.lineplot(x='time', y='diameter', estimator=np.nanmean, data=plot_ses,
                 color='k', errorbar='se', ax=ax1, legend=None)
    ylim = ax1.get_ylim()
    ax1.add_patch(Rectangle((0, 0), 1, 500, color='royalblue', alpha=0.25, lw=0))
    ax1.set(title='%s, expression: %d' % (nickname, expression),
            ylabel='Pupil size (%)', xlabel='Time (s)', ylim=ylim,
            xticks=np.arange(-1, TIME_BINS[-1]+0.1))

    sns.lineplot(x='time', y='baseline_subtracted', estimator=np.nanmean, data=plot_ses,
                 color='k', errorbar='se', ax=ax2, legend=None)
    ylim = ax2.get_ylim()
    ax2.add_patch(Rectangle((0, -100), 1, 200, color='royalblue', alpha=0.25, lw=0))
    ax2.set(title='%s, expression: %d' % (nickname, expression), ylim=ylim,
            ylabel='Baseline subtracted\npupil size (%)', xlabel='Time (s)',
            xticks=np.arange(-1, TIME_BINS[-1]+1))

    plt.tight_layout()
    sns.despine(trim=True)

    plt.savefig(join(fig_path, f'{nickname}_whisking_passive.jpg'), dpi=600)
    plt.close(f)


