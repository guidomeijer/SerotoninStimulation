#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:03:49 2022
By: Guido Meijer
"""

import numpy as np
import seaborn as sns
from os.path import join
import ssm
import matplotlib.pyplot as plt
from brainbox.task.closed_loop import roc_single_event
import math
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import (figure_style, load_passive_opto_times, get_neuron_qc, remap, paths,
                                 query_ephys_sessions, get_raw_smooth_pupil_diameter, get_dlc_XYs,
                                 smooth_interpolate_signal_sg)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

K = 2    # number of discrete states
T_BEFORE = 0  # for state classification
T_AFTER = 0.5
PRE_TIME = [0.5, 0]  # for modulation index
POST_TIME = [0, 0.5]
OVERWRITE = False
FEATURES = ['nose_tip', 'paw_l', 'paw_r']

# Get path
_, save_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    state_mod_df = pd.DataFrame()
else:
    state_mod_df = pd.read_csv(join(save_path, 'mov_pupil_state_mod.csv'))
    rec = rec[~rec['pid'].isin(state_mod_df['pid'])]

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    if not OVERWRITE:
        if eid in state_mod_df['eid'].values:
            continue
    print(f'\nStarting {subject}, {date} ({i+1} of {len(rec)})')

    # Load in camera timestamps and DLC output
    try:
        video_times, XYs = get_dlc_XYs(one, eid)
    except:
        print('Could not load video and/or DLC data')
        continue

    # Load opto times
    opto_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_times) == 0:
        continue

    # Select part of recording starting just before opto onset
    XYs = dict((k, XYs[k][video_times > opto_times[0] - 60, :]) for k in XYs.keys())
    video_times = video_times[video_times > opto_times[0] - 60]

    if np.sum(np.isnan(XYs['pupil_top_r'][:,0])) > XYs['pupil_top_r'].shape[0] / 2:
        continue

    # Smooth pupil trace
    print('Smoothing pupil trace..')
    raw_pupil, smooth_pupil = get_raw_smooth_pupil_diameter(XYs)

    # Select features to use
    XYs = dict((k, XYs[k]) for k in FEATURES)
    
    # Smooth traces
    print('Smoothing paw and nose traces..')
    XY_dist = np.empty((XYs[FEATURES[0]].shape[0]-1, len(FEATURES)))
    XY_smooth = XYs.copy()
    for j, point in enumerate(XYs.keys()):
        XY_smooth[point][:, 0] = smooth_interpolate_signal_sg(XYs[point][:, 0], window=61)
        XY_smooth[point][:, 1] = smooth_interpolate_signal_sg(XYs[point][:, 1], window=61)
        
        # Get displacement of traces
        xy_diff = np.diff(XY_smooth[point], axis=0)
        for kk in range(XY_dist.shape[0]):
            XY_dist[kk, j] = np.linalg.norm(xy_diff[kk, :])
    
    # Make an hmm and sample from it
    arhmm = ssm.HMM(K, len(FEATURES), observations='ar', transitions='sticky')
    arhmm.fit(XY_dist)
    zhat = arhmm.most_likely_states(XY_dist)
    
    # Make sure state 0 is inactive and state 1 active
    if np.mean(XY_dist[zhat == 0, 0]) > np.mean(XY_dist[zhat == 1, 0]):
        zhat = np.where((zhat==0)|(zhat==1), zhat^1, zhat)

    # Get state per stimulation onset
    pre_state, pre_pupil = np.empty(opto_times.shape), np.empty(opto_times.shape)
    for j, opto_time in enumerate(opto_times):
        
        # Movement HMM 
        pre_zhat = zhat[(video_times[:-1] > opto_time - T_BEFORE)
                        & (video_times[:-1] <= opto_time + T_AFTER)]
        if np.sum(pre_zhat == 0) > np.sum(pre_zhat == 1):
            pre_state[j] = 0
        else:
            pre_state[j] = 1
            
        # Pupil
        pupil_split = np.nanmedian(smooth_pupil)
        pre_pupil[j] = np.median(smooth_pupil[
            (video_times > opto_time - T_BEFORE)
            & (video_times <= opto_time + T_AFTER)] > pupil_split).astype(int)
        
    # Loop over probes
    insertions = one.alyx.rest('insertions', 'list', session=eid)
    pids = [i['id'] for i in insertions]
    for pid in pids:
        if pid not in rec['pid'].values:
            continue

        # Load in spikes
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)

        # Filter neurons that pass QC
        qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
        spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
        spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

        # Get modulation index for active and inactive state
        roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                                opto_times[pre_state == 0], pre_time=PRE_TIME,
                                                post_time=POST_TIME)
        mod_idx_inactive = 2 * (roc_auc - 0.5)
        roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                                opto_times[pre_state == 1], pre_time=PRE_TIME,
                                                post_time=POST_TIME)
        mod_idx_active = 2 * (roc_auc - 0.5)
        cluster_regions = remap(clusters.acronym[cluster_ids])
        
        # Get modulation index for large and small pupil
        roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                                opto_times[pre_pupil == 0], pre_time=PRE_TIME,
                                                post_time=POST_TIME)
        mod_idx_small_pupil = 2 * (roc_auc - 0.5)
        roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                                opto_times[pre_pupil == 1], pre_time=PRE_TIME,
                                                post_time=POST_TIME)
        mod_idx_large_pupil = 2 * (roc_auc - 0.5)
        cluster_regions = remap(clusters.acronym[cluster_ids])

        # Add to dataframe
        state_mod_df = pd.concat((state_mod_df, pd.DataFrame(data={
            'subject': subject, 'date': date, 'eid': eid, 'pid': pid,
            'region': cluster_regions, 'neuron_id': cluster_ids,
            'mod_index_inactive': mod_idx_inactive, 'mod_index_active': mod_idx_active,
            'mod_index_large_pupil': mod_idx_large_pupil, 'mod_index_small_pupil': mod_idx_small_pupil})))

    # Save to disk
    state_mod_df.to_csv(join(save_path, 'mov_pupil_state_mod.csv'))




