#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:52:05 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import ssm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from brainbox.io.one import SpikeSortingLoader
from stim_functions import (load_passive_opto_times, get_neuron_qc, paths, query_ephys_sessions,
                            figure_style, load_subjects, remap, high_level_regions,
                            get_artifact_neurons, calculate_peths)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
N_STATES = 2
BIN_SIZE = 0.2
MIN_NEURONS = 2
STIM_FREQS = [1, 5, 10, 25]
PRE_TIME = 1  # final time window to use
POST_TIME = 4
HMM_PRE_TIME = 2  # time window to run HMM on
HMM_POST_TIME = 5
OVERWRITE = True
PLOT = True

# Get path
fig_path, save_path = paths()

# Query sessions
rec = query_ephys_sessions(anesthesia='yes', one=one)
subjects = load_subjects()

# Get artifact neurons
artifact_neurons = get_artifact_neurons()

if OVERWRITE:
    up_down_state_df, up_down_state_null_df = pd.DataFrame(), pd.DataFrame()
else:
    up_down_state_df = pd.read_csv(join(save_path, 'updown_state_anesthesia_freqs.csv'))
    up_down_state_null_df = pd.read_csv(join(save_path, 'updown_state_null_anesthesia_freqs.csv'))
    rec = rec[~rec['pid'].isin(up_down_state_df['pid'])]

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'\nStarting {subject}, {date}, {probe} ({i+1} of {len(rec)})')

    if not OVERWRITE:
        if pid in up_down_state_df['pid'].values:
            continue
        
    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Filter neurons that pass QC
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]

    # Exclude artifact neurons
    clusters_pass = np.array([i for i in clusters_pass if i not in artifact_neurons.loc[
        artifact_neurons['pid'] == pid, 'neuron_id'].values])
    if clusters_pass.shape[0] == 0:
            continue

    # Select QC pass neurons
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    clusters_pass = clusters_pass[np.isin(clusters_pass, np.unique(spikes.clusters))]

    # Get regions from Beryl atlas
    clusters['region'] = remap(clusters['acronym'], combine=True)
    clusters['high_level_region'] = high_level_regions(clusters['acronym'])
    clusters_regions = clusters['high_level_region'][clusters_pass]

    # Loop over stimulation frequencies
    for f, freq in enumerate(STIM_FREQS):
        print(f'Starting {freq}Hz stimulation')
        
        # Load opto times for this frequency
        opto_times, _ = load_passive_opto_times(eid, one=one, freq=freq)

        # Loop over regions
        for r, region in enumerate(np.unique(clusters['high_level_region'])):
            if region == 'root':
                continue
    
            # Select spikes and clusters in this brain region
            clusters_in_region = clusters_pass[clusters_regions == region]
            if len(clusters_in_region) < MIN_NEURONS:
                continue
    
            # Get binned spikes centered at stimulation onset
            peth, binned_spikes = calculate_peths(spikes.times, spikes.clusters, clusters_in_region, opto_times,
                                                  pre_time=HMM_PRE_TIME, post_time=HMM_POST_TIME, bin_size=BIN_SIZE,
                                                  smoothing=0, return_fr=False)
            binned_spikes = binned_spikes.astype(int)
            full_time_ax = peth['tscale']
            use_timepoints = (full_time_ax > -PRE_TIME) & (full_time_ax < POST_TIME)
            time_ax = full_time_ax[use_timepoints]
    
            # Create list of (time_bins x neurons) per stimulation trial
            trial_data = []
            for i in range(binned_spikes.shape[0]):
                trial_data.append(np.transpose(binned_spikes[i, :, :]))
    
            # Initialize HMM
            simple_hmm = ssm.HMM(N_STATES, clusters_in_region.shape[0], observations='poisson')
    
            this_df = pd.DataFrame()
            trans_mat = np.empty((len(trial_data), full_time_ax.shape[0]))
            down_trans_mat = np.empty((len(trial_data), full_time_ax.shape[0]))
            up_trans_mat = np.empty((len(trial_data), full_time_ax.shape[0]))
           
            # Fit HMM on all data
            lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')
        
            for t in range(len(trial_data)):
        
                # Get posterior probability and most likely states for this trial
                posterior = simple_hmm.filter(trial_data[t])
                posterior = posterior[np.concatenate(([False], use_timepoints[:-1])), :]  
                zhat = simple_hmm.most_likely_states(trial_data[t])
                
                # Make sure 0 is down state and 1 is up state
                if np.mean(binned_spikes[t, :, zhat==0]) > np.mean(binned_spikes[t, :, zhat==1]):
                    # State 0 is up state
                    zhat = np.where((zhat==0)|(zhat==1), zhat^1, zhat)
                    p_pos_down = posterior[:, 1]
                else:
                    p_pos_down = posterior[:, 0]
                
                # Select trial timewindow
                zhat = zhat[use_timepoints]
        
                # Add to dataframe
                this_df = pd.concat((this_df, pd.DataFrame(data={
                    'state': zhat, 'p_pos_down': p_pos_down, 'region': region, 'time': time_ax,
                    'trial': t})))
    
            # Smooth and crop state change traces 
            p_state_change = gaussian_filter(np.mean(trans_mat, axis=0), 1)
            p_state_change = p_state_change[use_timepoints]
            p_down_state_change = gaussian_filter(np.mean(down_trans_mat, axis=0), 1)
            p_down_state_change = p_down_state_change[use_timepoints]
            p_up_state_change = gaussian_filter(np.mean(up_trans_mat, axis=0), 1)
            p_up_state_change = p_up_state_change[use_timepoints]
           
            # Add to dataframe
            p_down = this_df[['time', 'state']].groupby('time').mean().reset_index()
            p_down['state'] = 1-p_down['state']
            p_down['state_bl'] = p_down['state'] - p_down.loc[p_down['time'] < 0, 'state'].mean()
            
            up_down_state_df = pd.concat((up_down_state_df, pd.DataFrame(data={
                'p_down': p_down['state'], 'p_down_bl': p_down['state_bl'], 'time': p_down['time'],
                'freq': freq, 'subject': subject, 'pid': pid, 'region': region})))
    
            # Run the HMM on random onset times
            random_times = np.sort(np.random.uniform(opto_times[0], opto_times[-1], size=100))
            
            # Get binned spikes centered at stimulation onset
            peth, binned_spikes = calculate_peths(spikes.times, spikes.clusters, clusters_in_region, random_times,
                                                  pre_time=HMM_PRE_TIME, post_time=HMM_POST_TIME, bin_size=BIN_SIZE,
                                                  smoothing=0, return_fr=False)
            binned_spikes = binned_spikes.astype(int)
            full_time_ax = peth['tscale']
            use_timepoints = (full_time_ax > -PRE_TIME) & (full_time_ax < POST_TIME)
            time_ax = full_time_ax[use_timepoints]
        
            # Create list of (time_bins x neurons) per stimulation trial
            trial_data = []
            for i in range(binned_spikes.shape[0]):
                trial_data.append(np.transpose(binned_spikes[i, :, :]))
        
            # Initialize HMM
            simple_hmm = ssm.HMM(N_STATES, clusters_in_region.shape[0], observations='poisson')
        
            this_df = pd.DataFrame()
            trans_mat = np.empty((len(trial_data), full_time_ax.shape[0]))
            down_trans_mat = np.empty((len(trial_data), full_time_ax.shape[0]))
            up_trans_mat = np.empty((len(trial_data), full_time_ax.shape[0]))
           
            # Fit HMM on all data
            lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')
        
            for t in range(len(trial_data)):
        
                # Get posterior probability and most likely states for this trial
                posterior = simple_hmm.filter(trial_data[t])
                posterior = posterior[np.concatenate(([False], use_timepoints[:-1])), :]  
                zhat = simple_hmm.most_likely_states(trial_data[t])
                
                # Make sure 0 is down state and 1 is up state
                if np.mean(binned_spikes[t, :, zhat==0]) > np.mean(binned_spikes[t, :, zhat==1]):
                    # State 0 is up state
                    zhat = np.where((zhat==0)|(zhat==1), zhat^1, zhat)
                    p_pos_down = posterior[:, 1]
                else:
                    p_pos_down = posterior[:, 0]
                
                # Select trial timewindow
                zhat = zhat[use_timepoints]
        
                # Add to dataframe
                this_df = pd.concat((this_df, pd.DataFrame(data={
                    'state': zhat, 'p_pos_down': p_pos_down, 'region': region, 'time': time_ax,
                    'trial': t})))
        
            # Smooth and crop state change traces 
            p_state_change = gaussian_filter(np.mean(trans_mat, axis=0), 1)
            p_state_change = p_state_change[use_timepoints]
            p_down_state_change = gaussian_filter(np.mean(down_trans_mat, axis=0), 1)
            p_down_state_change = p_down_state_change[use_timepoints]
            p_up_state_change = gaussian_filter(np.mean(up_trans_mat, axis=0), 1)
            p_up_state_change = p_up_state_change[use_timepoints]
           
            # Add to dataframe
            p_down = this_df[['time', 'state']].groupby('time').mean().reset_index()
            p_down['state'] = 1-p_down['state']
            p_down['state_bl'] = p_down['state'] - p_down.loc[p_down['time'] < 0, 'state'].mean()
            
            up_down_state_null_df = pd.concat((up_down_state_null_df, pd.DataFrame(data={
                'p_down': p_down['state'], 'p_down_bl': p_down['state_bl'], 'time': p_down['time'],
                'freq': freq, 'subject': subject, 'pid': pid, 'region': region})))
        
    # Save result
    up_down_state_df.to_csv(join(save_path, 'updown_states_anesthesia_freqs.csv'))
    up_down_state_null_df.to_csv(join(save_path, 'updown_states_null_anesthesia_freqs.csv'))

