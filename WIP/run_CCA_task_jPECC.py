#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
np.random.seed(42)  # fix random seed for reproducibility
from os.path import join, isfile
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import gaussian
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from brainbox.io.one import SpikeSortingLoader
from stim_functions import (paths, remap, query_ephys_sessions, load_trials,
                            get_artifact_neurons, calculate_peths, get_neuron_qc, high_level_regions)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
cca = CCA(n_components=1, max_iter=5000)
pca = PCA(n_components=10)

# Settings
OVERWRITE = True  # whether to overwrite existing runs
NEURON_QC = True  # whether to use neuron qc to exclude bad units
PCA = True  # whether to use PCA on neural activity before CCA
N_PC = 10  # number of PCs to use
MIN_NEURONS = 10  # minimum neurons per region
WIN_SIZE = 0.01  # window size in seconds
SMOOTHING = 0.025  # smoothing of psth
SUBTRACT_MEAN = False  # whether to subtract the mean PSTH from each trial
DIV_BASELINE = True  # whether to divide over baseline + 1 spk/s
K_FOLD = 10  # k in k-fold
MIN_FR = 0.5  # minimum firing rate over the whole recording

CENTER_ON = 'firstMovement_times'  
PRE_TIME = 0.3  # time before event in s 
POST_TIME = 0.1  # time after event in s
"""
CENTER_ON = 'stimOn_times'  
PRE_TIME = 0.1  # time before event in s
POST_TIME = 0.4  # time after event in s
"""

# Paths
fig_path, save_path = paths()

# Initialize some things
n_time_bins = int((PRE_TIME + POST_TIME) / WIN_SIZE)
kfold = KFold(n_splits=K_FOLD, shuffle=False)
if SMOOTHING > 0:
    w = n_time_bins - 1 if n_time_bins % 2 == 0 else n_time_bins
    window = gaussian(w, std=SMOOTHING / WIN_SIZE)
    window /= np.sum(window)

# Query sessions with frontal
rec = query_ephys_sessions(one=one, n_trials=300)

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

# Determine file name
add_str = ''
if DIV_BASELINE:
    add_str += 'div-baseline'
if SUBTRACT_MEAN:
    add_str += 'subtr-mean'    
file_name = f'jPECC_task_{CENTER_ON[:-6]}_{int(1/WIN_SIZE)}ms-bins_' + add_str + '.pickle'
    
if ~OVERWRITE & isfile(join(save_path, file_name)):
    cca_df = pd.read_pickle(join(save_path, file_name))
else:
    cca_df = pd.DataFrame(columns=['region_pair', 'eid'])

for i, eid in enumerate(np.unique(rec['eid'])):

    # Get session details
    subject = rec.loc[rec['eid'] == eid, 'subject'].values[0]
    date = rec.loc[rec['eid'] == eid, 'date'].values[0]
    print(f'Starting {subject}, {date} [{i+1} of {len(np.unique(rec["eid"]))}]')

    # Load in trials
    trials = load_trials(eid, laser_stimulation=True, one=one)
    
    # Exclude trials with RT < 0.1 ms and > 1 s
    if CENTER_ON == 'firstMovement_times':
        trials = trials[(trials['reaction_times'] > 0.2) & (trials['reaction_times'] < 1)]

    # Load in neural data of both probes of the recording
    spikes, clusters, channels, clusters_pass = dict(), dict(), dict(), dict()
    for (pid, probe) in zip(rec.loc[rec['eid'] == eid, 'pid'].values, rec.loc[rec['eid'] == eid, 'probe'].values):

        try:
            sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
            spikes[probe], clusters[probe], channels[probe] = sl.load_spike_sorting()
            clusters[probe] = sl.merge_clusters(spikes[probe], clusters[probe], channels[probe])
        except Exception as err:
            print(err)
            continue

        # Filter neurons that pass QC and artifact neurons
        if NEURON_QC:
            qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
            clusters_pass[probe] = np.where(qc_metrics['label'] == 1)[0]
        else:
            clusters_pass[probe] = np.unique(spikes.clusters)
        clusters_pass[probe] = clusters_pass[probe][~np.isin(clusters_pass[probe], artifact_neurons.loc[
            artifact_neurons['pid'] == pid, 'neuron_id'].values)]
        clusters[probe]['region'] = high_level_regions(clusters[probe]['acronym'], only_vis=True)

    # Create population activity arrays for all regions
    pca_opto, spks_opto = dict(), dict()
    pca_no_opto, spks_no_opto = dict(), dict()
    for probe in spikes.keys():
        for region in np.unique(clusters[probe]['region']):

            # Exclude neurons with low firing rates
            clusters_in_region = np.where(clusters[probe]['region'] == region)[0]
            fr = np.empty(clusters_in_region.shape[0])
            for nn, neuron_id in enumerate(clusters_in_region):
                fr[nn] = np.sum(spikes[probe].clusters == neuron_id) / spikes[probe].clusters[-1]
            clusters_in_region = clusters_in_region[fr >= MIN_FR]

            # Get spikes and clusters
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)
                                              & np.isin(spikes[probe].clusters, clusters_pass[probe])]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)
                                                 & np.isin(spikes[probe].clusters, clusters_pass[probe])]

            if (len(np.unique(clus_region)) >= MIN_NEURONS) & (region != 'root'):
                print(f'Loading population activity for {region}')

                # Get PSTH and binned spikes for OPTO activity
                psth_opto, binned_spks_opto = calculate_peths(
                    spks_region, clus_region, np.unique(clus_region),
                    trials.loc[trials['laser_stimulation'] == 1, CENTER_ON],
                    pre_time=PRE_TIME, post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING,
                    return_fr=False)

                if DIV_BASELINE:
                    # Divide each trial over baseline + 1 spks/s
                    for nn in range(binned_spks_opto.shape[1]):
                        for tt in range(binned_spks_opto.shape[0]):
                            binned_spks_opto[tt, nn, :] = (binned_spks_opto[tt, nn, :]
                                                          / (np.mean(psth_opto['means'][nn, psth_opto['tscale'] < 0])
                                                             + (1/PRE_TIME)))

                if SUBTRACT_MEAN:
                    # Subtract mean PSTH from each opto stim
                    for tt in range(binned_spks_opto.shape[0]):
                        binned_spks_opto[tt, :, :] = binned_spks_opto[tt, :, :] - psth_opto['means']

                # Add to dict
                spks_opto[region] = binned_spks_opto

                # Perform PCA
                pca_opto[region] = np.empty([binned_spks_opto.shape[0], N_PC, binned_spks_opto.shape[2]])
                for tb in range(binned_spks_opto.shape[2]):
                    pca_opto[region][:, :, tb] = pca.fit_transform(binned_spks_opto[:, :, tb])
                    
                # Get PSTH and binned spikes for NO OPTO activity
                psth_no_opto, binned_spks_no_opto = calculate_peths(
                    spks_region, clus_region, np.unique(clus_region),
                    trials.loc[trials['laser_stimulation'] == 0, CENTER_ON],
                    pre_time=PRE_TIME, post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING,
                    return_fr=False)

                if DIV_BASELINE:
                    # Divide each trial over baseline + 1 spks/s
                    for nn in range(binned_spks_no_opto.shape[1]):
                        for tt in range(binned_spks_no_opto.shape[0]):
                            binned_spks_no_opto[tt, nn, :] = (binned_spks_no_opto[tt, nn, :]
                                                              / (np.median(psth_no_opto['means'][nn, psth_no_opto['tscale'] < 0])
                                                                 + (1/PRE_TIME)))

                if SUBTRACT_MEAN:
                    # Subtract mean PSTH from each opto stim
                    for tt in range(binned_spks_no_opto.shape[0]):
                        binned_spks_no_opto[tt, :, :] = binned_spks_no_opto[tt, :, :] - psth_no_opto['means']

                # Add to dict
                spks_no_opto[region] = binned_spks_no_opto

                # Perform PCA
                pca_no_opto[region] = np.empty([binned_spks_no_opto.shape[0], N_PC, binned_spks_no_opto.shape[2]])
                for tb in range(binned_spks_no_opto.shape[2]):
                    pca_no_opto[region][:, :, tb] = pca.fit_transform(binned_spks_no_opto[:, :, tb])


    # Perform CCA per region pair
    print('Starting CCA per region pair')
    all_cca_df = pd.DataFrame()
    for r1, region_1 in enumerate(pca_opto.keys()):
        for r2, region_2 in enumerate(list(pca_opto.keys())[r1:]):
            if region_1 == region_2:
                continue
            region_pair = f'{np.sort([region_1, region_2])[0]}-{np.sort([region_1, region_2])[1]}'
            
            # Skip if already processed
            if cca_df[(cca_df['region_pair'] == region_pair) & (cca_df['eid'] == eid)].shape[0] > 0:
                print(f'Found {region_1}-{region_2} for {subject} {date}')
                continue
    
            if (region_1 in pca_opto.keys()) & (region_2 in pca_opto.keys()):
                print(f'Calculating {region_pair}')
                
                # Run CCA per combination of two timebins
                print('Trials with opto stimulation')
                
                r_opto = np.empty((n_time_bins, n_time_bins))
                p_opto = np.empty((n_time_bins, n_time_bins))    
                for tb_1, time_1 in enumerate(psth_opto['tscale']):
                    if np.mod(tb_1, 10) == 0:
                        print(f'Timebin {tb_1} of {n_time_bins}..')
                    for tb_2, time_2 in enumerate(psth_opto['tscale']):
        
                        # Get activity matrix
                        if PCA:
                            act_mat = pca_opto
                        else:
                            act_mat = spks_opto
    
                        # Run CCA
                        x_test = np.empty(act_mat[region_1].shape[0])
                        y_test = np.empty(act_mat[region_1].shape[0])
                        for train_index, test_index in kfold.split(act_mat[region_1][:, :, 0]):
                            cca.fit(act_mat[region_1][train_index, :, tb_1],
                                    act_mat[region_2][train_index, :, tb_2])
                            x, y = cca.transform(act_mat[region_1][test_index, :, tb_1],
                                                 act_mat[region_2][test_index, :, tb_2])
                            x_test[test_index] = x.T[0]
                            y_test[test_index] = y.T[0]
                        r_opto[tb_1, tb_2], p_opto[tb_1, tb_2] = pearsonr(x_test, y_test)
                        
                # Run CCA per combination of two timebins
                print('Trials without opto stimulation')
                
                r_no_opto = np.empty((n_time_bins, n_time_bins))
                p_no_opto = np.empty((n_time_bins, n_time_bins))    
                for tb_1, time_1 in enumerate(psth_no_opto['tscale']):
                    if np.mod(tb_1, 10) == 0:
                        print(f'Timebin {tb_1} of {n_time_bins}..')
                    for tb_2, time_2 in enumerate(psth_no_opto['tscale']):
        
                        # Get activity matrix
                        if PCA:
                            act_mat = pca_no_opto
                        else:
                            act_mat = spks_no_opto
    
                        # Run CCA
                        x_test = np.empty(act_mat[region_1].shape[0])
                        y_test = np.empty(act_mat[region_1].shape[0])
                        for train_index, test_index in kfold.split(act_mat[region_1][:, :, 0]):
                            cca.fit(act_mat[region_1][train_index, :, tb_1],
                                    act_mat[region_2][train_index, :, tb_2])
                            x, y = cca.transform(act_mat[region_1][test_index, :, tb_1],
                                                 act_mat[region_2][test_index, :, tb_2])
                            x_test[test_index] = x.T[0]
                            y_test[test_index] = y.T[0]
                        r_no_opto[tb_1, tb_2], p_no_opto[tb_1, tb_2] = pearsonr(x_test, y_test)

            # Add to dataframe
            cca_df = pd.concat((cca_df, pd.DataFrame(index=[cca_df.shape[0]], data={
                'subject': subject, 'date': date, 'eid': eid, 'region_1': region_1, 'region_2': region_2,
                'region_pair': region_pair, 'r_opto': [r_opto], 'p_opto': [p_opto],
                'r_no_opto': [r_no_opto], 'p_no_opto': [p_no_opto], 'time': [psth_opto['tscale']]})))
    cca_df.to_pickle(join(save_path, file_name))

cca_df.to_pickle(join(save_path, file_name))
