# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:53:12 2024

By Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from brainbox.io.one import SpikeSortingLoader
from brainbox.population.decode import get_spike_counts_in_bins, classify
from brainbox.task.closed_loop import (responsive_units, roc_single_event, differentiate_units,
                                       roc_between_two_events, generate_pseudo_blocks)

from stim_functions import (paths, remap, query_ephys_sessions, load_trials, figure_style,
                            get_neuron_qc, combine_regions, load_subjects,
                            get_artifact_neurons, init_one, calculate_peths)

# Settings
PRE_TIME = 0.3
POST_TIME = 0
#EVENT = 'feedback_times'
#EVENT = 'stimOn_times'
EVENT = 'firstMovement_times'
MIN_NEURONS = 5
MIN_TRIALS = 5

# Initialize
one = init_one()
subjects = load_subjects()
rf_classifier = RandomForestClassifier()
k_fold = KFold(n_splits=10, shuffle=True)

# Set paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Extra plots', 'Task neurons')
colors, dpi = figure_style()

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

# Query sessions
rec = query_ephys_sessions(n_trials=400, one=one)

decoding_df = pd.DataFrame()
for i in rec.index.values:
    
    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'Session {i} of {rec.shape[0]}: {subject} {date}')
    
    # Load in trials and spikes
    try:
        trials = load_trials(eid, laser_stimulation=True, one=one)
        sl = SpikeSortingLoader(pid=pid, one=one)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue
    
    # Select neurons that pass QC and don't have artifacts
    qc_metrics = get_neuron_qc(pid, one=one)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
        artifact_neurons['pid'] == pid, 'neuron_id'].values)]
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    
    # Loop over regions
    clusters['region'] = combine_regions(remap(clusters['acronym']))
    for region in np.unique(clusters['region']):
        if region == 'root':
            continue
        
        # Select neurons from this region
        clusters_in_region = np.where(clusters['region'] == region)[0]
        region_spikes = spikes.times[np.isin(spikes.clusters, clusters_in_region)]
        region_clusters = spikes.clusters[np.isin(spikes.clusters, clusters_in_region)]
        if np.unique(region_clusters).shape[0] < MIN_NEURONS:
            continue
    
        # Decode stimulated trials
        stim_trials = trials[(trials['laser_probability'] != 0.25) & (trials['laser_stimulation'] == 1)]
        #stim_trials = trials[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 1)]
        if stim_trials.index[0] == 0:
            stim_trials = stim_trials[1:]
        stim_intervals = np.vstack((stim_trials[EVENT] - PRE_TIME,
                                    stim_trials[EVENT] + POST_TIME)).T
        spike_counts, neuron_ids = get_spike_counts_in_bins(region_spikes, region_clusters, stim_intervals)
        
        # Check if session is ok to run
        if np.sum(spike_counts) == 0:
            continue
        if (np.sum(stim_trials['choice'] == 1) < MIN_TRIALS) | (np.sum(stim_trials['choice'] == -1) < MIN_TRIALS):
            continue
        
        stim_prev, _, _ = classify(spike_counts.T,
                                   trials.loc[stim_trials.index - 1, 'choice'].values,
                                   rf_classifier,
                                   cross_validation=k_fold)
        stim_this, _, _ = classify(spike_counts.T,
                                   trials.loc[stim_trials.index, 'choice'].values,
                                   rf_classifier,
                                   cross_validation=k_fold)
        
        # Decode non-stimulated trials
        #no_stim_trials = trials[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 0)]
        no_stim_trials = trials[(trials['laser_probability'] != 0.75) & (trials['laser_stimulation'] == 0)]
        if no_stim_trials.index[0] == 0:
            no_stim_trials = no_stim_trials[1:]        
        no_stim_intervals = np.vstack((no_stim_trials[EVENT] - PRE_TIME,
                                       no_stim_trials[EVENT] + POST_TIME)).T
        spike_counts, neuron_ids = get_spike_counts_in_bins(region_spikes, region_clusters, no_stim_intervals)
        no_stim_prev, _, _ = classify(spike_counts.T,
                                      trials.loc[no_stim_trials.index - 1, 'choice'].values,
                                      rf_classifier,
                                      cross_validation=k_fold)
        no_stim_this, _, _ = classify(spike_counts.T,
                                      trials.loc[no_stim_trials.index, 'choice'].values,
                                      rf_classifier,
                                      cross_validation=k_fold)
        
        # Add to dataframe
        decoding_df = pd.concat((decoding_df, pd.DataFrame(index=[decoding_df.shape[0]], data={
            'stim_this_trial': stim_this, 'stim_prev_trial': stim_prev,
            'no_stim_this_trial': no_stim_this, 'no_stim_prev_trial': no_stim_prev,
            'region': region, 'subject': subject, 'sert-cre': sert_cre,
            'probe': probe, 'date': date, 'eid': eid, 'pid': pid})))
        decoding_df.to_csv(join(save_path, f'decoding_all_trials_prev_choice_{EVENT}.csv'), index=False)
    