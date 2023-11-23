#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from brainbox.task.closed_loop import roc_single_event
from zetapy import zetatest
from brainbox.io.one import SpikeSortingLoader
from stim_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                            remove_artifact_neurons, get_neuron_qc)
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
OVERWRITE = True
NEURON_QC = True
PRE_TIME = [0.5, 0]  # for modulation index
POST_TIME = [0.3, 0.8]
BIN_SIZE = 0.05
MIN_FR = 0.1
_, save_path = paths()

# Query sessions
rec = query_ephys_sessions(anesthesia='no&both', one=one)

if OVERWRITE:
    light_neurons = pd.DataFrame()
else:
    light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
    rec = rec[~rec['pid'].isin(light_neurons['pid'])]

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

    print(f'\nStarting {subject}, {date}')

    # Load in laser pulse times
    try:
        opto_train_times, _ = load_passive_opto_times(eid, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_train_times) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_train_times)} passive laser pulses')

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    if 'acronym' not in clusters.keys():
        print(f'No brain regions found for {eid}')
        continue

    # Filter neurons that pass QC
    if NEURON_QC:
        qc_metrics = get_neuron_qc(pid, one=one, ba=ba, force_rerun=False)
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    else:
        clusters_pass = np.unique(spikes.clusters)
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    if len(spikes.clusters) == 0:
        continue

    # Determine significant neurons
    print('Performing ZETA tests..')
    p_values = np.empty(np.unique(spikes.clusters).shape)
    latency_peak = np.empty(np.unique(spikes.clusters).shape)
    latency_peak_onset = np.empty(np.unique(spikes.clusters).shape)
    firing_rates = np.empty(np.unique(spikes.clusters).shape)
    for n, neuron_id in enumerate(np.unique(spikes.clusters)):
        if np.mod(n, 20) == 0:
            print(f'Neuron {n} of {np.unique(spikes.clusters).shape[0]}')

        # Perform ZETA test for neural responsiveness
        p_values[n], dZETA, dRate = zetatest(spikes.times[spikes.clusters == neuron_id],
                                             opto_train_times,
                                             tplRestrictRange=(0, 1), dblUseMaxDur=3,
                                             boolReturnRate=True)

        # Get modulation onset
        latency_peak[n] = dZETA['vecLatencies'][2]
        latency_peak_onset[n] = dZETA['vecLatencies'][3]

        # Get firing rate
        firing_rates[n] = (np.sum(spikes.times[spikes.clusters == neuron_id].shape[0])
                           / (spikes.times[-1]))

    # Exclude low firing rate units
    p_values[firing_rates < MIN_FR] = 1
    latency_peak[firing_rates < MIN_FR] = np.nan
    latency_peak_onset[firing_rates < MIN_FR] = np.nan
    print(f'Found {np.sum(p_values < 0.05)} opto modulated neurons')

    # Calculate modulation index
    roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                            opto_train_times, pre_time=PRE_TIME,
                                            post_time=POST_TIME)
    mod_idx = 2 * (roc_auc - 0.5)

    cluster_regions = remap(clusters.acronym[cluster_ids])
    light_neurons = pd.concat((light_neurons, pd.DataFrame(data={
        'subject': subject, 'date': date, 'eid': eid, 'probe': probe, 'pid': pid,
        'allen_acronym': clusters.acronym[cluster_ids], 'region': cluster_regions,
        'neuron_id': cluster_ids, 'mod_index': mod_idx,
        'modulated': p_values < 0.05, 'p_value': p_values, 'firing_rate': firing_rates,
        'latency_peak': latency_peak, 'latency_peak_onset': latency_peak_onset})))

    # Save output for this insertion
    light_neurons.to_csv(join(save_path, 'light_modulated_neurons.csv'), index=False)
    print('Saved output to disk')

# Remove artifact neurons
light_neurons = remove_artifact_neurons(light_neurons)

# Save output
light_neurons.to_csv(join(save_path, 'light_modulated_neurons.csv'), index=False)
