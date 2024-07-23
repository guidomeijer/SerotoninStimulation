# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:50:06 2022

@author: Guido
"""

import numpy as np
from os.path import join
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from stim_functions import paths, load_passive_opto_times, load_trials
from brainbox.population.decode import get_spike_counts_in_bins
from sklearn.metrics import roc_auc_score
from brainbox.task.closed_loop import roc_between_two_events
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
OVERWRITE = True
BASELINE = [0.5, 0]
PRE_TIME = 1
POST_TIME = 5
BIN_SIZE = 0.1
win_centers = np.arange(-PRE_TIME + (BIN_SIZE/2), POST_TIME, BIN_SIZE)

# Load in results
fig_path, save_path = paths()
light_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))

if OVERWRITE:
    mod_idx_df = pd.DataFrame()
else:
    mod_idx_df = pd.read_pickle(join(save_path, 'mod_over_time_task.pickle'))
    light_neurons = light_neurons[~np.isin(light_neurons['pid'], mod_idx_df['pid'])]

for i, pid in enumerate(np.unique(light_neurons['pid'])):

    # Get session data
    eid = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'eid'])[0]
    subject = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'date'])[0]
    print(f'Processing {subject} {date}')

    # Load in trials
    trials = load_trials(eid, laser_stimulation=True, one=one)

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    # Select only modulated neurons
    these_neurons = light_neurons[(light_neurons['opto_modulated'] == 1) & (light_neurons['pid'] == pid)]
    spike_times = spikes.times[np.isin(spikes.clusters, these_neurons['neuron_id'])]
    spike_clusters = spikes.clusters[np.isin(spikes.clusters, these_neurons['neuron_id'])]
    if spike_times.shape[0] == 0:
        continue
   
    # Loop over time bins
    mod_idx_auc = np.empty((len(np.unique(spike_clusters)), win_centers.shape[0]))
    for itb, win_c in enumerate(win_centers):
        
        mod_idx_auc[:, itb], _ = roc_between_two_events(
            spike_times, spike_clusters, trials['goCue_times'],
            trials['laser_stimulation'], 
            pre_time=(win_c - (BIN_SIZE/2)) * -1,
            post_time=win_c + (BIN_SIZE/2))
            
    # Rescale area under to curve to [-1, 1] range
    mod_idx = 2 * (mod_idx_auc - 0.5)

    # Add to dataframe
    for iin, neuron_id in enumerate(these_neurons['neuron_id']):
        mod_idx_df = pd.concat((mod_idx_df, pd.DataFrame(index=[mod_idx_df.shape[0]+1], data={
            'pid': pid, 'subject': subject, 'date': date, 'neuron_id': neuron_id,
            'mod_idx': [mod_idx[iin, :]], 'time': [win_centers],
            'region': these_neurons.loc[these_neurons['neuron_id'] == neuron_id, 'region'].values[0]})))

    # Save output
    mod_idx_df.to_pickle(join(save_path, 'mod_over_time_task.pickle'))

