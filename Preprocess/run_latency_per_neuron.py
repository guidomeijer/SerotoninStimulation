# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:50:06 2022

@author: Guido
"""

import numpy as np
from os.path import join
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from stim_functions import paths, load_passive_opto_times, init_one
from latenzy import latenzy
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = init_one()

# Settings
OVERWRITE = True
USE_DUR = 1.2  # s

# Load in results
fig_path, save_path = paths()
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
light_neurons = light_neurons.drop(columns=['latenzy'])

if OVERWRITE:
    latency_df = pd.DataFrame()
else:
    latency_df = pd.read_csv(join(save_path, 'latency.csv'))
    light_neurons = light_neurons[~np.isin(light_neurons['pid'], latency_df['pid'])]

for i, pid in enumerate(np.unique(light_neurons['pid'])):

    # Get session data
    eid = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'eid'])[0]
    subject = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'date'])[0]
    print(f'Recording {i} of {np.unique(light_neurons["pid"]).shape[0]}')

    # Load in laser pulse times
    try:
        opto_times, _ = load_passive_opto_times(eid, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_times) == 0:
        print('Did not find ANY laser pulses!')
        continue

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    # Select only modulated neurons
    mod_neurons = light_neurons.loc[(light_neurons['modulated'] == 1)
                                    & (light_neurons['pid'] == pid), 'neuron_id'].values
    latency = np.empty(mod_neurons.shape[0])
    for i, neuron_id in enumerate(mod_neurons):
        try:
            latency[i], _ = latenzy(spikes.times[spikes.clusters == neuron_id],
                                    opto_times,
                                    USE_DUR,
                                    jitter_size=2,
                                    peak_alpha=0.1
                                    )
        except Exception:
            latency[i] = np.nan
    
    # Add to dataframe
    light_neurons.loc[(light_neurons['modulated'] == 1)
                      & (light_neurons['pid'] == pid), 'latenzy'] = latency

# Save dataframe with new column
light_neurons.to_csv(join(save_path, 'light_modulated_neurons.csv'), index=False)



