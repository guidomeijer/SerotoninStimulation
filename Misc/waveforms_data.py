# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:11:32 2023

@author: Guido
"""

import pandas as pd
import numpy as np
from os.path import join
from stim_functions import paths, load_subjects, combine_regions

_, load_path = paths()
_, save_path = paths(save_dir='cache')

# Load in waveforms
waveforms_df = pd.read_pickle(join(load_path, 'waveform_metrics.p'))

# Load in light modulation
awake_neurons = pd.read_csv(join(load_path, 'light_modulated_neurons.csv'))
awake_neurons['anesthesia'] = 0
anes_neurons = pd.read_csv(join(load_path, 'light_modulated_neurons_anesthesia.csv'))
anes_neurons['anesthesia'] = 1
all_neurons = pd.concat((awake_neurons, anes_neurons))
all_neurons = all_neurons[all_neurons['region'] != 'root']
all_neurons = all_neurons[all_neurons['region'] != 'void']

# Combine regions
all_neurons['brain_region'] = combine_regions(all_neurons['region'])

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Merge dataframes
merged_df = pd.merge(waveforms_df, all_neurons, on=['subject', 'probe', 'pid', 'eid', 'neuron_id'])
merged_df = merged_df.drop(columns=['waveform_2D', 'dist_soma', 'p_value', 'latency_peak', 'subject_nr',
                                    'firing_rate_y', 'mod_index_early', 'region', 'latency_peak_onset'])
merged_df = merged_df.rename(columns={'firing_rate_x': 'firing_rate', 'mod_index_late': 'mod_index'})

# Select only regions of interest
merged_df = merged_df[merged_df['brain_region'].isin(['Visual cortex', 'Barrel cortex', 'Secondary motor cortex'])]

# Save to disk
merged_df.to_pickle(join(save_path, 'waveforms_5HT.pkl'))