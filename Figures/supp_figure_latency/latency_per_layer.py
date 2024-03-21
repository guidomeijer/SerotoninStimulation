#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from scipy.stats import mannwhitneyu
from stim_functions import (paths, figure_style, load_subjects,
                            combine_regions, high_level_regions)

# Settings
MIN_NEURONS = 10

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
neuron_type = pd.read_csv(join(save_path, 'neuron_type.csv'))
neuron_type = neuron_type.rename(columns={'cluster_id': 'neuron_id'})
all_neurons = pd.merge(light_neurons, neuron_type, on=['subject', 'probe', 'eid', 'pid', 'neuron_id'])
all_neurons['full_region'] = combine_regions(all_neurons['region'], split_thalamus=False)
all_neurons['high_region'] = high_level_regions(all_neurons['region'])

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only modulated neurons in sert-cre mice
sert_neurons = all_neurons[(all_neurons['sert-cre'] == 1) & (all_neurons['modulated'] == 1)]

# Transform to ms
sert_neurons['latency'] = sert_neurons['latency_peak_onset'] * 1000

# Do some filtering
sert_neurons = sert_neurons[sert_neurons['type'] != 'Und.']
sert_neurons = sert_neurons[~np.isnan(sert_neurons['latency'])]

# Select cortical neurons
sert_neurons['is_cortex'] = ['cortex' in e for e in sert_neurons['full_region']]
sert_neurons = sert_neurons[sert_neurons['is_cortex']]

# Get layer per neuron
sert_neurons['cortical_layer'] = [i[-1] for i in sert_neurons['allen_acronym']]
sert_neurons['cortical_layer'] = sert_neurons['cortical_layer'].replace({
    '3': '2/3', 'a': '6', 'b': '6'})

# %%

PROPS = {'boxprops':{'facecolor':'none', 'edgecolor':'none'}, 'medianprops':{'color':'none'},
         'whiskerprops':{'color':'none'}, 'capprops':{'color':'none'}}
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

sns.stripplot(data=sert_neurons, x='latency', y='cortical_layer', ax=ax1,
              order=['2/3', '4', '5', '6'], s=2, zorder=0)
sns.boxplot(x='latency', y='cortical_layer', ax=ax1, data=sert_neurons, showmeans=True,
            order=['2/3', '4', '5', '6'],
            meanprops={"marker": "|", "markeredgecolor": "red", "markersize": "10"},
            fliersize=0, zorder=1, **PROPS)
ax1.set(ylabel='Cortical layer', xlabel='Modulation onset latency (ms)')

sns.despine(trim=True)
plt.tight_layout()




