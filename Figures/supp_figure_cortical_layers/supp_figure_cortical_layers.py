# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:38:41 2023 by Guido Meijer
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style, combine_regions, load_subjects, remap

# Settings
MIN_NEURONS = 5

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1] + '.pdf')

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Select cortical neurons
light_neurons['cosmos_regions'] = remap(light_neurons['allen_acronym'], dest='Cosmos')
light_neurons = light_neurons[light_neurons['cosmos_regions'] == 'Isocortex']

# Get layer per neuron
light_neurons['cortical_layer'] = [i[-1] for i in light_neurons['allen_acronym']]
light_neurons['cortical_layer'] = light_neurons['cortical_layer'].replace({
    '3': '2/3', 'a': '6', 'b': '6'})

# Drop layer 1 and 4
light_neurons = light_neurons[((light_neurons['cortical_layer'] != '1') 
                               & (light_neurons['cortical_layer'] != '4'))]

# Get summary stats
summary_df = light_neurons.groupby(['pid', 'cortical_layer']).size().to_frame()
summary_df = summary_df.rename(columns={0: 'n_neurons'})
summary_df['perc_mod'] = (light_neurons.groupby(['pid', 'cortical_layer']).sum(numeric_only=True)['modulated']
                          / light_neurons.groupby(['pid', 'cortical_layer']).size())
summary_df['mod_index'] = light_neurons[light_neurons['modulated']].groupby(['pid', 'cortical_layer']).mean(
    numeric_only=True)['mod_index']
summary_df = summary_df[summary_df['n_neurons'] > MIN_NEURONS]
summary_df = summary_df.reset_index()

# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(1.75, 1.75), dpi=dpi)
#sns.swarmplot(data=summary_df, y='cortical_layer', x='perc_mod', ax=ax1, size=2,
#              order=['2/3', '5', '6'])
sns.barplot(data=summary_df, y='cortical_layer', x='perc_mod', ax=ax1, 
            order=['2/3', '5', '6'], errorbar='se', color='grey')
ax1.set(ylabel='Layer', xlim=[0, 0.3], xticks=[0, 0.15, 0.3],
        xticklabels=[0, 15, 30])
ax1.set_xlabel('Modulated neurons (%)', labelpad=1)


props = {'boxprops':{'facecolor':'none', 'edgecolor':'none'}, 'medianprops':{'color':'none'},
         'whiskerprops':{'color':'none'}, 'capprops':{'color':'none'}}
sns.stripplot(data=light_neurons[light_neurons['modulated']], y='cortical_layer', x='mod_index',
              order=['2/3', '5', '6'], size=2, zorder=1, ax=ax2, color='grey')
sns.boxplot(x='mod_index', y='cortical_layer', ax=ax2, data=light_neurons[light_neurons['modulated']],
            showmeans=True, order=['2/3', '5', '6'],
            meanprops={"marker": "|", "markeredgecolor": "red", "markersize": "8"},
            fliersize=0, zorder=2, **props)
ax2.set(ylabel='Layer', xlim=[-0.75, 0.75], xticks=[-0.75, 0, 0.75],
        xticklabels=[-0.75, 0, 0.75])
ax2.set_xlabel('Modulation index', labelpad=1)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path))
