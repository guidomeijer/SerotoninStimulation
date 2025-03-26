# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:49:28 2023

By Guido Meijer
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so
from scipy.stats import ttest_ind, ttest_rel
from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style, load_subjects, combine_regions

# Settings
MIN_NEURONS = 10

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'), index_col=0)
neuron_type = pd.read_csv(join(save_path, 'neuron_type.csv'))
neuron_type = neuron_type[neuron_type['type'] != 'Und.']
merged_df = pd.merge(light_neurons, neuron_type, on=['neuron_id', 'pid', 'eid', 'probe'])
merged_df['full_region'] = combine_regions(merged_df['region'], abbreviate=True)

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    merged_df.loc[merged_df['subject'] == nickname,
                  'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
merged_df = merged_df[merged_df['sert-cre'] == 1]

ns_perc = (merged_df[(merged_df['type'] == 'NS') & merged_df['modulated']].groupby('subject').size()
           / merged_df[merged_df['type'] == 'NS'].groupby('subject').size()) * 100
ws_perc = (merged_df[(merged_df['type'] == 'WS') & merged_df['modulated']].groupby('subject').size() 
           / merged_df[merged_df['type'] == 'WS'].groupby('subject').size()) * 100
per_mouse_df = pd.concat((ns_perc, ws_perc), axis=1)
per_mouse_df = per_mouse_df.rename(columns={0: 'NS', 1: 'WS'})

_, p = ttest_ind(merged_df.loc[(merged_df['type'] == 'WS') & merged_df['modulated'], 'mod_index'],
                 merged_df.loc[(merged_df['type'] == 'NS') & merged_df['modulated'], 'mod_index'])
print(f'Modulation index: p = {p:.2f}')

_, p = ttest_rel(per_mouse_df['NS'], per_mouse_df['WS'])
print(f'Percentage: p = {p:.2f}')

# %%
colors, dpi = figure_style()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.75*2, 1.75), dpi=dpi)

for i in per_mouse_df.index:
    ax1.plot([0, 1], [per_mouse_df.loc[i, 'NS'], per_mouse_df.loc[i, 'WS']], color='k', marker='o',
             markersize=2.5)
ax1.set(xticks=[0, 1], xticklabels=['Narrow\nspiking', 'Wide\nspiking'], yticks=[0, 50, 100],
        xlim=[-0.25, 1.25], ylabel='5-HT modulated neurons (%)')
ax1.text(0.5, 90, '*', ha='center', va='center', fontsize=12)

sns.violinplot(data=merged_df[merged_df['modulated']], x='type', y='mod_index',
               ax=ax2, palette=[colors['NS'], colors['WS']], linewidth=0.5)
ax2.set(ylabel='Modulation index', xticklabels=['Narrow\nspiking', 'Wide\nspiking'],
        xlabel='', yticks=[-1, -0.5, 0, 0.5, 1])
ax2.text(0.5, 1, 'n.s.', ha='center', va='center', fontsize=7)


sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'neuron_type_modulation.pdf'))

