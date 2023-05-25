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
from scipy.stats import ttest_ind
from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style, load_subjects, high_level_regions

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])
colors, dpi = figure_style()

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'), index_col=0)
neuron_type = pd.read_csv(join(save_path, 'neuron_type.csv'))
neuron_type = neuron_type[neuron_type['type'] != 'Und.']
merged_df = pd.merge(light_neurons, neuron_type, on=['neuron_id', 'pid', 'eid', 'probe'])
merged_df['high_level_region'] = high_level_regions(merged_df['region'], input_atlas='Beryl')
merged_df = merged_df[merged_df['high_level_region'] != 'root']

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    merged_df.loc[merged_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
merged_df = merged_df[merged_df['sert-cre'] == 1]

"""
# %%

ns_df = merged_df[merged_df['type'] == 'NS']
perc_ns = (ns_df.groupby(['pid', 'subject']).sum(numeric_only=True)['modulated']
           / ns_df.groupby(['pid', 'subject']).size()) * 100
perc_ns = perc_ns.to_frame().rename(columns={0: 'perc_mod'})
perc_ns['type'] = 'NS'
ws_df = merged_df[merged_df['type'] == 'WS']
perc_ws = (ws_df.groupby(['pid', 'subject']).sum(numeric_only=True)['modulated']
           / ws_df.groupby(['pid', 'subject']).size()) * 100
perc_ws = perc_ws.to_frame().rename(columns={0: 'perc_mod'})
perc_ws['type'] = 'WS'
perc_mod_df = pd.concat((perc_ws, perc_ns)).reset_index()
perc_mod_df = perc_mod_df.groupby(['subject', 'type']).mean().reset_index()

_, p = ttest_ind(perc_mod_df.loc[perc_mod_df['type'] == 'WS', 'perc_mod'],
                 perc_mod_df.loc[perc_mod_df['type'] == 'NS', 'perc_mod'])
print(f'p = {p:.2f}')

f, ax = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.swarmplot(data=perc_mod_df, x='type', y='perc_mod')
"""

# %%

merged_df = merged_df[merged_df['modulated']]
_, p = ttest_ind(merged_df.loc[merged_df['type'] == 'WS', 'mod_index_late'],
                 merged_df.loc[merged_df['type'] == 'NS', 'mod_index_late'])
print(f'p = {p:.2f}')

f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1.75), dpi=dpi)
sns.violinplot(data=merged_df, x='type', y='mod_index_late', ax=ax1,
               palette=[colors['NS'], colors['WS']], linewidth=1)
ax1.set(xticks=[], xlabel='', yticks=[-1, 0, 1])
ax1.set_ylabel('Modulation index', labelpad=0)
ax1.text(0.5, 1, 'n.s.', ha='center', va='center', fontsize=7)
sns.despine(trim=True, bottom=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'neuron_type_modulation.pdf'))