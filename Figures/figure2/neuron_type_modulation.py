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
    merged_df.loc[merged_df['subject'] == nickname,
                  'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
merged_df = merged_df[merged_df['sert-cre'] == 1]

# Get percentage modulated neurons
ns_df = merged_df[merged_df['type'] == 'NS']
perc_ns = (ns_df.groupby(['subject']).sum(numeric_only=True)['modulated']
           / ns_df.groupby(['subject']).size()) * 100
perc_ns = perc_ns.to_frame().rename(columns={0: 'perc_mod'})
perc_ns['mod_index'] = ns_df.groupby(['subject']).mean(numeric_only=True)['mod_index']
perc_ns['type'] = 'NS'
ws_df = merged_df[merged_df['type'] == 'WS']
perc_ws = (ws_df.groupby(['subject']).sum(numeric_only=True)['modulated']
           / ws_df.groupby(['subject']).size()) * 100
perc_ws = perc_ws.to_frame().rename(columns={0: 'perc_mod'})
perc_ws['mod_index'] = ws_df.groupby(['subject']).mean(numeric_only=True)['mod_index']
perc_ws['type'] = 'WS'
perc_mod_df = pd.concat((perc_ws, perc_ns)).reset_index()
perc_mod_df = perc_mod_df.groupby(['subject', 'type']).mean().reset_index()

_, p = ttest_rel(perc_mod_df.loc[perc_mod_df['type'] == 'WS', 'perc_mod'],
                 perc_mod_df.loc[perc_mod_df['type'] == 'NS', 'perc_mod'])
print(f'p = {p:.3f}')

# %%

f, ax = plt.subplots(figsize=(1.5, 1.75), dpi=dpi)
for i, subject in enumerate(np.unique(perc_mod_df['subject'])):
    ns = perc_mod_df.loc[(perc_mod_df['subject'] == subject) &
                         (perc_mod_df['type'] == 'NS'), 'perc_mod']
    ws = perc_mod_df.loc[(perc_mod_df['subject'] == subject) &
                         (perc_mod_df['type'] == 'WS'), 'perc_mod']
    ax.plot([0, 1], [ns, ws], color='grey', lw=1)
    ax.plot(0, ns, marker='o', ms=4, markeredgewidth=0.4,
            markeredgecolor='w', markerfacecolor=colors['NS'])
    ax.plot(1, ws, marker='o', ms=4, markeredgewidth=0.4,
            markeredgecolor='w', markerfacecolor=colors['WS'])
#ax.text(0.5, 67, '*', fontsize=10, ha='center')
ax.text(0.7, 75, 'n.s.', fontsize=7, ha='center')
ax.set(ylabel='Modulated neurons (%)', xlabel='', yticks=np.arange(0, 81, 40), xlim=[-0.25, 1.25],
       xticks=[0, 1], xticklabels=['NS', 'WS'])
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'neuron_type_perc_mod.pdf'))

# %%

merged_df = merged_df[merged_df['modulated']]
_, p = ttest_ind(merged_df.loc[merged_df['type'] == 'WS', 'mod_index'],
                 merged_df.loc[merged_df['type'] == 'NS', 'mod_index'])
print(f'p = {p:.3f}')

f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1.75), dpi=dpi)
sns.violinplot(data=merged_df, x='type', y='mod_index', ax=ax1,
               palette=[colors['NS'], colors['WS']], linewidth=0.75)
ax1.set(xticks=[], xlabel='', yticks=[-1, 0, 1])
ax1.set_ylabel('Modulation index', labelpad=0)
ax1.text(0.5, 1, 'n.s.', ha='center', va='center', fontsize=7)
sns.despine(trim=True, bottom=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'neuron_type_modulation.pdf'))
