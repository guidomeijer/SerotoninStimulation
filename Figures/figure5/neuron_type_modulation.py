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
    merged_df.loc[merged_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# %%
_, p = ttest_ind(merged_df.loc[merged_df['type'] == 'WS', 'mod_index_late'],
                 merged_df.loc[merged_df['type'] == 'NS', 'mod_index_late'])
print(f'p = {p:.2f}')

colors, dpi = figure_style()

f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1.75), dpi=dpi)
sns.violinplot(data=merged_df, x='type', y='mod_index_late', ax=ax1,
               palette=[colors['NS'], colors['WS']], linewidth=0)
ax1.set(xticks=[], xlabel='', yticks=[-1, 0, 1])
ax1.set_ylabel('Modulation index', labelpad=0)
ax1.text(0.5, 1, 'n.s.', ha='center', va='center', fontsize=7)
sns.despine(trim=True, bottom=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'neuron_type_modulation.pdf'))