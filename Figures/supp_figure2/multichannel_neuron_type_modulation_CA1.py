#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:49:47 2022
By: Guido Meijer
"""

from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style, load_subjects
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison

# Settings
var = 'mod_index_late'
#var = 'mod_index_late'
MIN_NEURONS = 1

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
neuron_type = pd.read_csv(join(save_path, 'neuron_type_multichannel_CA1.csv'))
all_neurons = pd.merge(light_neurons, neuron_type, on=['subject', 'probe', 'eid', 'pid',
                                                       'neuron_id', 'region'])

# Select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[
        subjects['subject'] == nickname, 'sert-cre'].values[0]
all_neurons = all_neurons[all_neurons['sert-cre'] == 1]

# Drop neurons that could not be classified into a group
all_neurons = all_neurons[(all_neurons['type'] != 'Und.') & (~all_neurons['type'].isnull())]

# Select only modulated neurons
mod_neurons = all_neurons[all_neurons['modulated']]
#mod_neurons = all_neurons

# %% Visual cortex
# Get percentage of modulated neurons per animal per neuron type
#vis_neurons = all_neurons[np.in1d(all_neurons['region'], ['VISp'])]
ca1_neurons = all_neurons[np.in1d(all_neurons['region'], ['CA1'])]
perc_mod = ((ca1_neurons.groupby(['subject', 'type']).sum(numeric_only=True)['modulated']
            / ca1_neurons.groupby(['subject', 'type']).size()) * 100).to_frame()
perc_mod['n_neurons'] = ca1_neurons.groupby(['subject', 'type']).size()
perc_mod = perc_mod.rename(columns={0: 'percentage'}).reset_index()
perc_mod = perc_mod[perc_mod['n_neurons'] >= MIN_NEURONS]

# Select only modulated neurons
ca1_neurons = ca1_neurons[ca1_neurons['modulated']]

# Run ANOVA
mod = ols(f'{var} ~ type', data=ca1_neurons).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
mc = MultiComparison(ca1_neurons[var], ca1_neurons['type'])
tukey_mod = mc.tukeyhsd(alpha=0.05)
print(f'\nVisual cortex\nANOVA modulation p = {aov_table.loc["type", "PR(>F)"]}\n')
print(tukey_mod)

mod = ols('percentage ~ type', data=perc_mod).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
mc = MultiComparison(perc_mod['percentage'], perc_mod['type'])
tukey_perc = mc.tukeyhsd(alpha=0.05)
print(f'\nANOVA percentage p = {aov_table.loc["type", "PR(>F)"]}\n')
print(tukey_perc)

# %% Plot modulation
PROPS = {'boxprops':{'facecolor':'none', 'edgecolor':'none'}, 'medianprops':{'color':'none'},
         'whiskerprops':{'color':'none'}, 'capprops':{'color':'none'}}

colors, dpi = figure_style()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
plt.subplots_adjust(wspace=2)

sns.barplot(data=perc_mod, x='type', y='percentage', errorbar='se', 
            order=['NS', 'RS1', 'RS2'],
            palette=[colors['NS'], colors['RS1'], colors['RS2']], ax=ax1)
"""
sns.swarmplot(data=perc_mod_merged, x='region', y='percentage', hue='type',
              hue_order=['NS', 'RS1', 'RS2'], dodge=True, legend=False, size=3,
              palette=['gray', 'gray', 'gray'], ax=ax1)
"""
ax1.set(ylim=[0, 101], ylabel='Modulated neurons (%)', xlabel='')
ax1.legend(frameon=False, prop={'size': 5.5}, bbox_to_anchor=(0.8, 0.7))

sns.swarmplot(data=mod_neurons, x='type', y=var, order=['NS', 'RS1', 'RS2'], legend=None,
              size=3, hue='type', palette=[colors['NS'], colors['RS1'], colors['RS2']], ax=ax2,
              zorder=1)

sns.boxplot(x='type', y=var, ax=ax2, data=mod_neurons, showmeans=True,
            meanprops={"marker": "_", "markeredgecolor": "black", "markersize": "8"},
            order=['NS', 'RS1', 'RS2'], fliersize=0, zorder=2, **PROPS)

ax2.set(ylabel='Modulation index', ylim=[-1, 0.5], yticks=[-1, 0, 1], xlabel='')


plt.tight_layout()
sns.despine(trim=False)
plt.savefig(join(fig_path, 'perc_type_mod_CA1.pdf'))
