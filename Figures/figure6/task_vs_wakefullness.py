# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:48:18 2023 by Guido Meijer
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
awake_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
task_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))

# Merge dataframes
awake_neurons = awake_neurons[awake_neurons['modulated']]
task_neurons = task_neurons[task_neurons['opto_modulated']]
merged_df = pd.merge(awake_neurons, task_neurons, on=[
    'pid', 'neuron_id', 'eid', 'subject', 'date', 'probe', 'region'])
merged_df['mod_index_late_abs'] = merged_df['mod_index_late'].abs()
merged_df['opto_mod_roc_abs'] = merged_df['opto_mod_roc'].abs()

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.plot([-1, 1], [-1, 1], color='grey', ls='--', zorder=0)
sns.regplot(data=merged_df, x='mod_index_late', y='opto_mod_roc', ax=ax1, ci=None, color='k',
            line_kws={'color': 'tab:red'}, scatter_kws={'lw': 0})
ax1.set(xlim=[-0.8, 0.6], ylim=[-0.8, 0.6], ylabel='Task modulation',
        yticks=np.arange(-0.8, 0.61, 0.4), xticks=np.arange(-0.8, 0.61, 0.4),
        xlabel='Quiet wakefullness\nmodulation')
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'task_vs_passive_modulation.pdf'))

# %%
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.plot([0, 0.6], [0, 0.6], color='grey', ls='--', zorder=0)
sns.regplot(data=merged_df, x='mod_index_late_abs', y='opto_mod_roc_abs', ax=ax1, ci=None, color='k',
            line_kws={'color': 'tab:red'}, scatter_kws={'lw': 0})
ax1.set(ylabel='Task modulation', xlabel='Quiet wakefullness\nmodulation')
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'task_vs_passive_abs_modulation.pdf'))


