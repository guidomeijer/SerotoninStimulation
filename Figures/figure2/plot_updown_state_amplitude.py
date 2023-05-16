# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:08:47 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
updown_df = pd.read_csv(join(save_path, 'updown_amplitude_anesthetized.csv'))

# %% Do statistics

_, p = kruskal(*[group['psd_mean'].values for name, group in updown_df[updown_df['state'] == 'anesthetized'].groupby('region')])    
print(f'\nKruskal wallis p = {p}\n')
dunn_ph = sp.posthoc_conover(updown_df[updown_df['state'] == 'anesthetized'], val_col='psd_mean', group_col='region')
print(dunn_ph)        

region_order = ['Cortex', 'Striatum', 'Amygdala', 'Hippocampus', 'Thalamus', 'Midbrain']
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
sns.barplot(data=updown_df, y='psd_mean', x='region',
            errorbar='se', ax=ax1, order=region_order,
            color=colors['grey'])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='center')
ax1.tick_params(axis='x', which='major', pad=0)
#ax1.set(yscale='log')
ax1.set(ylabel='Power spectral density\n(0.1 - 0.5 Hz)', xlabel='')

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(join(fig_path, 'updown_states_power.pdf'))
