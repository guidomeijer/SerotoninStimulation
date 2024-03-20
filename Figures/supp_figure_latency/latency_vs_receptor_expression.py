#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:04:32 2023
By: Guido Meijer
"""


import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from scipy.stats import pearsonr
from stim_functions import paths, figure_style, load_subjects, combine_regions

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
receptor_expr = pd.read_csv(join(save_path, '5HT_receptor_expressions.csv'))

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only modulated neurons in sert-cre mice
sert_neurons = all_neurons[(all_neurons['sert-cre'] == 1) & (all_neurons['modulated'] == 1)]

# Transform to ms
sert_neurons['latency'] = sert_neurons['latency_peak'] * 1000

# Get mean per area
sert_neurons['abr_region'] = combine_regions(sert_neurons['region'], abbreviate=True)
sert_neurons['region'] = combine_regions(sert_neurons['region'])
sert_neurons = sert_neurons[sert_neurons['region'] != 'root']
per_region = sert_neurons.groupby(['region', 'abr_region']).mean(numeric_only=True).reset_index()
grouped_df = pd.merge(per_region, receptor_expr, on=['region'])

# %% Plot

# Add colormap
colors, dpi = figure_style()
grouped_df['color'] = [colors[i] for i in grouped_df['region']]
this_df = grouped_df[grouped_df['receptor_name'] == 'htr3a']

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
(
     so.Plot(this_df, x='expression_energy', y='latency')
     .add(so.Dot(pointsize=0))
     .add(so.Line(color='grey', linewidth=1), so.PolyFit(order=1))
     .on(ax1)
     .plot()
)
for i in this_df.index:
    ax1.text(this_df.loc[i, 'expression_energy'] ,
             this_df.loc[i, 'latency'],
             this_df.loc[i, 'abr_region'],
             ha='center', va='center',
             color=this_df.loc[i, 'color'], fontsize=4.5, fontweight='bold')
ax1.set(yticks=[100, 200, 300, 400, 500, 600], xticks=[0, 0.5, 1],
        ylabel='Modulation latency (ms)', xlabel='5HT3a expression energy')
r, p = pearsonr(this_df['expression_energy'], this_df['latency'])
#ax1.text(0.1, 100, f'r = {r:.2f}', fontsize=6)
ax1.text(0.5, 520, '*', fontsize=10, ha='center')

sns.despine(offset=2, trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'expression_vs_latency.pdf'))



