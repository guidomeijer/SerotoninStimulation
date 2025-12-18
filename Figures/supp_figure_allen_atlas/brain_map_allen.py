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
from stim_functions import paths, figure_style, plot_scalar_on_slice, remap, load_subjects
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas(res_um=10)
colors, dpi = figure_style()

# Settings
AP = [2, -1.5, -3.5]
MIN_NEURONS = 5

# Paths
f_path, data_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in expression
expr_df = pd.read_csv(join(data_path, 'receptor_expression.csv'))
expr_df = expr_df[~np.isin(expr_df['acronym'], ['MMme', 'CUL4, 5', 'CUL4, 5gr', 'CUL4, 5mo'])]
expr_df['region'] = remap(expr_df['acronym'])
#expr_df['region'] = combine_regions(remap(expr_df['acronym']))
expression_mean = expr_df[['region', 'receptor', 'expression_energy']].groupby(
    ['region', 'receptor']).median().reset_index()

# load structure and expression data set
proj_df = pd.read_csv(join(data_path, 'dr_projection_strength.csv'))
proj_df = proj_df[~np.isin(proj_df['allen_acronym'], ['MMd', 'MMme', 'MMl', 'MMm', 'MMp', 'CUL4, 5'])]
proj_df['region'] = remap(proj_df['allen_acronym'])
#proj_df['region'] = combine_regions(remap(proj_df['allen_acronym']))
proj_summary = proj_df[['region', 'projection_density']].groupby(['region']).mean().reset_index()

# Get percentage of modulated neurons
light_neurons = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    light_neurons.loc[light_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
summary_df = light_neurons[light_neurons['sert-cre'] == 1].groupby(['region']).size().to_frame()
summary_df = summary_df.rename(columns={0: 'n_neurons'})
summary_df['modulation_index'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(
    ['region']).mean(numeric_only=True)['mod_index']
summary_df['latency'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(
    ['region']).median(numeric_only=True)['latenzy']
summary_df['modulated'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(
    ['region']).sum(numeric_only=True)['modulated']
summary_df = summary_df.reset_index()
summary_df['perc_mod'] =  (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['n_neurons'] >= MIN_NEURONS]


# %%

CMAP = 'magma'
MAX_C = 0.1
C_TICKS = 0.02

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=600)

plot_scalar_on_slice(proj_summary['region'].values, proj_summary['projection_density'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(proj_summary['region'].values, proj_summary['projection_density'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(proj_summary['region'].values, proj_summary['projection_density'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.4])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Projection density', rotation=270, labelpad=16)
cbar.ax.set_yticks(np.arange(MAX_C+C_TICKS, step=C_TICKS))
cbar.ax.set_yticklabels(np.arange(MAX_C+C_TICKS, step=C_TICKS))

plt.savefig(join(fig_path, 'projection_density.pdf'))

# %%

CMAP = 'magma'
MAX_C = 8
C_TICKS = 2
this_df = expression_mean[expression_mean['receptor'] == '5-HT2a']

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=600)

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.4])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Expression energy', rotation=270, labelpad=16)
cbar.ax.set_yticks(np.arange(MAX_C+C_TICKS, step=C_TICKS))
cbar.ax.set_yticklabels(np.arange(MAX_C+C_TICKS, step=C_TICKS))

plt.savefig(join(fig_path, '5HT2a_map.pdf'))

# %%

CMAP = 'magma'
MAX_C = 6
C_TICKS = 2
this_df = expression_mean[expression_mean['receptor'] == '5-HT2c']

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=600)

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.4])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Expression energy', rotation=270, labelpad=16)
cbar.ax.set_yticks(np.arange(MAX_C+C_TICKS, step=C_TICKS))
cbar.ax.set_yticklabels(np.arange(MAX_C+C_TICKS, step=C_TICKS))

plt.savefig(join(fig_path, '5HT2c_map.pdf'))


# %%

CMAP = 'magma'
MAX_C = 4
C_TICKS = 1
this_df = expression_mean[expression_mean['receptor'] == '5-HT1a']

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=600)

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.4])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Expression energy', rotation=270, labelpad=16)
cbar.ax.set_yticks(np.arange(MAX_C+C_TICKS, step=C_TICKS))
cbar.ax.set_yticklabels(np.arange(MAX_C+C_TICKS, step=C_TICKS))

plt.savefig(join(fig_path, '5HT1a_map.pdf'))


# %%

CMAP = 'magma'
MAX_C = 0.5
C_TICKS = 0.1
this_df = expression_mean[expression_mean['receptor'] == '5-HT1b']

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=600)

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(this_df['region'].values, this_df['expression_energy'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.4])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Expression energy', rotation=270, labelpad=16)
cbar.ax.set_yticks(np.arange(MAX_C+C_TICKS, step=C_TICKS))
cbar.ax.set_yticklabels(np.arange(MAX_C+C_TICKS, step=C_TICKS))

plt.savefig(join(fig_path, '5HT1b_map.pdf'))


# %%

CMAP = 'magma'
MAX_C = 60
C_TICKS = 20

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=600)

plot_scalar_on_slice(summary_df['region'].values, summary_df['perc_mod'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(summary_df['region'].values, summary_df['perc_mod'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(summary_df['region'].values, summary_df['perc_mod'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.3])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('5-HT modulated neurons (%)', rotation=270, labelpad=16)
cbar.ax.set_yticks(np.arange(MAX_C+C_TICKS, step=C_TICKS))
cbar.ax.set_yticklabels(np.arange(MAX_C+C_TICKS, step=C_TICKS))

plt.savefig(join(fig_path, 'perc_mod_map.pdf'))


# %%

CMAP = 'coolwarm'

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=600)

plot_scalar_on_slice(summary_df['region'].values, summary_df['modulation_index'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap=CMAP, clevels=[-0.1, 0.1])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(summary_df['region'].values, summary_df['modulation_index'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap=CMAP, clevels=[-0.1, 0.1])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(summary_df['region'].values, summary_df['modulation_index'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap=CMAP, clevels=[-0.1, 0.1])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.3])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Modulation index', rotation=270, labelpad=16)
cbar.ax.set_yticks([-0.1, 0, 0.1])
cbar.ax.set_yticklabels([-0.1, 0, 0.1])

plt.savefig(join(fig_path, 'modulation_index_map.pdf'))

# %%

CMAP = 'magma_r'
MAX_C = .8
C_TICKS = [0, .4, .8]

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=600)

plot_scalar_on_slice(summary_df['region'].values, summary_df['latency'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(summary_df['region'].values, summary_df['latency'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(summary_df['region'].values, summary_df['latency'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, MAX_C])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.3])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Modulation latency (s)', rotation=270, labelpad=16)
cbar.ax.set_yticks(C_TICKS)
cbar.ax.set_yticklabels(C_TICKS)

plt.savefig(join(fig_path, 'latency_map.pdf'))