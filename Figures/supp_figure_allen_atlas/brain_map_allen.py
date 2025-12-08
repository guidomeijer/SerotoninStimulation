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
from stim_functions import paths, figure_style, plot_scalar_on_slice, remap
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas(res_um=10)

# Settings
AP = [2, -1.5, -3.5]

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


# %%

colors, dpi = figure_style()
CMAP = 'turbo'

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=dpi)

plot_scalar_on_slice(reg_neurons['region'].values, np.log10(reg_neurons['n_neurons'].values), ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, 3])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, np.log10(reg_neurons['n_neurons'].values), ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, 3])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, np.log10(reg_neurons['n_neurons'].values), ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, 3])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.2])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Total number of\nrecorded neurons (log)', rotation=270, labelpad=16)
cbar.ax.set_yticks([0, 1, 2, 3])
cbar.ax.set_yticklabels([1, 10, 100, 1000])

plt.savefig(join(fig_path, 'brain_map_n_neurons.pdf'))

# %%

CMAP = 'YlOrRd'

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=dpi)

plot_scalar_on_slice(sim_neurons['region'].values, sim_neurons['n_neurons'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, 60])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(sim_neurons['region'].values, sim_neurons['n_neurons'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, 60])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(sim_neurons['region'].values, sim_neurons['n_neurons'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap=CMAP, clevels=[0, 60])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.2])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Simultaneously recorded\nneurons (median)', rotation=270, labelpad=16)
# cbar.ax.set_yticks([0, 1, 2, 3])
# cbar.ax.set_yticklabels([1, 10, 100, 1000])

plt.savefig(join(fig_path, 'brain_map_n_simultaneous_neurons.pdf'))
