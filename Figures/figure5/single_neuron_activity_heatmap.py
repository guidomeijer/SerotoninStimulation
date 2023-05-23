#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, realpath, dirname, split
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from stim_functions import figure_style, paths

# Settings
T_BEFORE = 1  # for plotting
T_AFTER = 2
VMIN = 0
VMAX = 2

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
peths_df = pd.read_pickle(join(save_path, 'psth.pickle'))
peths_df = peths_df.reset_index(drop=True)  # make sure index is reset
peths_df = peths_df.sort_values(['high_level_region', 'modulation'], ascending=[True, False])  # sort by modulation

# Get array of all PETHs and select time limits
time_ax = peths_df['time'][0]
all_peth = np.array(peths_df['peth'].tolist())
all_peth = all_peth[:, (time_ax > -T_BEFORE) & (time_ax < T_AFTER)]
time_ax = time_ax[(time_ax > -T_BEFORE) & (time_ax < T_AFTER)]

# Do baseline normalization
norm_peth = np.empty(all_peth.shape)
for i in range(all_peth.shape[0]):
    norm_peth[i, :] = all_peth[i, :] / (np.mean(all_peth[i, time_ax < 0]) + 0.1)


# %%
# Plot per region
colors, dpi = figure_style()
f, ((ax_mb, ax_fc, ax_str, ax_th, ax_am),
    (ax_sc, ax_hc, ax_cb, ax_off1, ax_off2)) = plt.subplots(2, 5, figsize=(7, 3), sharex=True, dpi=dpi)
title_font = 7
cmap = 'PRGn'

these_peths = norm_peth[peths_df['high_level_region'] == 'Midbrain']
img = ax_mb.imshow(these_peths, cmap=cmap,
                   vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1.5, 1], interpolation='none')
ax_mb.add_patch(Rectangle((0, -1.5), 1, 2.5, color='royalblue', alpha=0.25, lw=0))
ax_mb.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_mb.set_title('Midbrain', fontsize=title_font)

these_peths = norm_peth[peths_df['high_level_region'] == 'Frontal cortex']
img = ax_fc.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1.5, 1], interpolation='none')
ax_fc.add_patch(Rectangle((0, -1.5), 1, 2.5, color='royalblue', alpha=0.25, lw=0))
ax_fc.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_fc.set_title('Frontal cortex', fontsize=title_font)

these_peths = norm_peth[peths_df['high_level_region'] == 'Striatum']
img = ax_str.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1.5, 1], interpolation='none')
ax_str.add_patch(Rectangle((0, -1.5), 1, 2.5, color='royalblue', alpha=0.25, lw=0))
ax_str.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_str.set_title('Striatum', fontsize=title_font)

these_peths = norm_peth[peths_df['high_level_region'] == 'Thalamus']
img = ax_th.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1.5, 1], interpolation='none')
ax_th.add_patch(Rectangle((0, -1.5), 1, 2.5, color='royalblue', alpha=0.25, lw=0))
ax_th.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_th.set_title('Thalamus', fontsize=title_font)

these_peths = norm_peth[peths_df['high_level_region'] == 'Amygdala']
img = ax_am.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1.5, 1], interpolation='none')
ax_am.add_patch(Rectangle((0, -1.5), 1, 2.5, color='royalblue', alpha=0.25, lw=0))
ax_am.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_am.set_title('Amygdala', fontsize=title_font)

these_peths = norm_peth[peths_df['high_level_region'] == 'Sensory cortex']
img = ax_sc.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1.5, 1], interpolation='none')
ax_sc.add_patch(Rectangle((0, -1.5), 1, 2.5, color='royalblue', alpha=0.25, lw=0))
ax_sc.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_sc.set_title('Sensory cortex', fontsize=title_font)

these_peths = norm_peth[peths_df['high_level_region'] == 'Hippocampus']
img = ax_hc.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1.5, 1], interpolation='none')
ax_hc.add_patch(Rectangle((0, -1.5), 1, 2.5, color='royalblue', alpha=0.25, lw=0))
ax_hc.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_hc.set_title('Hippocampus', fontsize=title_font)
ax_hc.xaxis.set_tick_params(which='both', labelbottom=True)


ax_off1.axis('off')
ax_off2.axis('off')
#ax_off3.axis('off')
#ax_off4.axis('off')
ax_cb.axis('off')

f.text(0.01, 0.5, 'Number of significantly modulated neurons', va='center', rotation='vertical')
f.text(0.25, 0.02, 'Time from stimulation start (s)', ha='center')

#plt.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.05, right=0.98, top=0.95, wspace=0.3, hspace=0)

cb_ax = f.add_axes([0.41, 0.15, 0.01, 0.25])
cbar = f.colorbar(mappable=ax_mb.images[0], cax=cb_ax)
cbar.ax.set_ylabel('FR / baseline', rotation=270, labelpad=10)
cbar.ax.set_yticks([0, 1, 2])

plt.savefig(join(fig_path, 'heatmap_per_region.pdf'), bbox_inches='tight')



