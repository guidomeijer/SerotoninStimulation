#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style, high_level_regions, combine_regions, remap
from scipy.stats import kstest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import pandas as pd

# Settings
SW_CUTOFF = 0.35
FEATURES = ['spike_width', 'pt_ratio']

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])


def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


# Load in waveforms
waveforms_df = pd.read_pickle(join(save_path, 'waveform_metrics.p'))
waveforms_df['combined_region'] = combine_regions(remap(waveforms_df['acronym']))
waveforms_df['high_level_region'] = high_level_regions(waveforms_df['acronym'], merge_cortex=True)
waveforms_df = waveforms_df[waveforms_df['high_level_region'] != 'root']

# Exclude positive spikes
waveforms_df = waveforms_df[waveforms_df['pt_subtract'] <= -0.025]
waveforms_df = waveforms_df.reset_index(drop=True)

# Plot spike width distributions per region
colors, dpi = figure_style()
f, axs = plt.subplots(2, 3, figsize=(7, 3), dpi=dpi)
bins = {'Amygdala': 30, 'Cortex': 65, 'Hippocampus': 40, 'Midbrain': 30, 'Striatum': 35, 'Thalamus': 22}
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(waveforms_df['high_level_region'])):
    axs[i].hist(waveforms_df.loc[waveforms_df['high_level_region'] == region, 'spike_width'],
                bins=bins[region])
    axs[i].set(xlabel='Spike width (ms)', ylabel='Neuron count', title=f'{region}', xlim=[0, 1])
sns.despine(trim=True)
plt.tight_layout()

# Drop all regions that do not have narrow-spiking interneurons
waveforms_df = waveforms_df[(waveforms_df['high_level_region'] != 'Amygdala')
                            & (waveforms_df['high_level_region'] != 'Midbrain')
                            & (waveforms_df['high_level_region'] != 'Thalamus')]

# Cutoff
waveforms_df.loc[waveforms_df['spike_width'] < SW_CUTOFF, 'type'] = 'NS'
waveforms_df.loc[waveforms_df['spike_width'] >= SW_CUTOFF, 'type'] = 'WS'

perc_reg = (np.sum(waveforms_df["type"] == "WS") / waveforms_df.shape[0]) * 100
perc_fast = (np.sum(waveforms_df["type"] == "NS") / waveforms_df.shape[0]) * 100
print(f'{perc_fast:.2f}% fast spiking')
print(f'{perc_reg:.2f}% regural spiking')

# Save result
neuron_type = waveforms_df.copy()
neuron_type = neuron_type.drop(['waveform', 'spike_width', 'rp_slope', 'spike_amp', 'pt_ratio',
                                'rc_slope', 'pt_subtract', 'peak_to_trough', 'n_waveforms',
                                'waveform_2D'], axis=1)
neuron_type.to_csv(join(save_path, 'neuron_type.csv'), index=False)

_, p_value = kstest(waveforms_df.loc[waveforms_df['type'] == 'WS', 'firing_rate'],
                    waveforms_df.loc[waveforms_df['type'] == 'NS', 'firing_rate'])
print(f'KS-test p-value: {p_value}')

# %% Plot mean waveforms
time_ax = np.linspace(0, (waveforms_df.loc[waveforms_df.index[0], 'waveform'].shape[0]/30000)*1000,
                      waveforms_df.loc[waveforms_df.index[0], 'waveform'].shape[0])

f, ax = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
ax.plot(time_ax, waveforms_df.loc[waveforms_df['type'] == 'WS', 'waveform'].to_numpy().mean(),
         color=colors['WS'], label='WS')
ax.plot(time_ax, waveforms_df.loc[waveforms_df['type'] == 'NS', 'waveform'].to_numpy().mean(),
         color=colors['NS'], label='NS')
#ax.plot([0.1, 0.1], [-0.18, -0.08], color='k', lw=0.5)
#ax.plot([0.1, 1.1], [-0.18, -0.18], color='k', lw=0.5)
#ax.text(-0.25, -0.16, '0.1 mV', rotation=90)
#ax.text(0.25, -0.21, '1 ms')
#ax.set(xlim=[0, 3], ylim=[-0.3, 0.101])
ax.axis('off')

#plt.tight_layout()
#sns.despine(trim=True)
plt.savefig(join(fig_path, 'mean_waveforms.pdf'), bbox_inches='tight')


# %% Plot waveform histogram
f, ax = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
hst = sns.histplot(data=waveforms_df, x='spike_width', hue='type', hue_order=['NS', 'WS'],
                   palette=[colors['NS'], colors['WS']], legend='brief', multiple='stack', bins=37)
ax.set(xlim=[0, 1.2], xticks=[0, 0.6, 1.2], xlabel='Spike width (ms)', ylabel='Neuron count',
       ylim=[0, 400], yticks=[0, 400])
ax.yaxis.labelpad = -7
hst.get_legend().set_frame_on(False)
hst.get_legend().set_title('')
hst.get_legend().set_bbox_to_anchor((0.55, 0.6))

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'waveform_histogram.pdf'))
plt.savefig(join(fig_path, 'waveform_histogram.jpg'), dpi=600)

# %% Plot firing rate distribution of clusteWS

f, ax = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
ax.hist(waveforms_df.loc[waveforms_df['type'] == 'NS', 'firing_rate'], histtype='step',
         color=colors['NS'], density=True, bins=100, cumulative=True, label='Narrow spiking (NS)')
ax.hist(waveforms_df.loc[waveforms_df['type'] == 'WS', 'firing_rate'], histtype='step',
         color=colors['WS'], density=True, bins=100, cumulative=True, label='Regular spiking (WS)')
ax.set(xlabel='Firing rate (spks/s)', ylabel='Density', xticks=[0, 50, 100])
ax.text(50, 0.8, '***', fontsize=10, ha='center', va='center')
custom_lines = [Line2D([0], [0], color=colors['NS'], lw=1),
                Line2D([0], [0], color=colors['WS'], lw=1)]
ax.legend(custom_lines, ['NS', 'WS'])
fix_hist_step_vertical_line_at_end(ax)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'firing_rate_dist.pdf'))







