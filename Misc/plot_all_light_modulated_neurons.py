# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:05:36 2023

By Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from brainbox.plot import peri_event_time_histogram
from brainbox.singlecell import calculate_peths
from stim_functions import paths, load_passive_opto_times, figure_style
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
#PATH = 'C:\\Users\\Guido1\\Figures\\5-HT\\SingleNeurons'
PATH = r'C:\Users\guido\Figures\5HT\Ephys\SingleNeurons\LightModNeurons'
T_BEFORE = 1  # for plotting
T_AFTER = 2
BIN_SIZE = 0.05
SMOOTHING = 0.025
PLOT_LATENCY = True
_, save_path = paths()

# Load in data
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

for i, pid in enumerate(np.unique(all_neurons['pid'])):

    # Get eid
    eid = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'eid'])[0]
    probe = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'probe'])[0]
    subject = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'date'])[0]
    print(f'Starting {subject}, {date}, {probe}')

    # Load in laser pulse times
    opto_train_times, _ = load_passive_opto_times(eid, one=one)

    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    if 'acronym' not in clusters.keys():
        print(f'No brain regions found for {eid}')
        continue

    # Take slice of dataframe
    modulated = all_neurons[(all_neurons['pid'] == pid) & (all_neurons['modulated'] == 1)]

    for n, ind in enumerate(modulated.index.values):
        region = modulated.loc[ind, 'region']
        subject = modulated.loc[ind, 'subject']
        date = modulated.loc[ind, 'date']
        neuron_id = modulated.loc[ind, 'neuron_id']

        # Plot PSTH
        colors, dpi = figure_style()
        p, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
        ax.add_patch(Rectangle((0, 0), 1, 100, color='royalblue', alpha=0.25, lw=0))
        ax.add_patch(Rectangle((0, 0), 1, -100, color='royalblue', alpha=0.25, lw=0))
        try:
            peri_event_time_histogram(spikes.times, spikes.clusters, opto_train_times,
                                      neuron_id, t_before=T_BEFORE, t_after=T_AFTER, smoothing=SMOOTHING,
                                      bin_size=BIN_SIZE, include_raster=True, error_bars='sem', ax=ax,
                                      pethline_kwargs={'color': 'black', 'lw': 1},
                                      errbar_kwargs={'color': 'black', 'alpha': 0.3},
                                      raster_kwargs={'color': 'black', 'lw': 0.3},
                                      eventline_kwargs={'lw': 0})
        except Exception as err:
            print(err)
            continue
        ax.set(ylabel='Firing rate (spks/s)', xlabel='Time (s)',
               yticks=[np.round(ax.get_ylim()[1])],
               ylim=[ax.get_ylim()[0], np.round(ax.get_ylim()[1])])
        # ax.plot([0, 1], [0, 0], lw=2.5, color='royalblue')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        if PLOT_LATENCY & ~np.isnan(modulated.loc[ind, 'latenzy']):

            peths, _ = calculate_peths(spikes.times, spikes.clusters, [neuron_id],
                                       opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
            peak_ind = np.argmin(
                np.abs(peths['tscale'] - modulated.loc[ind, 'latenzy']))
            peak_act = peths['means'][0][peak_ind]
            ax.plot([modulated.loc[ind, 'latenzy'], modulated.loc[ind, 'latenzy']],
                    [peak_act, peak_act], 'xr', lw=2)

        plt.tight_layout()

        plt.savefig(
            join(PATH, f'{region}_{subject}_{date}_{probe}_neuron{neuron_id}.jpg'), dpi=300)
        plt.close(p)
