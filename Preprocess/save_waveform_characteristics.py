#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from stim_functions import paths, query_ephys_sessions, get_neuron_qc, get_artifact_neurons
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
OVERWRITE = True
_, save_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)

# Get artifact neurons
artifact_neurons = get_artifact_neurons()

if OVERWRITE:
    waveforms_df = pd.DataFrame()
else:
    waveforms_df = pd.read_pickle(join(save_path, 'waveform_metrics.p'))


# %% Functions


def gaus(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2 / (2 * sigma**2))


# %%
for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    if not OVERWRITE:
        if pid in waveforms_df['pid'].values:
            continue

    print(f'Starting {subject}, {date}')

    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    if 'acronym' not in clusters.keys():
       print(f'No brain regions found for {eid}')
       continue

    # Get data collection
    collections = one.list_collections(eid)
    if f'alf/{probe}/pykilosort' in collections:
        collection = f'alf/{probe}/pykilosort'
    else:
        collection = f'alf/{probe}'

    # Load in waveforms
    data = one.load_datasets(eid, datasets=['_phy_spikes_subset.waveforms', '_phy_spikes_subset.spikes',
                                            '_phy_spikes_subset.channels'],
                             collections=[collection]*3)[0]
    waveforms, wf_spikes, wf_channels = data[0], data[1], data[2]

    # Convert to uV
    waveforms = waveforms * 1000

    # Filter neurons that pass QC and exclude artifact neurons
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
        artifact_neurons['pid'] == pid, 'neuron_id'].values)]
    clusters_regions = clusters['acronym'][clusters_pass]

    # Loop over clusters
    for n, neuron_id in enumerate(clusters_pass):

        # Get mean waveform of channel with max amplitude
        n_waveforms = waveforms[spikes.clusters[wf_spikes] == neuron_id].shape[0]
        if n_waveforms == 0:
            continue

        mean_wf_ch = np.mean(waveforms[spikes.clusters[wf_spikes] == neuron_id], axis=0)
        mean_wf_ch = (mean_wf_ch
                      - np.tile(np.mean(mean_wf_ch, axis=0), (mean_wf_ch.shape[0], 1)))
        mean_wf = mean_wf_ch[:, np.argmin(np.min(mean_wf_ch, axis=0))]
        wf_time = np.linspace(0, (mean_wf.shape[0] / 30000) * 1000, mean_wf.shape[0])
        spike_amp = np.abs(np.min(mean_wf) - np.max(mean_wf))

        # Get peak-to-trough ratio
        pt_ratio = np.max(mean_wf) / np.abs(np.min(mean_wf))

        # Get peak minus through
        pt_subtract = np.max(mean_wf) - np.abs(np.min(mean_wf))

        # Get part of spike from trough to first peak after the trough
        peak_after_trough = np.argmax(mean_wf[np.argmin(mean_wf):]) + np.argmin(mean_wf)
        repolarization = mean_wf[np.argmin(mean_wf):np.argmax(mean_wf[np.argmin(mean_wf):]) + np.argmin(mean_wf)]

        # Get spike width in ms
        x_time = np.linspace(0, (mean_wf.shape[0] / 30000) * 1000, mean_wf.shape[0])
        peak_to_trough = ((np.argmax(mean_wf) - np.argmin(mean_wf)) / 30000) * 1000
        spike_width = ((peak_after_trough - np.argmin(mean_wf)) / 30000) * 1000

        # Get repolarization slope
        if spike_width <= 0.08:
            continue
        else:
            rp_slope, _, = np.polyfit(x_time[np.argmin(mean_wf):peak_after_trough],
                                      mean_wf[np.argmin(mean_wf):peak_after_trough], 1)

        # Get recovery slope
        rc_slope, _ = np.polyfit(x_time[peak_after_trough:], mean_wf[peak_after_trough:], 1)

        # Get firing rate
        neuron_fr = (np.sum(spikes['clusters'] == neuron_id)
                     / np.max(spikes['times']))

        # Get multichannel features
        these_channels = wf_channels[spikes.clusters[wf_spikes] == neuron_id][0, :]

        # Select channels on the side of the probe with the max amplitude
        if channels['lateral_um'][these_channels][0] > 35:
            lat_channels = channels['lateral_um'][these_channels] > 35
        elif channels['lateral_um'][these_channels][0] < 35:
            lat_channels = channels['lateral_um'][these_channels] < 35

        # Select channels within 100 um of soma
        ax_channels = np.abs(channels['axial_um'][these_channels]
                             - channels['axial_um'][these_channels[0]]) <= 100
        use_channels = lat_channels & ax_channels

        # Get distance to soma and sort channels accordingly
        dist_soma = np.sort(channels['axial_um'][these_channels[use_channels]]
                            - channels['axial_um'][these_channels[use_channels][0]])
        dist_soma = dist_soma / 1000  # convert to mm
        sort_ch = np.argsort(channels['axial_um'][these_channels[use_channels]]
                             - channels['axial_um'][these_channels[use_channels][0]])
        wf_ch_sort = mean_wf_ch[:, use_channels]
        wf_ch_sort = wf_ch_sort[:, sort_ch]
        wf_ch_sort = wf_ch_sort.T  # put time on the second dimension

        # Add to dataframe
        waveforms_df = pd.concat((waveforms_df, pd.DataFrame(index=[waveforms_df.shape[0] + 1], data={
            'pid': pid, 'eid': eid, 'probe': probe, 'subject': subject, 'waveform': [mean_wf],
            'neuron_id': neuron_id, 'acronym': clusters_regions[n], 'spike_amp': spike_amp,
            'pt_ratio': pt_ratio, 'rp_slope': rp_slope, 'pt_subtract': pt_subtract,
            'rc_slope': rc_slope, 'peak_to_trough': peak_to_trough, 'spike_width': spike_width,
            'firing_rate': neuron_fr, 'n_waveforms': n_waveforms, 'waveform_2D': [wf_ch_sort],
            'dist_soma': [dist_soma]})))

    waveforms_df.to_pickle(join(save_path, 'waveform_metrics.p'))

