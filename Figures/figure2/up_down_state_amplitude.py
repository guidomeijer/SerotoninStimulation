# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:46:43 2023

@author: Guido
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, realpath, dirname, split
from brainbox.singlecell import calculate_peths
from brainbox.io.one import SpikeSortingLoader
from scipy.signal import welch
import scikit_posthocs as sp
from scipy.stats import kruskal
from stim_functions import (query_ephys_sessions, high_level_regions, get_neuron_qc, remap,
                            figure_style, load_passive_opto_times, paths)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

MIN_NEURONS = 5
BIN_SIZE = 0.2
SMOOTHING = 0.2
RANGE = [0.25, 0.5]  # Hz

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Query sessions
rec_both = query_ephys_sessions(anesthesia='both', one=one)
rec_both['anesthesia'] = 'both'
rec_anes = query_ephys_sessions(anesthesia='yes', one=one)
rec_anes['anesthesia'] = 'yes'
rec = pd.concat((rec_both, rec_anes)).reset_index(drop=True)

updown_psd_df = pd.DataFrame()
updown_max_df = pd.DataFrame()
for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    print(f'{subject}, {date}')
    
    # Load opto times
    if rec.loc[i, 'anesthesia'] == 'yes': 
        opto_times, _ = load_passive_opto_times(eid, one=one)
    elif rec.loc[i, 'anesthesia'] == 'both':
        opto_times, _ = load_passive_opto_times(eid, anesthesia=True, one=one)
        
    # Load in neural data
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    
    # Filter neurons that pass QC
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    
    # Remap to high level regions
    clusters.regions = high_level_regions(remap(clusters.acronym), merge_cortex=True)

    for j, region in enumerate(np.unique(clusters.regions)):

        # Get spikes in region
        region_spikes = spikes.times[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == region])]
        region_clusters = spikes.clusters[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == region])]
        if (np.unique(region_clusters).shape[0] < MIN_NEURONS) | (region == 'root'):
            continue

        # Get smoothed firing rates    
        peth, _ = calculate_peths(region_spikes, region_clusters, np.unique(region_clusters),
                                  [opto_times[0]-360], pre_time=0, post_time=360,
                                  bin_size=BIN_SIZE, smoothing=SMOOTHING)
        tscale = peth['tscale'] + spikes.times[0]            
        pop_act = peth['means'].T
        
        # Calculate power spectrum
        freq, psd = welch(np.mean(pop_act, axis=1), fs=1/BIN_SIZE)
        
        # Add to dataframe
        updown_psd_df = pd.concat((updown_psd_df, pd.DataFrame(data={
            'freq': freq, 'psd': psd, 'region': region, 'subject': subject, 'date': date})))
        updown_max_df = pd.concat((updown_max_df, pd.DataFrame(index=[updown_max_df.shape[0]+1], data={
            'psd_max': np.max(psd[(freq >= RANGE[0]) & (freq <= RANGE[1])]),
            'psd_mean': np.mean(psd[(freq >= RANGE[0]) & (freq <= RANGE[1])]),
            'region': region, 'subject': subject, 'date': date})))
  
# %% Do statistics

_, p = kruskal(*[group["psd_max"].values for name, group in updown_max_df.groupby("region")])    
print(f'\nKruskal wallis p = {p}\n')
dunn_ph = sp.posthoc_conover(updown_max_df, val_col='psd_max', group_col='region')
print(dunn_ph)        

# %% Plot
region_order = ['Cortex', 'Amygdala', 'Striatum', 'Hippocampus', 'Thalamus', 'Midbrain']
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
sns.barplot(data=updown_max_df, y='psd_max', x='region', errorbar='se', ax=ax1, order=region_order,
            color=colors['grey'])
ax1.set(ylabel='Power spectral density', xlabel='', ylim=[0, 30], yticks=np.arange(0, 31, 10))
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(join(fig_path, 'updown_states_power.pdf'))

