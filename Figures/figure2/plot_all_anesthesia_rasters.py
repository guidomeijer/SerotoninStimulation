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
from brainbox.io.one import SpikeSortingLoader
from brainbox.processing import bincount2D
from stim_functions import (query_ephys_sessions, high_level_regions, get_neuron_qc, remap,
                            figure_style, load_passive_opto_times, paths)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

MIN_NEURONS = 5
BIN_SIZE = 0.2
SMOOTHING = 0.05
DURATION = 360  # s
RANGE = [0.2, 1]  # Hz

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

# Loop over recordings
for i, pid in enumerate(rec['pid']):
        
    # Load opto times
    opto_times, _ = load_passive_opto_times(one.pid2eid(pid)[0])
    
    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    if len(spikes) == 0:
        continue
    
    # Only keep IBL good neurons
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters.cluster_id[clusters.label == 1])]
    spikes.depths = spikes.depths[np.isin(spikes.clusters, clusters.cluster_id[clusters.label == 1])]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters.cluster_id[clusters.label == 1])]
    
    # Get spike raster
    iok = ~np.isnan(spikes.depths)
    R, times, depths = bincount2D(spikes.times[iok], spikes.depths[iok], 0.01, 20, weights=None)
        
    # Plot figure
    f, ax1 = plt.subplots(1, 1, figsize=(5, 2.5), dpi=300)
    ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R) * 2,
              extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='lower')
    ax1.invert_yaxis()
    
    ax1.set(xlim=[spikes.times[-1]-360, spikes.times[-1]], ylim=[0, 4000])
    
    ch_depths = (np.flip(channels['axial_um']) + 60)
    for k in range(0, channels['acronym'].shape[0], 20):
        ax1.text(ax1.get_xlim()[1]+5, ch_depths[k], channels['acronym'][k], fontsize=5)
        
    plt.tight_layout()
    sns.despine(trim=True, offset=4)
    plt.show()    
    


