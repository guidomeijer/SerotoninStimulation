# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:23:17 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from stim_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                            get_artifact_neurons, get_neuron_qc, init_one, calculate_mi,
                            calculate_peths, high_level_regions, load_subjects)
from iblatlas.atlas import AllenAtlas
one = init_one()
ba = AllenAtlas()

# Settings
T_BEFORE = 1
T_AFTER = 4
BIN_SIZE = 0.05
OVERWRITE = False

# Query sessions
rec = query_ephys_sessions(one=one)

# Initialize
fig_path, save_path = paths()
subjects = load_subjects()
artifact_neurons = get_artifact_neurons()

if OVERWRITE:
    mi_df = pd.DataFrame()
else:
    mi_df = pd.read_csv(join(save_path, 'region_mutual_information.csv'))
    rec = rec[~np.isin(rec['eid'], mi_df['eid'])]

for i, eid in enumerate(np.unique(rec['eid'])[1:]):

    # Get session details
    subject = rec.loc[rec['eid'] == eid, 'subject'].values[0]
    date = rec.loc[rec['eid'] == eid, 'date'].values[0]
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'Starting {subject}, {date} [{i+1} of {len(np.unique(rec["eid"]))}]')

    # Load in laser pulse times
    opto_train_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_train_times) == 0:
        continue

    # Load in neural data of both probes
    region_spikes, region_spikes_bl = dict(), dict()
    for k, (pid, probe) in enumerate(zip(rec.loc[rec['eid'] == eid, 'pid'].values,
                                         rec.loc[rec['eid'] == eid, 'probe'].values)):

        try:
            sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
            spikes, clusters, channels = sl.load_spike_sorting()
            clusters = sl.merge_clusters(spikes, clusters, channels)
        except Exception as err:
            print(err)
            continue

        # Filter neurons that pass QC and artifact neurons
        qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
        clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
            artifact_neurons['pid'] == pid, 'neuron_id'].values)]
        clusters['region'] = high_level_regions(remap(clusters['acronym']))

        # Get spike counts per region
        for j, region in enumerate(np.unique(clusters['region'])):
            if region == 'root':
                continue

            # Get binned spike counts (trials x neurons x time)
            peths, binned_spikes = calculate_peths(
                spikes.times, spikes.clusters,
                clusters['cluster_id'][(clusters['region'] == region)
                                       & np.isin(clusters['cluster_id'], clusters_pass)],
                opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, 0)
            
            # Add to dictionary
            if (k == 1) & (region in region_spikes.keys()):
                # Concatenate activity from both probes that recorded the same region
                region_spikes[region] = np.concatenate((region_spikes[region], binned_spikes), axis=1)
            else:
                region_spikes[region] = binned_spikes
            
            # Get time scale
            tscale = peths['tscale']
    
    # Get pairwise neural correlations between all neuron pairs in both regions
    these_regions = list(region_spikes.keys())
    for r1, region_1 in enumerate(these_regions[:-1]):
        for r2, region_2 in enumerate(these_regions[r1+1:]):
            print(f'Starting {region_1} - {region_2}')
            
            # Loop over timebins
            mi_over_time = np.empty(tscale.shape[0])
            for tt in range(tscale.shape[0]):
                
                # Loop over pairwise neurons
                pairwise_mi = []
                for n1 in range(region_spikes[region_1].shape[1]):  # Neurons in region 1
                    for n2 in range(region_spikes[region_2].shape[1]):  # Neurons in region 2
                    
                        # Calculate mutual information for this pair of neurons
                        this_mi = calculate_mi(region_spikes[region_1][:, n1, tt],
                                               region_spikes[region_2][:, n2, tt])
                        pairwise_mi.append(this_mi)
                mi_over_time[tt] = np.mean(pairwise_mi)
                        
            # Add to dataframe
            mi_df = pd.concat((mi_df, pd.DataFrame(data={
                'mutual_information': mi_over_time, 'time': tscale,
                'region_1': region_1, 'region_2': region_2,
                'region_pair': f'{region_1}-{region_2}', 'subject': subject,
                'eid': eid, 'date': date, 'sert-cre': sert_cre})), ignore_index=True)

    # Save to disk
    mi_df.to_csv(join(save_path, 'region_mutual_information.csv'), index=False)