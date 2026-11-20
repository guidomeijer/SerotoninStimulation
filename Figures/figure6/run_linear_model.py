# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 03/03/2026
"""
# %%

import numpy as np
import pandas as pd
from os.path import join
import statsmodels.api as sm
import statsmodels.formula.api as smf
from brainbox.io.one import SpikeSortingLoader
from brainbox.population.decode import get_spike_counts_in_bins
from stim_functions import query_ephys_sessions, init_one, load_trials, get_neuron_qc, paths
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = init_one()
_, save_path = paths()

# Settings
TIME_WIN = [0.3, 0.8]
ONLY_SIG = False

# Load in neurons
sig_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Query sessions
rec = query_ephys_sessions(one=one)

linear_df = pd.DataFrame()
for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    print(f'\nStarting {i} of {len(rec)}: {eid} {subject}, {date}')

    # Load trials dataframe
    try:
        trials_df = load_trials(eid, laser_stimulation=True, one=one)
    except:
        print('Could not load trials')
        continue

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    if 'acronym' not in clusters.keys():
        print(f'No brain regions found for {eid}')
        continue

    # Filter neurons that pass QC
    these_neurons = sig_neurons[sig_neurons['pid'] == pid]
    if ONLY_SIG:
        clusters_pass = these_neurons.loc[these_neurons['modulated'], 'neuron_id'].values
    else:
        clusters_pass = these_neurons['neuron_id'].values
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    if len(spikes.clusters) == 0:
        continue

    # Get spike count array
    stim_intervals = np.column_stack(((trials_df['stimOn_times'] + TIME_WIN[0]), (trials_df['stimOn_times'] + TIME_WIN[1])))
    spike_counts, neuron_ids = get_spike_counts_in_bins(spikes.times, spikes.clusters, stim_intervals)

    # Loop over neurons
    for k, neuron_id in enumerate(neuron_ids):

        # Build GLM input dataframe
        glm_df = pd.DataFrame(data={
            'spike_counts': spike_counts[k, :],
            'choice': trials_df['choice'].values.astype(int),  # -1 = left, 1 = right
            'stim': trials_df['laser_stimulation'].values.astype(int),  # 0 = no stim, 1 = stim
        })

        # Drop trials with no choice
        glm_df = glm_df.dropna()

        # Fit GLM
        try:
            model = smf.glm(formula='spike_counts ~ choice + stim + choice:stim', data=glm_df,
                            family=sm.families.Poisson()).fit()
        except Exception:
            continue

        # Add to dataframe
        linear_df = pd.concat((linear_df, pd.DataFrame(index=[0], data={
            'coef_choice': model.params['choice'],
            'coef_stim': model.params['stim'],
            'coef_interaction': model.params['choice:stim'],
            'p_choice': model.pvalues['choice'],
            'p_stim': model.pvalues['stim'],
            'p_interaction': model.pvalues['choice:stim'],
            'eid': eid, 'pid': pid, 'probe': probe, 'subject': subject, 'date': date,
            'neuron_id': neuron_id, 'acronym': clusters.acronym[clusters.cluster_id == neuron_id],
        })), ignore_index=True)

# Save results
if ONLY_SIG:
    linear_df.to_csv(join(save_path, 'linear_model_results_sig.csv'), index=False)
else:
    linear_df.to_csv(join(save_path, 'linear_model_results_3-8.csv'), index=False)
