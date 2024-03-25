#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:03:53 2021
By: Guido Meijer
"""

import numpy as np
np.random.seed(42)
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import seaborn as sns
from zetapy import zetatest2
import statsmodels.api as sm
from statsmodels.formula.api import ols
from brainbox.io.one import SpikeSortingLoader
from brainbox.task.closed_loop import (responsive_units, roc_single_event, differentiate_units,
                                       roc_between_two_events, generate_pseudo_blocks)
from stim_functions import (paths, remap, query_ephys_sessions, load_trials, figure_style,
                            get_neuron_qc, peri_multiple_events_time_histogram,
                            remove_artifact_neurons, init_one, calculate_peths)
one = init_one()

# Settings
OVERWRITE = True
NEURON_QC = True
PLOT = False

T_BEFORE = 1  # for plotting
T_AFTER = 4
BIN_SIZE = 0.05

PRE_TIME = [0.5, 0]  # for responsiveness testing
POST_TIME = [0, 0.5]

STIM_TIME = [0.2, 1]  # time for stimulation ROC and statistics

fig_path, save_path = paths()
fig_path = join(fig_path, 'Extra plots', 'Task neurons')
colors, dpi = figure_style()

# Query sessions
rec = query_ephys_sessions(n_trials=200, one=one)

if OVERWRITE:
    task_neurons = pd.DataFrame()
else:
    task_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
    rec = rec[~rec['eid'].isin(task_neurons['eid'])].reset_index()

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    print(f'Session {i} of {rec.shape[0]}: {subject} {date}')

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

    # Load trials dataframe
    try:
        trials = load_trials(eid, laser_stimulation=True, one=one)
    except:
        print('Could not load trials')
        continue

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one)
        spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['spikes.samples'])
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue
    
    # Filter neurons that pass QC
    if NEURON_QC:
        qc_metrics = get_neuron_qc(pid, one=one)
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    else:
        clusters_pass = np.unique(spikes.clusters)
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    if len(spikes.clusters) == 0:
        continue

    # Determine task responsive neurons
    _, _, p_values, _ = responsive_units(spikes.times, spikes.clusters, trials['goCue_times'],
                                         pre_time=PRE_TIME, post_time=POST_TIME)
    task_resp = p_values < 0.05
    roc_task, neuron_ids = roc_single_event(spikes.times, spikes.clusters, trials['goCue_times'],
                                            pre_time=PRE_TIME, post_time=POST_TIME)
    roc_task = 2 * (roc_task - 0.5)  # Recalculate modulation index

    roc_no_opto_task, neuron_ids = roc_single_event(spikes.times, spikes.clusters,
                                                    trials.loc[trials['laser_stimulation']
                                                               == 0, 'goCue_times'],
                                                    pre_time=PRE_TIME, post_time=POST_TIME)
    roc_no_opto_task = 2 * (roc_no_opto_task - 0.5)

    roc_opto_task, neuron_ids = roc_single_event(spikes.times, spikes.clusters,
                                                 trials.loc[trials['laser_stimulation']
                                                            == 1, 'goCue_times'],
                                                 pre_time=PRE_TIME, post_time=POST_TIME)
    roc_opto_task = 2 * (roc_opto_task - 0.5)

    # Get choice selective neurons
    choice_lr = trials['choice'].values
    choice_lr[choice_lr == -1] = 0
    choice_no_stim_roc = roc_between_two_events(spikes.times, spikes.clusters,
                                                trials.loc[trials['laser_stimulation']
                                                           == 0, 'goCue_times'],
                                                choice_lr[trials['laser_stimulation'] == 0],
                                                pre_time=POST_TIME[0], post_time=POST_TIME[1])[0]
    choice_no_stim_roc = 2 * (choice_no_stim_roc - 0.5)

    if len(np.unique(choice_lr[trials['laser_stimulation'] == 1])) == 2:
        choice_stim_roc = roc_between_two_events(spikes.times, spikes.clusters,
                                                 trials.loc[trials['laser_stimulation']
                                                            == 1, 'goCue_times'],
                                                 choice_lr[trials['laser_stimulation'] == 1],
                                                 pre_time=POST_TIME[0], post_time=POST_TIME[1])[0]
        choice_stim_roc = 2 * (choice_stim_roc - 0.5)
    else:
        choice_stim_roc = np.array([np.nan] * roc_opto_task.shape[0])

    choice_no_stim_p = differentiate_units(spikes.times, spikes.clusters,
                                           trials.loc[trials['laser_stimulation']
                                                      == 0, 'goCue_times'],
                                           choice_lr[trials['laser_stimulation'] == 0],
                                           pre_time=POST_TIME[0], post_time=POST_TIME[1])[2]

    choice_stim_p = differentiate_units(spikes.times, spikes.clusters,
                                        trials.loc[trials['laser_stimulation']
                                                   == 1, 'goCue_times'],
                                        choice_lr[trials['laser_stimulation'] == 1],
                                        pre_time=POST_TIME[0], post_time=POST_TIME[1])[2]

    # Determine stimulus evoked light modulated neurons
    roc_auc, neuron_ids = roc_between_two_events(spikes.times, spikes.clusters, trials['goCue_times'],
                                                 trials['laser_stimulation'], pre_time=-
                                                 STIM_TIME[0],
                                                 post_time=STIM_TIME[1])
    roc_stim_mod = 2 * (roc_auc - 0.5)

    # Get significantly opto modulated neurons
    opto_mod_p = np.empty(neuron_ids.shape[0])
    for n, neuron_id in enumerate(neuron_ids):
        if np.mod(n, 20) == 0:
            print(f'Neuron {n} of {neuron_ids.shape[0]}')

        # Perform ZETA test for neural responsiveness
        zero_contr_trials = trials[trials['signed_contrast'] == 0]
        opto_mod_p[n], dZETA = zetatest2(
            spikes.times[spikes.clusters == neuron_id],
            zero_contr_trials.loc[zero_contr_trials['laser_stimulation'] == 1, 'goCue_times'],
            spikes.times[spikes.clusters == neuron_id],
            zero_contr_trials.loc[zero_contr_trials['laser_stimulation'] == 0, 'goCue_times'],
            dblUseMaxDur=3)
           
    opto_mod = opto_mod_p < 0.05
    print(f'{np.sum(opto_mod)} out of {opto_mod.shape[0]} opto modulated neurons '
          f'({np.round((np.sum(opto_mod)/opto_mod.shape[0])*100, 1)}%)')

    # Add results to df
    cluster_regions = remap(clusters.acronym[neuron_ids])
    task_neurons = pd.concat((task_neurons, pd.DataFrame(data={
        'subject': subject, 'date': date, 'eid': eid, 'probe': probe, 'neuron_id': neuron_ids,
        'pid': pid, 'region': cluster_regions, 'task_responsive': task_resp,
        'task_roc': roc_task, 'task_no_opto_roc': roc_no_opto_task, 'task_opto_roc': roc_opto_task,
        'choice_no_stim_roc': choice_no_stim_roc, 'choice_stim_roc': choice_stim_roc,
        'choice_stim_p': choice_stim_p, 'choice_no_stim_p': choice_no_stim_p,
        'opto_modulated': opto_mod, 'opto_mod_roc': roc_stim_mod, 'opto_mod_p': opto_mod_p})))
    task_neurons = remove_artifact_neurons(task_neurons)

    if PLOT:
        for n, neuron_id in enumerate(task_neurons.loc[(task_neurons['pid'] == pid)
                                                       & task_neurons['opto_modulated'], 'neuron_id']):
            # Plot PSTH
            p, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
            peri_multiple_events_time_histogram(
                spikes.times, spikes.clusters, zero_contr_trials['goCue_times'],
                zero_contr_trials['laser_stimulation'],
                neuron_id, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=ax,
                pethline_kwargs=[{'color': colors['no-stim'], 'lw': 1},
                                 {'color': colors['stim'], 'lw': 1}],
                errbar_kwargs=[{'color': colors['no-stim'], 'alpha': 0.3, 'lw': 0},
                               {'color': colors['stim'], 'alpha': 0.3, 'lw': 0}],
                raster_kwargs=[{'color': colors['no-stim'], 'lw': 0.5},
                               {'color': colors['stim'], 'lw': 0.5}],
                eventline_kwargs={'lw': 0}, include_raster=True)
            ax.set(ylabel='Firing rate (spikes/s)', xlabel='Time from trial start (s)',
                   yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3), xticks=[-1, 0, 1, 2])
            if np.round(ax.get_ylim()[1]) % 2 == 0:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            sns.despine(trim=False)
            plt.tight_layout()
            plt.savefig(join(
                fig_path, f'{cluster_regions[neuron_ids == neuron_id][0]}_{subject}_{date}_{probe}_neuron{neuron_id}.jpg'))
            plt.close(p)

    task_neurons.to_csv(join(save_path, 'task_modulated_neurons.csv'))

# Remove artifact neurons
task_neurons = remove_artifact_neurons(task_neurons)

# Save
task_neurons.to_csv(join(save_path, 'task_modulated_neurons.csv'))
