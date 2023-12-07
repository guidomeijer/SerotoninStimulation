# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:26:54 2023

@author: Guido
"""


import numpy as np
from os.path import join, realpath, dirname, split
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from brainbox.io.one import SpikeSortingLoader
from scipy.ndimage import gaussian_filter
from brainbox.singlecell import calculate_peths
from stim_functions import (paths, query_ephys_sessions, load_passive_opto_times, load_subjects,
                            figure_style, remap, high_level_regions, combine_regions)

# Settings
PRE_TIME = 1
POST_TIME = 2
MAP = 'high_level'  # combined, high_level or beryl

# Get paths
parent_fig_path, repo_path = paths()
fig_path = join(parent_fig_path, 'Extra plots', 'State', 'All neurons')
data_path = join(repo_path, 'HMM', 'PassiveEventAllNeurons')
rec_files = glob(join(data_path, '*.pickle'))

subjects = load_subjects()
colors, dpi = figure_style()
all_regions, all_n_neurons, all_loadings = [], [], []
for i, file_path in enumerate(rec_files):
    
    # Get info
    subject = split(file_path)[1][:9]
    if subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0] == 0:
        continue
    
    # Load in data
    with open(file_path, 'rb') as handle:
        hmm_dict = pickle.load(handle)    
    n_states = hmm_dict['prob_mat'].shape[2]
    
    # Create region map
    if MAP == 'combined':
        neuron_regions = combine_regions(remap(hmm_dict['regions'], abbreviate=False))
    elif MAP == 'high_level':         
        neuron_regions = high_level_regions(hmm_dict['regions'])
    elif MAP == 'beryl':
        neuron_regions = remap(hmm_dict['regions'])        
    unique_regions = np.unique(neuron_regions)
    unique_regions = unique_regions[unique_regions != 'root']
    
    # Calculate mean loading of regions to states
    log_lambdas = np.empty((unique_regions.shape[0], n_states))
    for s in range(n_states):
        for r, region in enumerate(unique_regions):
            log_lambdas[r, s] = np.mean(hmm_dict['log_lambdas'][s, neuron_regions == region])
    neuron_loadings = np.exp(log_lambdas)
    
    # Get number of neurons per region
    n_neurons = [np.sum(i == neuron_regions) for i in unique_regions]
    all_n_neurons.append(n_neurons)
    all_regions.append(unique_regions)
    all_loadings.append(np.mean(np.rot90(neuron_loadings), axis=0))
    
    # Plot    
    cmap = sns.color_palette(colors['states_light'], n_states)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5, 3), dpi=dpi)
    
    ax1.add_patch(Rectangle((0, 1), 1, hmm_dict['prob_mat'].shape[0], color='royalblue', alpha=0.25, lw=0))
    ax1.imshow(np.flipud(hmm_dict['state_mat']), aspect='auto', cmap=ListedColormap(cmap),
               vmin=0, vmax=n_states-1,
               extent=(-PRE_TIME, POST_TIME, 1, hmm_dict['prob_mat'].shape[0]+1), interpolation=None)
    #ax1.plot([-1, 4], [trial+1, trial+1], color='k', lw=0.5)
    #x1.plot([-1, 4], [trial+2.1, trial+2.1], color='k', lw=0.5)
    ax1.set(xticks=[], yticks=np.array([1, hmm_dict['prob_mat'].shape[0]]) + 0.5,
            yticklabels=np.array([1, hmm_dict['prob_mat'].shape[0]]))
    ax1.set_ylabel('Trials', labelpad=-10)
    ax1.plot([0, 2], [0.5, 0.5], lw=0.75, color='k', clip_on=False)
    ax1.text(1, -1.5, '2s', ha='center', va='center')
    
    ax2.add_patch(Rectangle((0, -0.2), 1, 2.4, color='royalblue', alpha=0.25, lw=0))
    for this_state in np.arange(n_states):
        mean_state = (np.mean(hmm_dict['prob_mat'][:,:,this_state], axis=0)
                      - np.mean(hmm_dict['prob_mat'][:,hmm_dict['time_ax'] < 0,this_state])) + (this_state/3)
        
        sem_state = np.std(hmm_dict['prob_mat'][:,:,this_state], axis=0) / np.sqrt(hmm_dict['prob_mat'].shape[0])
        ax2.plot(hmm_dict['time_ax'], mean_state, color=cmap[this_state])
        ax2.fill_between(hmm_dict['time_ax'], mean_state + sem_state, mean_state - sem_state, alpha=0.25,
                         color=cmap[this_state], lw=0)
        ax2.text(POST_TIME+0.1, this_state/3, f'{this_state+1}', color=cmap[this_state])
    ax2.plot([-1.1, -1.1], [0, 0.25], color='k')
    ax2.plot([0, 2], [-0.2, -0.2], color='k')
    ax2.text(-1.4, 0.175, '25%', rotation=90, ha='center', va='center')
    ax2.text(1, -0.3, '2s', ha='center', va='center')
    ax2.set(xticks=[], yticks=[])
    ax2.set_ylabel('P(state)', labelpad=0)
    
    ax3.imshow(np.rot90(neuron_loadings))
    ax3.set(xticks=np.arange(unique_regions.shape[0]), yticks=np.arange(n_states),
            yticklabels=np.arange(n_states, 0, -1),
            ylabel='State')
    ax3.set_xticklabels(unique_regions, rotation=45, ha='right')
        
    sns.despine(trim=True, bottom=True, left=True)
    plt.tight_layout()
    
    plt.savefig(join(fig_path, split(file_path)[1][:20] + '.jpg'), dpi=600)
    plt.close(f)
n_neuron_df = pd.DataFrame(data={
    'n_neurons': np.concatenate(all_n_neurons), 'loading': np.concatenate(all_loadings),
    'region': np.concatenate(all_regions)})
    
    
# %%
f, ax1 = plt.subplots(figsize=(2, 2), dpi=dpi)
sns.scatterplot(data=n_neuron_df, x='n_neurons', y='loading', hue='region')
g = ax1.legend(title='', bbox_to_anchor=(1, 1), prop={'size': 5})
ax1.set(xlabel='Number of neurons', ylabel='Mean loading to brain state')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'n_neurons_loadings.jpg'), dpi=600)
    