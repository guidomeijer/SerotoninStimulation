# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:26:54 2023

@author: Guido
"""


import numpy as np
from os.path import join, split
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from stim_functions import paths, load_subjects, figure_style

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
    cmap = sns.color_palette(colors['states_light'], n_states)
    
    # Plot per probe
    for p, probe in enumerate(np.unique(hmm_dict['probe'])):
        
        # Get neurons on this probe
        neuron_depth = hmm_dict['neuron_depth'][hmm_dict['probe'] == probe]
        neuron_loadings = np.exp(hmm_dict['log_lambdas'][:, hmm_dict['probe'] == probe])
        
        f, axs = plt.subplots(n_states, 1, figsize=(3, 5), dpi=dpi, sharex=True)
        for i, ax in enumerate(axs):
            ax.scatter(neuron_depth, neuron_loadings[i, :], color=cmap[i])
            ax.set(xticks=[0, 1000, 2000, 3000, 4000], yticks=[0, 5])
            if i == n_states-1:
                ax.set(xlabel='Depth (um)')
        f.supylabel('Neuron loading to state', fontsize=7)
        sns.despine(trim=True)
        plt.tight_layout()
            
        plt.savefig(join(fig_path, split(file_path)[1][:20] + f'_depth_{probe}.jpg'), dpi=600)
        plt.close(f)
        
    