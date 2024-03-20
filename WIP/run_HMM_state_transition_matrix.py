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
data_path = join(repo_path, 'HMM', 'PassiveEvent', 'jitter', 'state_mat')
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
    asd
    
    
   