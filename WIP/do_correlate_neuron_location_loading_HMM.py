# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:12:54 2023 by Guido Meijer
"""


import numpy as np
from os.path import join, split
import pickle
from glob import glob
from brainbox.io.one import SpikeSortingLoader
from stim_functions import paths, load_subjects
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Get paths
parent_fig_path, repo_path = paths()
fig_path = join(parent_fig_path, 'Extra plots', 'State', 'All neurons')
data_path = join(repo_path, 'HMM', 'PassiveEventAllNeurons')
rec_files = glob(join(data_path, '*.pickle'))

subjects = load_subjects()
all_regions, all_n_neurons, all_loadings = [], [], []
for i, file_path in enumerate(rec_files):
    
    # Get info
    subject = split(file_path)[1][:9]
    date = split(file_path)[1][10:20]
    print(f'Processing session {i} of {len(rec_files)}')
    
    # Load in data
    with open(file_path, 'rb') as handle:
        hmm_dict = pickle.load(handle)    
    n_states = hmm_dict['prob_mat'].shape[2]
    
    # Load in depth along the probe of HMM neurons
    hmm_dict['depth'] = np.zeros(hmm_dict['neuron_id'].shape).astype(int)
    for p, probe in enumerate(np.unique(hmm_dict['probe'])):
        eid = one.search(subject=subject, date=date)[0]
        pid = one.eid2pid(eid)[0][p]
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
        hmm_dict['depth'][hmm_dict['probe'] == probe] = clusters.axial_um[
            hmm_dict['neuron_id'][hmm_dict['probe'] == probe]]
    
    # Save result
    with open(file_path, 'wb') as fp:
        pickle.dump(hmm_dict, fp)
        
    
    
    