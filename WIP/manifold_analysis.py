# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:30:26 2023

By Guido Meijer
"""

import numpy as np
from brainwidemap.manifold import state_space_bwm
from stim_functions import query_ephys_sessions 
from one.api import ONE
one = ONE()

rec = query_ephys_sessions(n_trials=400, one=one)

for split in ['choice', 'stim', 'fback', 'block']:
    # computes PETHs, distance sums
    state_space_bwm.get_all_d_vars(split, eids_plus=rec[['eid', 'probe', 'pid']].values)
    # combine results across insertions
    state_space_bwm.d_var_stacked(split)

# this replicates the main figure of the paper
state_space_bwm.plot_all()