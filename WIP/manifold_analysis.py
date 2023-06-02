# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:30:26 2023

By Guido Meijer
"""

from manifold import state_space
from stim_functions import query_ephys_sessions 
from one.api import ONE
one = ONE()

rec = query_ephys_sessions(n_trials=200, one=one)
SPLIT = 'opto'  # choice, stim, fback, block or opto

# computes PETHs, distance sums
state_space.get_all_d_vars(SPLIT, eids_plus=rec[['eid', 'probe', 'pid']].values)
# combine results across insertions
state_space.d_var_stacked(SPLIT)

# this replicates the main figure of the paper
#state_space.plot_all()