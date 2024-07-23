# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:59:11 2024

By Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from stim_functions import paths, figure_style, combine_regions, load_subjects

# Get paths
_, save_path = paths()

# Load in data
psth_df = pd.read_pickle(join(save_path, 'psth_task.pickle'))

# Loop over neurons
for i in psth_df.index.values:
    
    # Loop over timebins
    for t in range(psth_df.loc[i, 'peth_stim'].shape[0])
        
    
    psth_df.loc[i, 'peth_stim']

