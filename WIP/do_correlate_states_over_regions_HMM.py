# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:11:41 2023

By Guido Meijer
"""

import numpy as np
import seaborn as sns
from glob import glob
from scipy.stats import pearsonr
from os.path import join, split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from stim_functions import paths, figure_style

# Settings
CMAP = 'Set2'
PRE_TIME = 1
POST_TIME = 4

# Plotting
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Regions')

# Get all files
all_files = glob(join(save_path, 'HMM', '*.npy'))

# Get all recordings
all_rec = np.unique([split(i)[1][:28] for i in all_files])

for i, this_rec in enumerate(all_rec):
    
    # Get all brain regions simultaneously recorded in this recording session
    rec_region_paths = glob(join(save_path, 'HMM', f'{this_rec[:20]}*'))
    rec_region = dict()
    for ii in range(len(rec_region_paths)):
        rec_region[split(rec_region_paths[ii])[1][29:-4]] = np.load(join(rec_region_paths[ii]))
    
    # Correlate each area with each other area
    for r1, region1 in enumerate(rec_region.keys()):    
        for r2, region2 in enumerate(list(rec_region.keys())[r1:]):
            if region1 == region2:
                continue
            
            # Loop over timebins
            for tb in range(rec_region[region1].shape[1]):
                
                # Correlate each state with each other state
                for state1 in range(rec_region[region1].shape[2]):
                    for state2 in range(rec_region[region2].shape[2]):
                        this_r = pearsonr(rec_region[region1][:, tb, state1],
                                          rec_region[region2][:, tb, state2])[0]
                        asd
        