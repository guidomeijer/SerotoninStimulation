# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:22:23 2023

By Guido Meijer
"""

import numpy as np
from os.path import join, realpath, dirname, split
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from stim_functions import figure_style, paths

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
peths_df = pd.read_pickle(join(save_path, 'psth_task.pickle'))

for r, region in enumerate(np.unique(peths_df['high_level_region'])):
    
    # Get array of all PETHs and select time limits
    time_ax = peths_df['time'][0]
    L_opto = np.array(peths_df.loc[(peths_df['high_level_region'] == region)
                                   & (peths_df['split'] == 'L_opto'), 'peth'].tolist())
    R_opto = np.array(peths_df.loc[(peths_df['high_level_region'] == region)
                                   & (peths_df['split'] == 'R_opto'), 'peth'].tolist())
    L_no_opto = np.array(peths_df.loc[(peths_df['high_level_region'] == region)
                                      & (peths_df['split'] == 'L_no_opto'), 'peth'].tolist())
    R_no_opto = np.array(peths_df.loc[(peths_df['high_level_region'] == region)
                                      & (peths_df['split'] == 'R_no_opto'), 'peth'].tolist())
    
    
    
    asd