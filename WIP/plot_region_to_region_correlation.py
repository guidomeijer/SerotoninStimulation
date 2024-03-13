# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:22:15 2024 by Guido Meijer
"""


import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from stim_functions import paths, load_subjects, figure_style

# Settings
TIME_WIN = 1.9

# Load in data
fig_path, save_path = paths()
corr_df = pd.read_csv(join(save_path, 'region_corr_200.csv'))
corr_df = corr_df[(corr_df['sert-cre'] == 1) & (corr_df['time'] == TIME_WIN)]

# Create region to region matrix
all_regions = np.unique(corr_df['region_1'])
for i, region_1 in enumerate(all_regions):
    for j, region_2 in enumerate(all_regions):
        region_pair_df = corr_df[]
        