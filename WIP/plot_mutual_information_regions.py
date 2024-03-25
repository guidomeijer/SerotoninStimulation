# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:45:13 2024

By Guido Meijer
"""


import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stim_functions import paths, figure_style
colors, dpi = figure_style()
    
# Load in data
fig_path, save_path = paths()
mi_df = pd.read_csv(join(save_path, 'region_mutual_information_pairwise.csv'))
mi_df = mi_df[mi_df['sert-cre'] == 1]

f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(data=mi_df,
             x='time', y='mi_over_baseline', errorbar='se', ax=ax1,
             err_kws={'lw': 0})
ax1.set(xlim=[-0.5, 1])


g = sns.FacetGrid(mi_df, col='region_pair', col_wrap=16)
g.map(sns.lineplot, 'time', 'mi_over_baseline', errorbar='se')