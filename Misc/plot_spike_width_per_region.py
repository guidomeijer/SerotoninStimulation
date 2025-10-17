# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:42:44 2025

By Guido Meijer
"""

from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import paths, remap, combine_regions, figure_style
colors, dpi = figure_style()
fig_path, save_path = paths()

waveform_df = pd.read_pickle(path.join(save_path, 'waveform_metrics.p'))
waveform_df['region'] = combine_regions(remap(waveform_df['acronym']))
waveform_df = waveform_df[waveform_df['region'] != 'root']
waveform_df = waveform_df[waveform_df['region'] != 'AI']
waveform_df = waveform_df[waveform_df['region'] != 'ZI']
waveform_df = waveform_df[waveform_df['region'] != 'BC']
all_regions = np.unique(waveform_df['region'])

# %%
f, axs = plt.subplots(3, 5, figsize=(9, 5), dpi=dpi, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(all_regions):
    
    sns.histplot(data=waveform_df[waveform_df['region'] == region], x='spike_width', ax=axs[i],
                 binwidth=0.032)
    axs[i].set(title=region, xlim=[0, 1.5], ylabel='', xlabel='', xticks=[0, 0.5, 1, 1.5])
    
sns.despine(trim=True)
plt.tight_layout()