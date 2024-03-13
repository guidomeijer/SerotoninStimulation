# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:11:20 2024 by Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from stim_functions import paths, load_subjects, figure_style
from dlc_functions import smooth_interpolate_signal_sg

# Settings
REGIONS_1 = ['Frontal cortex', 'Sensory cortex']
REGIONS_2 = ['Midbrain', 'Thalamus', 'Amygdala']

# Paths
fig_path, save_path = paths()

# Load in data
pca_df = pd.read_csv(join(save_path, 'pca_regions.csv'))
pca_dist_df = pd.read_csv(join(save_path, 'pca_dist_regions.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    pca_df.loc[pca_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    pca_dist_df.loc[pca_dist_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    
# Smooth traces (this is ugly I know)
for i, pid in enumerate(pca_dist_df['pid'].unique()):
    for r, region in enumerate(REGIONS_1 + REGIONS_2):
        if pca_dist_df.loc[(pca_dist_df['pid'] == pid) & (pca_dist_df['region'] == region)].shape[0] > 0:
            pca_dist_df.loc[(pca_dist_df['pid'] == pid) & (pca_dist_df['region'] == region), 'pca_dist_smooth'] = (
                smooth_interpolate_signal_sg(pca_dist_df.loc[(pca_dist_df['pid'] == pid)
                                                             & (pca_dist_df['region'] == region), 'pca_dist'], window=11))
    
    
# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 3, color='royalblue', alpha=0.25, lw=0))
#ax.add_patch(Rectangle((0, 0), 1, -100, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='pca_dist', ax=ax1, legend='brief', hue='region', errorbar='se',
             data=pca_dist_df[(pca_dist_df['sert-cre'] == 1) & (pca_dist_df['region'].isin(REGIONS_1))],
             hue_order=REGIONS_1, palette=colors)
ax1.set(xlim=[-0.5, 1.5], xlabel='Time (s)', ylabel='PCA traj. displacement (a.u.)', xticks=[-0.5, 0, 0.5, 1, 1.5],
        ylim=[0, 2])
ax1.legend(title='', frameon=False, bbox_to_anchor=(0.55, 0.6))

ax2.add_patch(Rectangle((0, 0), 1, 3, color='royalblue', alpha=0.25, lw=0))
#ax.add_patch(Rectangle((0, 0), 1, -100, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='pca_dist', ax=ax2, legend='brief', hue='region', errorbar='se',
             data=pca_dist_df[(pca_dist_df['sert-cre'] == 1) & (pca_dist_df['region'].isin(REGIONS_2))],
             hue_order=REGIONS_2, palette=colors)
ax2.set(xlim=[-0.5, 1.5], xlabel='Time (s)', ylabel='PCA traj. displacement (a.u.)', xticks=[-0.5, 0, 0.5, 1, 1.5],
        ylim=[0, 2])
ax2.legend(title='', frameon=False, bbox_to_anchor=(0.55, 0.6))
plt.tight_layout()
sns.despine(trim=True)
