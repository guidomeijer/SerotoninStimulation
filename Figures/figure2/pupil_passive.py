# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:07:11 2023

By Guido Meijer
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from stim_functions import figure_style, paths
from os.path import join, realpath, dirname, split

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
pupil_df = pd.read_csv(join(save_path, 'pupil_passive.csv'))
pupil_df = pupil_df[pupil_df['expression'] == 1]

# Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -4), 1, 11, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=pupil_df, x='time', y='baseline_subtracted', 
             errorbar='se', ax=ax1, err_kws={'lw': 0}, color='k')
ax1.set(ylabel=u'Δ pupil size (%)', xlabel='', xticks=[], yticks=[-3, 0, 3, 6], ylim=[-3.3, 6])
ax1.plot([0, 1], [ax1.get_ylim()[0]-0.2, ax1.get_ylim()[0]-0.2], color='k', lw=0.75, clip_on=False)
ax1.text(0.5, ax1.get_ylim()[0]-0.4, '1s', ha='center', va='top')
ax1.text(2, -2.5, f'n = {np.unique(pupil_df["subject"]).shape[0]} mice')

sns.despine(trim=True, bottom=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'pupil_passive.pdf'))