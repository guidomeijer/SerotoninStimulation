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
pupil_df = pd.read_csv(join(save_path, 'sniffing_passive.csv'))
pupil_df = pupil_df[pupil_df['expression'] == 1]

# Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -50), 1, 350, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=pupil_df, x='time', y='baseline_subtracted', 
             errorbar='se', ax=ax1, err_kws={'lw': 0}, color='k')
ax1.set(ylabel=u'Δ sniffing (%)', xlabel='', xticks=[], yticks=[-50, 0, 50, 100, 150, 200, 250, 300],
        ylim=[-50, 300])
ax1.plot([0, 1], [ax1.get_ylim()[0]-10, ax1.get_ylim()[0]-10], color='k', lw=0.75, clip_on=False)
ax1.text(0.5, ax1.get_ylim()[0]-20, '1s', ha='center', va='top')
ax1.text(1.3, -20, f'n = {np.unique(pupil_df["subject"]).shape[0]} mice')

sns.despine(trim=True, bottom=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'sniffing_passive.pdf'))