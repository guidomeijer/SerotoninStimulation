# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:07:11 2023

By Guido Meijer
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from matplotlib.patches import Rectangle
from stim_functions import figure_style, paths, add_significance
from os.path import join, realpath, dirname, split

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
whisking_df = pd.read_csv(join(save_path, 'whisking_passive.csv'))

# Do statistics
p_values = []
for timepoint, df_time in whisking_df.groupby("time"):
    y1 = df_time[df_time["expression"] == 0]["baseline_subtracted"]
    y2 = df_time[df_time["expression"] == 1]["baseline_subtracted"]
    _, p_value = stats.ttest_ind(y1[~np.isnan(y1)], y2[~np.isnan(y2)])
    p_values.append(p_value)
p_values = np.array(p_values)

# Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -50), 1, 200, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=whisking_df, x='time', y='baseline_subtracted', hue='expression',
             hue_order=[1, 0], palette=[colors['sert'], colors['wt']],
             errorbar='se', ax=ax1, err_kws={'lw': 0})
ax1.set(ylabel=u'Δ whisking (%)', xlabel='', xticks=[], yticks=[-50, 0, 50, 100, 150],
        ylim=[-50, 150])
handles, previous_labels = ax1.get_legend_handles_labels()
ax1.legend(title='', handles=handles, labels=['SERT', 'WT'])
add_significance(np.unique(whisking_df['time']), p_values, ax1)

ax1.plot([0, 1], [ax1.get_ylim()[0]-5, ax1.get_ylim()[0]-5], color='k', lw=0.75, clip_on=False)
ax1.text(0.5, ax1.get_ylim()[0]-10, '1s', ha='center', va='top')
#ax1.text(1.3, -10, f'n = {np.unique(whisking_df["subject"]).shape[0]} mice')

sns.despine(trim=True, bottom=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'whisking_passive.pdf'))