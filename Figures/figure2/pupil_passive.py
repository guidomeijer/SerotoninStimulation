# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:07:11 2023

By Guido Meijer
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from matplotlib.patches import Rectangle
from stim_functions import figure_style, paths, add_significance
from os.path import join, realpath, dirname, split

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
pupil_df = pd.read_csv(join(save_path, 'pupil_passive.csv'))

# Exclude some session with bad pupil tracking
pupil_df = pupil_df[pupil_df['eid'] != 'd0387aa6-b648-466a-b5a6-bb647c8acc41']
pupil_df = pupil_df[pupil_df['eid'] != 'c5eff7e4-71ad-4983-96ac-a7e5affaadbb']

# Do statistics
p_values = []
for timepoint, df_time in pupil_df.groupby("time"):
    y1 = df_time[df_time["expression"] == 0]["baseline_subtracted"]
    y2 = df_time[df_time["expression"] == 1]["baseline_subtracted"]
    _, p_value = stats.ttest_ind(y1[~np.isnan(y1)], y2[~np.isnan(y2)])
    p_values.append(p_value)
p_values = np.array(p_values)

# Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -4), 1, 11, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=pupil_df, x='time', y='baseline_subtracted', hue='expression',
             errorbar='se', ax=ax1, err_kws={'lw': 0}, hue_order=[1, 0],
             palette=[colors['sert'], colors['wt']])
ax1.set(ylabel=u'Δ pupil size (%)', xlabel='', xticks=[], yticks=[-3, 0, 3, 6], ylim=[-3.3, 6])
ax1.plot([0, 1], [ax1.get_ylim()[0]-0.2, ax1.get_ylim()[0]-0.2], color='k', lw=0.75, clip_on=False)
ax1.text(0.5, ax1.get_ylim()[0]-0.4, '1s', ha='center', va='top')
#ax1.text(2, -2.5, f'n = {np.unique(pupil_df["subject"]).shape[0]} mice')
handles, previous_labels = ax1.get_legend_handles_labels()
ax1.legend(title='', handles=handles, labels=['SERT', 'WT'])
add_significance(np.unique(pupil_df['time']), p_values, ax1)

sns.despine(trim=True, bottom=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'pupil_passive.pdf'))