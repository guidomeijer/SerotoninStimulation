# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:26:46 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join, realpath, dirname, split
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Rectangle
from stim_functions import figure_style, paths, load_subjects, add_significance

# Set paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
ripple_df = pd.read_csv(join(save_path, 'ripple_freq.csv'))
ripple_df['pid'] = ripple_df['pid'].astype(str)

# Exclude recordings because of light artifacts
excl_pids = ['87ddbc18-7a81-49f7-a9fb-d0add506ee27',
             '6b68ff0d-9014-4a81-b2c2-915ddfdba3a3',
             '05db6d4d-cfda-4686-b485-481423912b33',
             'f06f5a5e-2e8a-4cf0-ac5f-8d14aeec4c98']
ripple_df = ripple_df[~ripple_df['pid'].isin(excl_pids)]

# Do baseline subraction
for i, pid in enumerate(np.unique(ripple_df['pid'])):
    baseline = np.mean(ripple_df.loc[(ripple_df['pid'] == pid) & (ripple_df['time'] < 0), 'ripple_freq'])
    ripple_df.loc[ripple_df['pid'] == pid, 'ripple_bl'] = ripple_df.loc[ripple_df['pid'] == pid, 'ripple_freq'] - baseline

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    ripple_df.loc[ripple_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Do statistics
p_values = []
for timepoint, df_time in ripple_df.groupby("time"):
    y1 = df_time[df_time["sert-cre"] == 0]["ripple_bl"]
    y2 = df_time[df_time["sert-cre"] == 1]["ripple_bl"]
    _, p_value = stats.ttest_ind(y1[~np.isnan(y1)], y2[~np.isnan(y2)])
    p_values.append(p_value)
p_values = np.array(p_values)

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

ax1.add_patch(Rectangle((0, -0.3), 1, 0.4, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=ripple_df, x='time', y='ripple_bl', errorbar='se', hue='sert-cre',
             hue_order=[1, 0], palette=[colors['sert'], colors['wt']],
             err_kws={'lw': 0}, ax=ax1)
#ax1.text(2, 0.32, f'n = {np.unique(ripple_df["subject"]).shape[0]} mice', ha='left', va='center')
ax1.set(ylabel=u'Δ ripple frequency (rip./s)',
        xlabel='', xticks=[], ylim=[-0.3, 0.1], yticks=[-0.3, -0.2, -0.1, 0, 0.1],
        yticklabels=[-0.3, -0.2, -0.1, 0, 0.1])
ax1.plot([0, 1], [ax1.get_ylim()[0]-0.01, ax1.get_ylim()[0]-0.01], color='k', lw=0.75, clip_on=False)
ax1.text(0.5, ax1.get_ylim()[0]-0.02, '1s', ha='center', va='top')
handles, previous_labels = ax1.get_legend_handles_labels()
ax1.legend(title='', handles=handles, labels=['SERT', 'WT'])
add_significance(np.unique(ripple_df['time']), p_values, ax1)

sns.despine(trim=True, bottom=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'ripples_stim.pdf'))


