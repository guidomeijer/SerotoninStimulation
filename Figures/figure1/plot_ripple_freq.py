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
from matplotlib.patches import Rectangle
from stim_functions import figure_style, paths, load_subjects

# Set paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
ripple_df = pd.read_csv(join(save_path, 'ripple_freq.csv'))

# Exclude recordings because of light artifacts
excl_pids = ['87ddbc18-7a81-49f7-a9fb-d0add506ee27',
             '6b68ff0d-9014-4a81-b2c2-915ddfdba3a3',
             '05db6d4d-cfda-4686-b485-481423912b33']
ripple_df = ripple_df[~ripple_df['pid'].isin(excl_pids)]

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    ripple_df.loc[ripple_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
ripple_df = ripple_df[ripple_df['sert-cre'] == 1]

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(1.4, 1.6), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 0.35, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=ripple_df, x='time', y='ripple_freq', errorbar='se', color='k',
             err_kws={'lw': 0})
ax1.text(1.5, 0.32, f'n = {np.unique(ripple_df["subject"]).shape[0]} mice', ha='left', va='center')
ax1.set(xticks=[], ylim=[0, 0.35], yticks=[0, 0.35], yticklabels=[0, 0.35], xlabel='')
ax1.set_ylabel('Hippocampal sharp\nwave ripple rate', labelpad=-10)
ax1.plot([0, 1], [ax1.get_ylim()[0]-0.008, ax1.get_ylim()[0]-0.008], color='k', lw=0.75, clip_on=False)
ax1.text(0.5, ax1.get_ylim()[0]-0.02, '1s', ha='center', va='top')

sns.despine(trim=True, bottom=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'ripples_stim.pdf'))


