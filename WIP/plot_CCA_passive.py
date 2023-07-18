# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:21:26 2022

@author: Guido
"""

import pandas as pd
import numpy as np
from os.path import join
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from stim_functions import paths, load_subjects, figure_style

# Settings

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Extra plots', 'CCA')

# Load in data
cca_df = pd.read_pickle(join(save_path, 'cca_results.pickle'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    cca_df.loc[cca_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Get time axis
time_ax = np.round(cca_df['time'].mean(), 3)

# Create long dataframe
cca_long_df = pd.DataFrame()
for i in cca_df.index:
    region1 = cca_df.loc[i, 'region_1']
    region2 = cca_df.loc[i, 'region_2']
    cca_long_df = pd.concat((cca_long_df, pd.DataFrame(data={
        'time': time_ax, 'r': cca_df.loc[i, 'r_mean'],
        'r_bl': cca_df.loc[i, 'r_mean'] - np.mean(cca_df.loc[i, 'r_mean'][time_ax < 0]),
        'subject': cca_df.loc[i, 'subject'],
        'date': cca_df.loc[i, 'date'], 'eid': cca_df.loc[i, 'eid'], 
        'sert-cre': cca_df.loc[i, 'sert-cre'],
        'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}'})))


# %%
colors, dpi = figure_style()

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -0.2), 1, 0.8, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=cca_long_df, x='time', y='r_bl', color='k', errorbar='se', ax=ax1,
             err_kws={'lw': 0}, hue='sert-cre', hue_order=[0, 1], palette=[colors['wt'], colors['sert']])
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.85), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['WT', 'SERT']):
    t.set_text(l)
ax1.set(ylabel='Canonical correlation (r)', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3],
        ylim=[-0.05, 0.05])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'CCA_passive.jpg'), dpi=600)

# %% Plot region pairs seperately
g = sns.FacetGrid(cca_long_df[cca_long_df['sert-cre'] == 1], col='region_pair', col_wrap=5, height=3.5,
                  ylim=(-0.1, 0.4))
g.map(sns.lineplot, 'time', 'r', color='k', errorbar='se')
plt.savefig(join(fig_path, 'CCA_passive_all_region_pair.jpg'), dpi=600)