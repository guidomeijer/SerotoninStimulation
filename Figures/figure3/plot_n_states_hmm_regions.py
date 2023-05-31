#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:27:38 2023
By: Guido Meijer
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from stim_functions import figure_style, paths, load_subjects, N_STATES
from os.path import join, realpath, dirname, split

BIN_SIZE = 100
MAX_STATES = 12

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
hmm_ll_df = pd.read_csv(join(save_path, f'hmm_log_likelihood_regions_{BIN_SIZE}ms_bins.csv'))

# Only plot until state X
hmm_ll_df = hmm_ll_df[hmm_ll_df['n_states'] <= MAX_STATES]

# Convert ll
hmm_ll_df['xcorr'] = -2 * hmm_ll_df['log_likelihood']

# Normalize ll
n_states = np.unique(hmm_ll_df['n_states'])
for i in hmm_ll_df[hmm_ll_df['n_states'] == hmm_ll_df['n_states'].min()].index:
    hmm_ll_df.loc[i:i+len(n_states)-1, 'll_norm'] = (hmm_ll_df.loc[i:i+len(n_states)-1, 'xcorr']
                                                     / hmm_ll_df.loc[i, 'xcorr'])

# Select only sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    hmm_ll_df.loc[hmm_ll_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
hmm_ll_df = hmm_ll_df[hmm_ll_df['sert-cre'] == 1]

# %% Plot 

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
#sns.lineplot(data=ll_mean_df, x='n_states', y='ll_norm', estimator=None, units='subject', ax=ax1, marker='o')
sns.lineplot(data=hmm_ll_df, x='n_states', y='ll_norm', errorbar='se', ax=ax1, marker='o',
             err_kws={'lw': 0}, zorder=1)
state_y = hmm_ll_df.groupby('n_states').mean(numeric_only=True)['ll_norm'][N_STATES]
ax1.scatter(N_STATES, state_y, color='tab:red', zorder=2, s=3)
ax1.set(ylabel='Normalized log likelihood', xlabel='Number of states',
        xlim=[1.5, MAX_STATES+0.5], xticks=np.arange(2, MAX_STATES+1))

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'hmm_n_states.pdf'))

# %%

regions = np.unique(hmm_ll_df['region'])

f, axs = plt.subplots(1, len(regions), figsize=(9, 1.75), dpi=dpi)
#sns.lineplot(data=ll_mean_df, x='n_states', y='ll_norm', estimator=None, units='subject', ax=ax1, marker='o')
for i, region in enumerate(regions):
    sns.lineplot(data=hmm_ll_df[hmm_ll_df['region'] == region], x='n_states', y='ll_norm',
                 errorbar='se', ax=axs[i], marker='o', err_kws={'lw': 0}, zorder=1)
    axs[i].plot([N_STATES, N_STATES], axs[i].get_ylim(), color='grey', ls='--', lw=0.75)
    if i == 0:
        axs[i].set(ylabel='Normalized log likelihood', xlabel='Number of states', title=f'{region}',
                   xlim=[1.5, MAX_STATES+0.5], xticks=np.arange(2, MAX_STATES+1, 2))
    else:
        axs[i].set(xlabel='Number of states', title=f'{region}', ylabel='',
                   xlim=[1.5, MAX_STATES+0.5], xticks=np.arange(2, MAX_STATES+1, 2))

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'hmm_n_states_regions.pdf'))