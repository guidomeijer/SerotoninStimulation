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
from os.path import join

BIN_SIZE = 100
MAX_STATES = 12

# Get paths
fig_path, save_path = paths()

# Load in data
hmm_ll_df = pd.read_csv(join(save_path, f'hmm_log_likelihood_all-neurons_{BIN_SIZE}ms_bins.csv'))

# Only plot until state X
hmm_ll_df = hmm_ll_df[hmm_ll_df['n_states'] <= MAX_STATES]

# Convert ll
hmm_ll_df['xcorr'] = -2 * hmm_ll_df['log_likelihood']

# Calculate AIC
hmm_ll_df['AIC'] = (2 * hmm_ll_df['n_states']) - (2 * hmm_ll_df['log_likelihood'])

# Normalize ll
n_states = np.unique(hmm_ll_df['n_states'])
for i in hmm_ll_df[hmm_ll_df['n_states'] == hmm_ll_df['n_states'].min()].index:
    hmm_ll_df.loc[i:i+len(n_states)-1, 'll_norm'] = (hmm_ll_df.loc[i:i+len(n_states)-1, 'xcorr']
                                                     / hmm_ll_df.loc[i, 'xcorr'])
    hmm_ll_df.loc[i:i+len(n_states)-1, 'AIC_norm'] = (hmm_ll_df.loc[i:i+len(n_states)-1, 'AIC']
                                                     / hmm_ll_df.loc[i, 'AIC'])

# Select only sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    hmm_ll_df.loc[hmm_ll_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
hmm_ll_df = hmm_ll_df[hmm_ll_df['sert-cre'] == 1]

# Plot 
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

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
#sns.lineplot(data=ll_mean_df, x='n_states', y='ll_norm', estimator=None, units='subject', ax=ax1, marker='o')
sns.lineplot(data=hmm_ll_df, x='n_states', y='AIC', errorbar='se', ax=ax1, marker='o',
             err_kws={'lw': 0}, zorder=1)
#state_y = hmm_ll_df.groupby('n_states').mean(numeric_only=True)['ll_norm'][N_STATES]
#ax1.scatter(N_STATES, state_y, color='tab:red', zorder=2, s=3)
#ax1.set(ylabel='Normalized log likelihood', xlabel='Number of states',
#        xlim=[1.5, MAX_STATES+0.5], xticks=np.arange(2, MAX_STATES+1))

plt.tight_layout()
sns.despine(trim=True)