# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:12:22 2024 by Guido Meijer
"""


import numpy as np
from os.path import join, realpath, dirname, split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from stim_functions import figure_style, paths
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
decoding_df = pd.read_csv(join(save_path, 'decoding_all_trials_prev_choice_firstMovement_times.csv'))
#decoding_df = pd.read_csv(join(save_path, 'decoding_prev_choice_feedback_times.csv'))
#decoding_df = pd.read_csv(join(save_path, 'decoding_prev_choice_stimOn_times.csv'))
decoding_df = decoding_df[decoding_df['sert-cre'] == 1]

# Get comparisons
decoding_df['rel_this_trial'] = decoding_df['stim_this_trial'] - decoding_df['no_stim_this_trial']
decoding_df['rel_prev_trial'] = decoding_df['stim_prev_trial'] - decoding_df['no_stim_prev_trial']
decoding_df['this_vs_prev_stim'] = decoding_df['stim_this_trial'] - decoding_df['stim_prev_trial']
decoding_df['this_vs_prev_no_stim'] = decoding_df['no_stim_this_trial'] - decoding_df['no_stim_prev_trial']
decoding_df['trial_vs_stim'] = decoding_df['this_vs_prev_stim'] - decoding_df['this_vs_prev_no_stim']

# Plot
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.barplot(data=decoding_df, x='rel_prev_trial', y='region',
            errorbar='se', color='grey', ax=ax1)
plt.tight_layout()