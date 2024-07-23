# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:46:56 2024

By Guido Meijer
"""


import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stim_functions import (paths, remap, query_ephys_sessions, load_trials, figure_style,
                            load_subjects)
colors, dpi = figure_style()
_, data_path = paths()

# Load in data
all_trials_df = pd.read_csv(join(data_path, 'behavioral_strategy.csv'))

all_trials_df = all_trials_df[all_trials_df['subject'] != 'ZFM-04300']

per_subject_df = all_trials_df.groupby(['subject', 'rel_trial', 'laser_stimulation']).median(numeric_only=True)

# Plot
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7, 2), dpi=dpi)

sns.lineplot(data=per_subject_df, x='rel_trial', y='go_cued', hue='laser_stimulation',
             errorbar='se', ax=ax1)
ax1.set(xlim=[0, 50], title='Go cued')

sns.lineplot(data=per_subject_df, x='rel_trial', y='repeat_choice', hue='laser_stimulation',
             errorbar='se', ax=ax2)
ax2.set(xlim=[0, 50], title='Repeat choice')

sns.lineplot(data=per_subject_df, x='rel_trial', y='win_stay_lose_shift', hue='laser_stimulation',
             errorbar='se', ax=ax3)
ax3.set(xlim=[0, 50], title='Win stay lose shift')

sns.lineplot(data=per_subject_df, x='rel_trial', y='integration_window', hue='laser_stimulation',
             errorbar='se', ax=ax4)
ax4.set(xlim=[0, 50], title='Integration window')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2), dpi=dpi)

sns.lineplot(data=per_subject_df, x='rel_trial', y='lose_shift_cued', hue='laser_stimulation',
             errorbar='se', ax=ax1)
ax1.set(xlim=[0, 50], title='Lose shift')

sns.lineplot(data=per_subject_df, x='rel_trial', y='win_stay_cued', hue='laser_stimulation',
             errorbar='se', ax=ax2)
ax2.set(xlim=[0, 50], title='Win stay')