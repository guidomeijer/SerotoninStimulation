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

per_subject_df = all_trials_df.groupby(['subject', 'rel_trial', 'laser_stimulation']).mean(numeric_only=True)

# Plot
f, ax1 = plt.subplots(1, 1, figsize=(2,2), dpi=dpi)
sns.lineplot(data=per_subject_df, x='rel_trial', y='sticky', hue='laser_stimulation', errorbar='se')
ax1.set(xlim=[0, 60])