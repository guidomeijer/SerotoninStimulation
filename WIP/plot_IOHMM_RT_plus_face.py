# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:13:53 2025

By Guido Meijer
"""

import numpy as np
from os import path
from glob import glob
import pandas as pd
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
from stim_functions import paths, figure_style

# Get processed sessions
fig_path, save_path = paths()
ses = glob(path.join(save_path, 'IOHMM', '*.csv'))

# Load in all sessions into one big dataframe
all_trials = pd.DataFrame()
for i, ses_path in enumerate(ses):
    these_trials = pd.read_csv(ses_path)
    these_trials['pupil_zscore'] = zscore(these_trials['pupil'])
    these_trials['whisking_zscore'] = zscore(these_trials['whisking'])
    these_trials['sniffing_zscore'] = zscore(these_trials['sniffing'])
    these_trials['rt_zscore'] = zscore(np.log10(these_trials['reaction_times']))
    these_trials['subject'] = path.split(ses_path)[1][:9]
    these_trials['date'] = path.split(ses_path)[1][10:20]
    all_trials = pd.concat((all_trials, these_trials))
    
# Calculate means per subject for opto and no opto
per_sub_opto = all_trials.groupby(['subject', 'laser_stimulation']).mean(numeric_only=True)    
per_sub_diff = (per_sub_opto.xs(key=1, level='laser_stimulation')
                - per_sub_opto.xs(key=0, level='laser_stimulation')) * 100

# Calculate means per subject for the three states
per_sub_state = all_trials.groupby(['subject', 'state']).mean(numeric_only=True)    
per_sub_diff = (per_sub_opto.xs(key=1, level='laser_stimulation')
                - per_sub_opto.xs(key=0, level='laser_stimulation')) * 100

# %% Plot
colors, dpi = figure_style()

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
per_state = all_trials.groupby('state').mean(numeric_only=True)
ax1.imshow(per_state[['rt_zscore', 'pupil_zscore', 'whisking_zscore', 'sniffing_zscore']].to_numpy().T,
           cmap='coolwarm', vmin=-0.5, vmax=0.5)
ax1.set(yticks=[0, 1, 2, 3], yticklabels=['RT', 'Pupil', 'Whisking', 'Sniffing'], 
        xticks=[0, 1, 2], xticklabels=[1, 2, 3], xlabel='State')

# %%

long_df = per_sub_diff.melt(value_vars=['p_engaged', 'p_exploratory', 'p_disengaged'])
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.swarmplot(data=long_df, x='variable', y='value')