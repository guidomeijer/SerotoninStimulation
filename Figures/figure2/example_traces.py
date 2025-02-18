# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:17:02 2025

By Guido Meijer
"""


import numpy as np
from os.path import join, realpath, dirname, split
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from stim_functions import (paths, figure_style, load_passive_opto_times, load_subjects,
                            load_trials, init_one, get_dlc_XYs, smooth_interpolate_signal_sg,
                            get_raw_smooth_pupil_diameter)
one = init_one()
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Settings
#EID = '595c8ddd-a8e3-438d-9c31-62b2a07cc6cf' # ZFM-03330
#EID = '37ba7dd8-714f-4c63-a7e3-548e35d8664a'  # ZFM-03329
#EID = '54014195-8d46-4a06-a291-4761d1c98f10'  # ZFM-04811
EID = '08371a0c-22d9-4f6e-ac22-c15788b39180'  # ZFM-04080

# Load in data
opto_train_times, _ = load_passive_opto_times(EID, one=one)
video_times, XYs = get_dlc_XYs(one, EID)

# Get whisking
roi_motion = one.load_dataset(EID, dataset='leftCamera.ROIMotionEnergy.npy')
whisking = smooth_interpolate_signal_sg(roi_motion)

# Get pupil diameter
raw_diameter, diameter = get_raw_smooth_pupil_diameter(XYs)

# Get sniffing
distances = np.sqrt(np.sum(np.diff(XYs['nose_tip'], axis=0)**2, axis=1))
sniffing = smooth_interpolate_signal_sg(distances)
sniffing = np.concatenate((sniffing, [0]))

# Z-score
pupil_zscore = np.full(diameter.shape[0], np.nan)
pupil_zscore[~np.isnan(diameter)] = zscore(diameter[~np.isnan(diameter)])
whisking_zscore = zscore(whisking)
sniffing_zscore = zscore(sniffing)

# %% Plot

colors, dpi = figure_style()
t_start = 4038

f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
for i, opto_start in enumerate(opto_train_times):
    ax1.add_patch(Rectangle((opto_start, -4), 1, 21, color='royalblue', alpha=0.25, lw=0))
    
ax1.plot(video_times, pupil_zscore, color=sns.color_palette('Dark2')[0])
ax1.plot(video_times, whisking_zscore + 5, color=sns.color_palette('Dark2')[1])
ax1.plot(video_times, sniffing_zscore + 10, color=sns.color_palette('Dark2')[2])
ax1.plot([t_start, t_start+5], [-4, -4], color='k')
ax1.text(t_start+2.5, -5.5, '5s', ha='center', va='center')
ax1.plot([t_start-0.5, t_start-0.5], [-2, 2], color='k', clip_on=False)
ax1.text(t_start-1.5, 0, '4 SD', ha='center', va='center', rotation=90)

ax1.set(xlim=[t_start, t_start+30])
ax1.axis('off')

plt.tight_layout()
plt.savefig(join(fig_path, 'example_traces.pdf'))
