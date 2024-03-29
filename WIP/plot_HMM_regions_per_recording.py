# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:11:41 2023

By Guido Meijer
"""

import numpy as np
import seaborn as sns
from glob import glob
from os.path import join, split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from stim_functions import paths, figure_style, init_one, load_trials

# Settings
CMAP = 'Set2'
PRE_TIME = 1
POST_TIME = 4
BEHAVIOR = 'Passive'  # Passive or Task
CONTROL = 'jitter'

# We only need ONE for loading in the trials
if BEHAVIOR == 'Task':
    one = init_one()

# Plotting
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Simultaneous Regions', f'{BEHAVIOR}')

# Get all files
all_files = glob(join(save_path, 'HMM', f'{BEHAVIOR}', f'{CONTROL}', 'state_mat', '*.npy'))

# Get all recordings
all_rec = np.unique([split(i)[1][:28] for i in all_files])

for i, this_rec in enumerate(all_rec):

    # Get all brain regions simultaneously recorded in this recording session
    rec_region = glob(join(save_path, 'HMM', f'{BEHAVIOR}',
                      f'{CONTROL}', 'state_mat', f'{this_rec[:20]}*'))

    # Load in which trials were 5-HT stimulated
    if BEHAVIOR == 'Task':
        eid = one.search(subject=this_rec[:9], date=this_rec[10:20])[0]
        trials = load_trials(eid, laser_stimulation=True, one=one)
        n_subplots = len(rec_region) + 1
    else:
        n_subplots = len(rec_region)

    # Plot
    f, axs = plt.subplots(2, n_subplots, dpi=dpi, figsize=(1.5*len(rec_region), 3.5),
                          sharey=True)
    if len(rec_region) == 1:
        axs = np.vstack(axs)
    for ii in range(len(rec_region)):
        state_mat = np.load(f'{rec_region[ii]}')
        n_states = np.unique(state_mat).shape[0]

        axs[0, ii].imshow(state_mat[int(state_mat.shape[0]/2):],
                          aspect='auto', cmap=ListedColormap(sns.color_palette(CMAP, n_states)),
                          vmin=0, vmax=n_states-1,
                          extent=(-PRE_TIME, POST_TIME, 1, state_mat.shape[0]/2), interpolation=None)
        axs[0, ii].plot([0, 0], [1, state_mat.shape[0]/2], ls='--', color='k', lw=0.75)
        axs[0, ii].set(yticks=[1, 50], xticks=[],
                       title=f'{split(rec_region[ii])[1][29:-4]}')
        axs[1, ii].imshow(state_mat[:int(state_mat.shape[0]/2)],
                          aspect='auto', cmap=ListedColormap(sns.color_palette(CMAP, n_states)),
                          vmin=0, vmax=n_states-1,
                          extent=(-PRE_TIME, POST_TIME, 1, state_mat.shape[0]/2), interpolation=None)
        axs[1, ii].plot([0, 0], [1, state_mat.shape[0]/2], ls='--', color='k', lw=0.75)
        axs[1, ii].set(yticks=[1, state_mat.shape[0]/2], xticks=[-1, 0, 1, 2, 3, 4])

        f.text(0.04, 0.5, 'Trials', ha='center', rotation=90)

    if BEHAVIOR == 'Task':
        axs[-1].imshow(np.vstack((trials['laser_stimulation'],
                                  (trials['probabilityLeft'] == 0.2).astype(int) + 2)).T,
                       extent=(0, 20, 1, state_mat.shape[1]),
                       cmap=ListedColormap([[0.5, 0.5, 0.5], [0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    f.text(0.5, 0.02, 'Time from stimulation start (s)', ha='center')
    sns.despine(trim=True)
    plt.subplots_adjust(bottom=0.19, left=0.1, right=0.95)

    plt.savefig(join(fig_path, f'{this_rec[:20]}.jpg'), dpi=600)
    plt.close(f)
