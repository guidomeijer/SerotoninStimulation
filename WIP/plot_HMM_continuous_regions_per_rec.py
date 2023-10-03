# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:42:23 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join, split
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from stim_functions import (paths, query_ephys_sessions, load_passive_opto_times, init_one,
                            high_level_regions, figure_style, N_STATES_CONT)
one = init_one()

# Settings
CMAP = 'Set2'
PRE_TIME = 1
POST_TIME = 4
BIN_SIZE = 0.1

# Get paths
f_path, s_path = paths()
save_path = join(s_path, 'HMM', 'PassiveContinuous')
fig_path = join(f_path, 'Extra plots', 'State', 'Simultaneous regions', 'PassiveContinuous')

# Get all files
all_files = glob(join(save_path, '*zhat.npy'))

# Get all recordings
all_rec = np.unique([split(i)[1][:20] for i in all_files])

colors, dpi = figure_style()
for i, this_rec in enumerate(all_rec):

    # Get session info
    subject = this_rec[:9]
    date = this_rec[10:20]
    eid = one.search(subject=subject, date_range=date)[0]
    opto_times, _ = load_passive_opto_times(eid, one=one)

    # Skip long opto sessions for now
    if np.mean(np.diff(opto_times)) > 20:
        continue

    # Get all brain regions simultaneously recorded in this recording session
    rec_region = glob(join(save_path, f'{this_rec[:20]}*zhat.npy'))

    f, axs = plt.subplots(1, len(rec_region), dpi=dpi, figsize=(1.75*len(rec_region), 3.5),
                          sharey=True)
    if len(rec_region) == 1:
        axs = [axs]
    for ii, this_region in enumerate(rec_region):

        # Load in HMM output
        states = np.load(this_region)
        time_ax = np.load(this_region[:-8] + 'time.npy')
        rel_time = np.arange(-PRE_TIME + BIN_SIZE/2, (POST_TIME - BIN_SIZE/2) + BIN_SIZE, BIN_SIZE)

        state_mat = []
        for t, this_time in enumerate(opto_times):
            state_mat.append(states[(time_ax >= this_time - PRE_TIME)
                             & (time_ax <= this_time + POST_TIME)])
        state_mat = np.vstack(state_mat)

        axs[ii].imshow(state_mat,
                       aspect='auto', cmap=ListedColormap(sns.color_palette(CMAP, N_STATES_CONT)),
                       vmin=0, vmax=N_STATES_CONT-1,
                       extent=(-PRE_TIME, POST_TIME, 1, state_mat.shape[0]), interpolation=None)
        axs[ii].plot([0, 0], [1, state_mat.shape[0]], ls='--', color='k', lw=0.75)
        axs[ii].set(yticks=[1, state_mat.shape[0]], xticks=[-1, 0, 1, 2, 3, 4],
                    title=f'{split(this_region)[1][29:-9]}')
    f.text(0.04, 0.5, 'Trials', ha='center', rotation=90)
    f.text(0.5, 0.02, 'Time from stimulation start (s)', ha='center')
    sns.despine(trim=True)
    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.95)
    plt.savefig(join(fig_path, f'{this_rec[:20]}.jpg'))
    plt.close(f)
